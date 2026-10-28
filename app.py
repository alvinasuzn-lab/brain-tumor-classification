
from pathlib import Path
import io

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Brain Tumor MRI Classification")
st.write(
    "Upload a brain MRI image to classify it into one of four "
    "categories. The Grad-CAM visualization is an approximate "
    "model explanation, not a confirmed tumor boundary."
)

# --------------------------------------------------
# MODEL CONFIGURATION
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_FILENAME = "best_brain_tumor_finetuned.keras"

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary",
]


# --------------------------------------------------
# FIND MODEL FILE
# --------------------------------------------------
def find_model_path():
    possible_paths = [
        BASE_DIR / MODEL_FILENAME,
        BASE_DIR / "models" / MODEL_FILENAME,
        BASE_DIR / "model" / MODEL_FILENAME,
    ]

    for path in possible_paths:
        if path.is_file():
            return path

    return None


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model(model_path_string):
    return tf.keras.models.load_model(
        model_path_string,
        compile=False,
    )


model_path = find_model_path()

if model_path is None:
    st.error("The trained model file was not found.")

    st.markdown(
        """
        **How to fix this**

        1. Open your GitHub repository:
           `brain-tumor-classification`
        2. Upload `best_brain_tumor_finetuned.keras`.
        3. Place it in the same folder as `app.py`.
        4. Commit the uploaded file and wait for Streamlit Cloud
           to redeploy.

        Your repository should contain:

        - `app.py`
        - `requirements.txt`
        - `best_brain_tumor_finetuned.keras`
        """
    )

    st.stop()

try:
    model = load_model(str(model_path))
except Exception as exc:
    st.error("The model file exists, but TensorFlow could not load it.")
    st.exception(exc)
    st.stop()


# --------------------------------------------------
# PREPROCESS MRI IMAGE
# --------------------------------------------------
def prepare_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))

    image_array = np.asarray(image, dtype=np.float32)
    batch = np.expand_dims(image_array, axis=0)

    return batch


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
def predict_mri(image):
    batch = prepare_image(image)
    predictions = model.predict(batch, verbose=0)[0]

    if len(predictions) != len(CLASS_NAMES):
        raise ValueError(
            f"Expected 4 output scores, but received "
            f"{len(predictions)}."
        )

    predicted_index = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[predicted_index])

    return predicted_class, confidence, predictions, batch


# --------------------------------------------------
# GRAD-CAM
# --------------------------------------------------
def generate_gradcam(batch, class_index):
    """
    Generates a grayscale Grad-CAM explanation using the
    trained ResNet50 feature extractor.

    Returns:
        grayscale activation map, grayscale MRI overlay

    Returns (None, None) if the model architecture does not
    support this Grad-CAM implementation.
    """
    try:
        base_model = model.get_layer("resnet50")

        # Locate a convolutional feature layer.
        layer_names = {layer.name for layer in base_model.layers}

        if "conv4_block6_out" in layer_names:
            target_layer_name = "conv4_block6_out"
        elif "conv5_block3_out" in layer_names:
            target_layer_name = "conv5_block3_out"
        else:
            return None, None

        feature_model = tf.keras.Model(
            inputs=base_model.input,
            outputs=[
                base_model.get_layer(target_layer_name).output,
                base_model.output,
            ],
        )

        # Follow the trained classification head.
        gap_layer = model.get_layer("gap")
        dropout_layer = model.get_layer("dropout")
        dense_layer = model.get_layer("dense_features")
        output_layer = model.get_layer("class_predictions")

        # Grad-CAM needs the gradients of the chosen class score
        # with respect to the convolutional feature maps.
        rgb_batch = tf.convert_to_tensor(batch, dtype=tf.float32)
        processed_batch = preprocess_input(rgb_batch)

        with tf.GradientTape() as tape:
            conv_outputs, backbone_output = feature_model(
                processed_batch,
                training=False,
            )

            tape.watch(conv_outputs)

            x = gap_layer(backbone_output)
            x = dropout_layer(x, training=False)
            x = dense_layer(x)

            logits = tf.linalg.matmul(
                x,
                output_layer.kernel,
            )

            if output_layer.use_bias:
                logits = logits + output_layer.bias

            class_score = logits[:, class_index]

        gradients = tape.gradient(class_score, conv_outputs)

        if gradients is None:
            return None, None

        # Average gradients over the spatial dimensions.
        weights = tf.reduce_mean(
            gradients,
            axis=(1, 2),
            keepdims=True,
        )

        cam = tf.reduce_sum(
            weights * conv_outputs,
            axis=-1,
        )[0]

        cam = tf.nn.relu(cam).numpy()

        if not np.isfinite(cam).all() or cam.max() <= 1e-8:
            return None, None

        # Normalize activation.
        cam = cam / (cam.max() + 1e-8)

        # Resize to the original 224 x 224 input dimensions.
        cam = cv2.resize(
            cam,
            (224, 224),
            interpolation=cv2.INTER_CUBIC,
        )

        cam = np.clip(cam, 0, 1)

        # Improve grayscale contrast while preserving activation
        # ordering. This does not make localization more accurate.
        positive_values = cam[cam > 0]

        if positive_values.size > 10:
            low = np.percentile(positive_values, 5)
            high = np.percentile(positive_values, 99)

            if high > low:
                cam = np.clip((cam - low) / (high - low), 0, 1)

        # Grayscale activation map.
        heatmap_gray = np.uint8(cam * 255)

        # Grayscale MRI background.
        original = batch[0].astype(np.uint8)
        original_gray = cv2.cvtColor(
            original,
            cv2.COLOR_RGB2GRAY,
        )

        # Blend grayscale MRI with grayscale activation map.
        overlay_gray = cv2.addWeighted(
            original_gray,
            0.65,
            heatmap_gray,
            0.35,
            0,
        )

        return heatmap_gray, overlay_gray

    except Exception as exc:
        st.warning(f"Grad-CAM could not be generated: {exc}")
        return None, None


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a brain MRI image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Uploaded MRI")
        st.image(
            image,
            caption="Original MRI image",
            width=400,
        )

        if st.button("Classify MRI", type="primary"):
            with st.spinner("Analyzing MRI..."):
                predicted_class, confidence, scores, batch = (
                    predict_mri(image)
                )

                heatmap, overlay = generate_gradcam(
                    batch,
                    int(np.argmax(scores)),
                )

            st.divider()
            st.subheader("Classification Result")

            st.success(
                f"Predicted category: {predicted_class.upper()}"
            )

            st.metric(
                "Model score",
                f"{confidence * 100:.2f}%",
            )

            st.caption(
                "This score is the model's output, not a medical "
                "diagnosis or a guarantee of accuracy."
            )

            st.subheader("Scores for Each Category")

            for name, score in zip(CLASS_NAMES, scores):
                st.write(f"**{name.capitalize()}**")
                st.progress(
                    float(np.clip(score, 0.0, 1.0)),
                    text=f"{score * 100:.2f}%",
                )

            st.divider()
            st.subheader("Grayscale Grad-CAM Explanation")

            if heatmap is not None and overlay is not None:
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.image(
                        image,
                        caption="Original MRI",
                        use_container_width=True,
                    )

                with col2:
                    st.image(
                        heatmap,
                        caption="Grayscale activation map",
                        clamp=True,
                        use_container_width=True,
                    )

                with col3:
                    st.image(
                        overlay,
                        caption="Grayscale overlay",
                        clamp=True,
                        use_container_width=True,
                    )

                output_image = Image.fromarray(overlay)
                buffer = io.BytesIO()
                output_image.save(buffer, format="PNG")

                st.download_button(
                    label="Download grayscale Grad-CAM",
                    data=buffer.getvalue(),
                    file_name="brain_mri_gradcam.png",
                    mime="image/png",
                )

                st.info(
                    "Bright areas indicate regions that influenced "
                    "the model's score. Grad-CAM is approximate and "
                    "does not establish the exact tumor location."
                )
            else:
                st.warning(
                    "The classification completed, but a reliable "
                    "Grad-CAM map could not be generated for this "
                    "image and model architecture."
                )

    except Exception as exc:
        st.error("Something went wrong while processing the MRI.")
        st.exception(exc)
