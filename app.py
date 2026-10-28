
from pathlib import Path
import io

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Brain MRI Classifier",
    page_icon="🧠",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_NAME = "best_brain_tumor_finetuned.keras"

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary",
]

st.title("🧠 Brain MRI Classification")
st.caption(
    "Deep learning using your fine-tuned ResNet50 model"
)

st.markdown(
    """
    Upload a brain MRI image to obtain a prediction from the
    trained four-class classifier.
    """
)


# =========================================================
# 2. FIND THE MODEL
# =========================================================
def find_model():
    candidates = [
        BASE_DIR / MODEL_NAME,
        BASE_DIR / "models" / MODEL_NAME,
        BASE_DIR / "model" / MODEL_NAME,
    ]

    for path in candidates:
        if path.is_file():
            return path

    return None


model_path = find_model()

if model_path is None:
    st.error("The trained model file could not be found.")

    st.markdown(
        f"""
        Place `{MODEL_NAME}` in the same directory as `app.py`,
        or update the model path to match your deployment setup.
        """
    )
    st.stop()


# =========================================================
# 3. LOAD THE MODEL ONCE
# =========================================================
@st.cache_resource(show_spinner="Loading the trained model...")
def load_model(path, modified_time):
    return tf.keras.models.load_model(
        path,
        compile=False,
    )


try:
    model = load_model(
        str(model_path),
        model_path.stat().st_mtime,
    )
except Exception as error:
    st.error("Unable to load the trained model.")
    st.exception(error)
    st.stop()


# =========================================================
# 4. PREPROCESS THE IMAGE
# =========================================================
def prepare_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))

    image_array = np.asarray(
        image,
        dtype=np.float32,
    )

    return np.expand_dims(image_array, axis=0)


# =========================================================
# 5. CLASSIFY THE MRI
# =========================================================
def predict_image(batch):
    # The existing model includes its own preprocessing.
    predictions = model.predict(
        batch,
        verbose=0,
    )[0]

    if len(predictions) != len(CLASS_NAMES):
        raise ValueError(
            "The model output does not match the four classes."
        )

    predicted_index = int(np.argmax(predictions))

    return (
        CLASS_NAMES[predicted_index],
        float(predictions[predicted_index]),
        predictions,
        predicted_index,
    )


# =========================================================
# 6. GRAD-CAM GENERATION
# =========================================================
def generate_gradcam(batch, class_index):
    """
    Generate an approximate grayscale Grad-CAM visualization.

    This function uses the trained ResNet50 backbone and the
    existing trained classification head.
    """
    try:
        backbone = model.get_layer("resnet50")

        layer_names = {
            layer.name for layer in backbone.layers
        }

        if "conv4_block6_out" in layer_names:
            target_layer = "conv4_block6_out"
        elif "conv5_block3_out" in layer_names:
            target_layer = "conv5_block3_out"
        else:
            return None, None

        feature_model = tf.keras.Model(
            inputs=backbone.input,
            outputs=[
                backbone.get_layer(target_layer).output,
                backbone.output,
            ],
        )

        gap = model.get_layer("gap")
        dropout = model.get_layer("dropout")
        dense = model.get_layer("dense_features")
        output_layer = model.get_layer("class_predictions")

        input_tensor = preprocess_input(
            tf.convert_to_tensor(
                batch,
                dtype=tf.float32,
            )
        )

        with tf.GradientTape() as tape:
            conv_features, backbone_output = feature_model(
                input_tensor,
                training=False,
            )

            pooled = gap(backbone_output)
            dropped = dropout(pooled, training=False)
            features = dense(dropped)

            logits = tf.linalg.matmul(
                features,
                output_layer.kernel,
            )

            if output_layer.use_bias:
                logits = logits + output_layer.bias

            class_score = logits[:, class_index]

        gradients = tape.gradient(
            class_score,
            conv_features,
        )

        if gradients is None:
            return None, None

        weights = tf.reduce_mean(
            gradients,
            axis=(1, 2),
            keepdims=True,
        )

        cam = tf.reduce_sum(
            weights * conv_features,
            axis=-1,
        )[0]

        cam = tf.nn.relu(cam).numpy()

        if (
            not np.isfinite(cam).all()
            or cam.max() <= 1e-8
        ):
            return None, None

        cam = cam / (cam.max() + 1e-8)

        cam = cv2.resize(
            cam,
            (224, 224),
            interpolation=cv2.INTER_LINEAR,
        )

        heatmap = np.uint8(
            np.clip(cam, 0, 1) * 255
        )

        original = batch[0].astype(np.uint8)

        gray_mri = cv2.cvtColor(
            original,
            cv2.COLOR_RGB2GRAY,
        )

        overlay = cv2.addWeighted(
            gray_mri,
            0.65,
            heatmap,
            0.35,
            0,
        )

        return heatmap, overlay

    except Exception as error:
        st.warning(
            f"Grad-CAM could not be generated: {error}"
        )
        return None, None


# =========================================================
# 7. UPLOAD SECTION
# =========================================================
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

        show_gradcam = st.checkbox(
            "Generate grayscale Grad-CAM",
            value=False,
        )

        if st.button(
            "Analyze MRI",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner("Analyzing MRI..."):
                batch = prepare_image(image)

                (
                    predicted_class,
                    confidence,
                    scores,
                    predicted_index,
                ) = predict_image(batch)

            # Show prediction before generating Grad-CAM.
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
                "The score is the model's output, not a medical "
                "diagnosis or a guarantee of accuracy."
            )

            with st.expander("View scores for all classes"):
                for name, score in zip(
                    CLASS_NAMES,
                    scores,
                ):
                    st.write(
                        f"**{name.capitalize()}**: "
                        f"{float(score) * 100:.2f}%"
                    )

                    st.progress(
                        float(np.clip(score, 0, 1))
                    )

            # Optional Grad-CAM.
            if show_gradcam:
                with st.spinner(
                    "Generating grayscale Grad-CAM..."
                ):
                    heatmap, overlay = generate_gradcam(
                        batch,
                        predicted_index,
                    )

                st.divider()
                st.subheader("Grad-CAM Visualization")

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
                            caption="Grayscale activation",
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

                    output = io.BytesIO()

                    Image.fromarray(overlay).save(
                        output,
                        format="PNG",
                    )

                    st.download_button(
                        "Download Grad-CAM image",
                        data=output.getvalue(),
                        file_name="brain_mri_gradcam.png",
                        mime="image/png",
                    )

                    st.info(
                        "Grad-CAM shows approximate regions that "
                        "influenced the prediction. It does not "
                        "identify exact tumour boundaries."
                    )

                else:
                    st.warning(
                        "The prediction succeeded, but Grad-CAM "
                        "was unavailable for this image."
                    )

    except Exception as error:
        st.error("An error occurred while processing the image.")
        st.exception(error)


# =========================================================
# 8. FOOTER
# =========================================================
st.divider()

st.caption(
    "Academic project | AI-assisted MRI classification | "
    "Not intended for clinical diagnosis"
)
