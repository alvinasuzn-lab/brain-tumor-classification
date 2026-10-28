
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

from PIL import Image
from pathlib import Path


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="wide"
)

# This order must match the class_indices used during training.
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

MODEL_PATH = Path(__file__).parent / "best_brain_tumor_finetuned.keras"
IMG_SIZE = (224, 224)


# --------------------------------------------------
# LOAD MODEL AND GRAD-CAM COMPONENTS
# --------------------------------------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. "
            "Place best_brain_tumor_finetuned.keras beside app.py."
        )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    # Existing layers in your trained model.
    augmentation_layer = model.get_layer("data_augmentation")
    resnet = model.get_layer("resnet50")
    gap_layer = model.get_layer("gap")
    dropout_layer = model.get_layer("dropout")
    dense_layer = model.get_layer("dense_features")
    classifier_layer = model.get_layer("class_predictions")

    # Final output activation of the final ResNet50 residual block.
    conv_layer = resnet.get_layer("conv5_block3_out")

    # Reuse the original model's input-to-augmentation path.
    preprocessing_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=augmentation_layer.output
    )

    # Extract final convolutional activations from ResNet50.
    feature_model = tf.keras.Model(
        inputs=resnet.inputs,
        outputs=conv_layer.output
    )

    return (
        model,
        preprocessing_model,
        feature_model,
        gap_layer,
        dropout_layer,
        dense_layer,
        classifier_layer
    )


# --------------------------------------------------
# PREPARE MRI IMAGE
# --------------------------------------------------

def prepare_image(image):
    image = image.convert("RGB").resize(IMG_SIZE)
    original = np.asarray(image, dtype=np.uint8)

    # Match the input format used by the original model.
    batch = original.astype(np.float32)[None, ...]

    return original, batch


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

def predict_mri(image_batch, model):
    probabilities = model(
        tf.convert_to_tensor(image_batch, dtype=tf.float32),
        training=False
    ).numpy()[0]

    if probabilities.shape[0] != len(CLASS_NAMES):
        raise ValueError(
            f"Expected {len(CLASS_NAMES)} outputs, "
            f"but model returned {probabilities.shape[0]}."
        )

    # Verify that the outputs look like probabilities.
    if (
        not np.all(np.isfinite(probabilities))
        or np.any(probabilities < -1e-5)
        or np.any(probabilities > 1.00001)
        or not np.isclose(np.sum(probabilities), 1.0, atol=1e-3)
    ):
        raise ValueError(
            "Model outputs do not look like softmax probabilities. "
            "Check the classifier activation before displaying confidence."
        )

    predicted_index = int(np.argmax(probabilities))

    scores = {
        CLASS_NAMES[i]: float(probabilities[i])
        for i in range(len(CLASS_NAMES))
    }

    return scores, predicted_index


# --------------------------------------------------
# GRAD-CAM
# --------------------------------------------------

def make_gradcam(
    image_batch,
    preprocessing_model,
    feature_model,
    gap_layer,
    dropout_layer,
    dense_layer,
    classifier_layer,
    class_index
):
    image_tensor = tf.convert_to_tensor(
        image_batch,
        dtype=tf.float32
    )

    # IMPORTANT:
    # Follow the same model preprocessing path as prediction.
    # Do not apply preprocess_input separately here.
    backbone_input = preprocessing_model(
        image_tensor,
        training=False
    )

    with tf.GradientTape() as tape:
        conv_outputs = feature_model(
            backbone_input,
            training=False
        )

        # Reconnect the final convolutional activations to
        # the original trained classification head.
        x = gap_layer(conv_outputs)
        x = dropout_layer(x, training=False)
        x = dense_layer(x)
        class_predictions = classifier_layer(x)

        # Target the predicted class score.
        class_score = class_predictions[:, class_index]

    gradients = tape.gradient(class_score, conv_outputs)

    if gradients is None:
        raise ValueError(
            "Could not calculate Grad-CAM gradients."
        )

    # Average gradients across spatial dimensions.
    channel_weights = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    # Weighted sum of the feature maps.
    heatmap = tf.reduce_sum(
        conv_outputs * channel_weights[:, None, None, :],
        axis=-1
    )[0]

    # Standard Grad-CAM uses positive class-supporting responses.
    heatmap = tf.nn.relu(heatmap)

    heatmap = heatmap.numpy()
    max_value = float(np.max(heatmap))

    if not np.isfinite(max_value) or max_value <= 1e-8:
        raise ValueError(
            "Grad-CAM has no meaningful positive activation "
            "for this target class on this image."
        )

    # Normalize without moving or reshaping activation locations.
    heatmap = heatmap / max_value

    # Upsample the coarse 7x7 map for display only.
    heatmap = cv2.resize(
        heatmap,
        IMG_SIZE,
        interpolation=cv2.INTER_LINEAR
    )
    heatmap = np.clip(heatmap, 0, 1)

    return heatmap


# --------------------------------------------------
# HEATMAP COLOURING AND OVERLAY
# --------------------------------------------------

def create_visualizations(original, heatmap):
    height, width = original.shape[:2]

    # Resize to the exact original display dimensions.
    heatmap_resized = cv2.resize(
        heatmap,
        (width, height),
        interpolation=cv2.INTER_LINEAR
    )

    heatmap_uint8 = np.uint8(
        np.clip(heatmap_resized, 0, 1) * 255
    )

    # OpenCV colour maps return BGR; convert to RGB for Streamlit.
    heatmap_bgr = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    heatmap_rgb = cv2.cvtColor(
        heatmap_bgr,
        cv2.COLOR_BGR2RGB
    )

    # Display blend only; no spatial shifting or lesion-based editing.
    overlay = cv2.addWeighted(
        original,
        0.60,
        heatmap_rgb,
        0.40,
        0
    )

    return heatmap_rgb, overlay


# --------------------------------------------------
# STREAMLIT INTERFACE
# --------------------------------------------------

st.title("🧠 Brain Tumor Classification from MRI Images")

st.write(
    "Upload an MRI image to view the model's predicted class, "
    "confidence scores, and Grad-CAM visualization."
)

st.warning(
    "Educational project demonstration only. This model is not "
    "validated for medical diagnosis. Grad-CAM highlights image "
    "regions influencing the model; it does not confirm or segment a tumour."
)

uploaded_file = st.file_uploader(
    "Upload a brain MRI image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as error:
        st.error(f"Could not open this image: {error}")
        st.stop()

    st.subheader("Uploaded MRI")
    st.image(image, use_container_width=True)

    if st.button("Classify MRI", type="primary"):
        try:
            with st.spinner("Analyzing image and generating Grad-CAM..."):
                (
                    model,
                    preprocessing_model,
                    feature_model,
                    gap_layer,
                    dropout_layer,
                    dense_layer,
                    classifier_layer
                ) = load_model()

                original, image_batch = prepare_image(image)

                scores, predicted_index = predict_mri(
                    image_batch,
                    model
                )

                heatmap = make_gradcam(
                    image_batch,
                    preprocessing_model,
                    feature_model,
                    gap_layer,
                    dropout_layer,
                    dense_layer,
                    classifier_layer,
                    predicted_index
                )

                colored_heatmap, overlay = create_visualizations(
                    original,
                    heatmap
                )

            predicted_class = CLASS_NAMES[predicted_index]
            confidence = scores[predicted_class]

            st.subheader("Prediction Results")
            st.success(
                f"Predicted class: {predicted_class} "
                f"(model confidence: {confidence:.2%})"
            )

            st.write("Model confidence scores:")
            st.bar_chart(scores)

            st.subheader("Grad-CAM Visualizations")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.caption("Original MRI")
                st.image(original, use_container_width=True)

            with col2:
                st.caption("Grad-CAM Heatmap")
                st.image(colored_heatmap, use_container_width=True)

            with col3:
                st.caption("Heatmap Overlay")
                st.image(overlay, use_container_width=True)

            st.caption(
                "Warm colours represent stronger positive Grad-CAM "
                "responses for the predicted class. They are not a "
                "validated tumour boundary or a probability map of "
                "tumour location."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")
            st.exception(error)
