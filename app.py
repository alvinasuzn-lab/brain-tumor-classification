from pathlib import Path

import cv2
import gdown
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ---------------- CONFIGURATION ----------------
MODEL_ID = "1sO3yNwEIu9oCfP3W_wlC6r07fhv0hEYI"
MODEL_NAME = "best_brain_tumor_finetuned.keras"
IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary Tumor",
]


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model_and_gradcam():
    model_dir = Path.home() / ".brain_tumor_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / MODEL_NAME

    if not model_path.exists() or model_path.stat().st_size == 0:
        downloaded = gdown.download(
            id=MODEL_ID,
            output=str(model_path),
            quiet=False,
            fuzzy=True,
        )

        if not downloaded or not model_path.exists():
            raise RuntimeError(
                "Model download failed. Check Google Drive sharing permissions."
            )

    model = tf.keras.models.load_model(
        str(model_path), compile=False
    )

    # Access the trained ResNet50 backbone
    resnet = model.get_layer("resnet50")
    last_conv = resnet.get_layer("conv5_block3_3_conv")

    # Return convolutional activations and backbone features
    backbone_grad_model = tf.keras.Model(
        inputs=resnet.inputs,
        outputs=[last_conv.output, resnet.output],
    )

    # Get the actual tensor feeding ResNet50 in the saved model
    resnet_input = resnet._inbound_nodes[0].input_tensors[0]

    conv_maps, backbone_features = backbone_grad_model(
        resnet_input
    )

    # Reuse the original trained classification layers
    x = model.get_layer("gap")(backbone_features)
    x = model.get_layer("dropout")(x, training=False)
    x = model.get_layer("dense_features")(x)
    scores = model.get_layer("class_predictions")(x)

    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[conv_maps, scores],
    )

    return model, grad_model


# ---------------- PREDICTION + GRAD-CAM ----------------
def predict_and_gradcam(model, grad_model, pil_image):
    image = pil_image.convert("RGB").resize(IMG_SIZE)

    image_array = np.asarray(image, dtype=np.float32)
    image_batch = np.expand_dims(image_array, axis=0)
    input_tensor = tf.convert_to_tensor(image_batch)

    # Prediction from the complete saved model
    original_probs = model(
        input_tensor, training=False
    ).numpy()[0]

    predicted_index = int(np.argmax(original_probs))

    # Grad-CAM for the same predicted class
    with tf.GradientTape() as tape:
        conv_outputs, cam_probs = grad_model(
            input_tensor, training=False
        )
        class_score = cam_probs[:, predicted_index]

    gradients = tape.gradient(class_score, conv_outputs)

    if gradients is None:
        raise RuntimeError("Grad-CAM gradients could not be calculated.")

    cam_probs = cam_probs.numpy()[0]

    # Prevent displaying a heatmap if its classifier output differs
    if not np.allclose(original_probs, cam_probs, atol=1e-4):
        raise RuntimeError(
            "Grad-CAM probabilities do not match the model predictions."
        )

    # Global average of gradients gives channel importance weights
    weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
    feature_maps = conv_outputs[0]

    heatmap = tf.reduce_sum(
        feature_maps * weights, axis=-1
    )
    heatmap = tf.maximum(heatmap, 0)

    max_value = tf.reduce_max(heatmap)
    heatmap = tf.where(
        max_value > 0,
        heatmap / (max_value + tf.keras.backend.epsilon()),
        tf.zeros_like(heatmap),
    )

    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, IMG_SIZE)

    # Convert heatmap to a color image
    heatmap_uint8 = np.uint8(
        np.clip(heatmap * 255, 0, 255)
    )
    colored_heatmap = cv2.applyColorMap(
        heatmap_uint8, cv2.COLORMAP_JET
    )
    colored_heatmap = cv2.cvtColor(
        colored_heatmap, cv2.COLOR_BGR2RGB
    )

    original_uint8 = np.uint8(
        np.clip(image_array, 0, 255)
    )

    overlay = cv2.addWeighted(
        original_uint8, 0.6,
        colored_heatmap, 0.4,
        0,
    )

    confidence = float(original_probs[predicted_index])

    return (
        predicted_index,
        confidence,
        original_probs,
        original_uint8,
        heatmap,
        overlay,
    )


# ---------------- STREAMLIT INTERFACE ----------------
st.set_page_config(
    page_title="Brain Tumor MRI Classification",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Brain Tumor MRI Classification")

st.write(
    "Upload an MRI image to view the predicted class, "
    "model confidence, and Grad-CAM visualization."
)

st.warning(
    "For educational and research use only. This model is not a "
    "medical diagnostic tool. Grad-CAM highlights regions that "
    "influence the prediction; it does not outline or confirm a tumour."
)

uploaded_file = st.file_uploader(
    "Upload an MRI image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    try:
        uploaded_image = Image.open(uploaded_file).convert("RGB")

        with st.spinner("Loading model and analyzing MRI..."):
            model, grad_model = load_model_and_gradcam()

            (
                predicted_index,
                confidence,
                probabilities,
                original_image,
                heatmap,
                overlay,
            ) = predict_and_gradcam(
                model, grad_model, uploaded_image
            )

        st.subheader("Prediction")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Predicted class",
                CLASS_NAMES[predicted_index],
            )

        with col2:
            st.metric(
                "Model confidence",
                f"{confidence * 100:.2f}%",
            )

        st.subheader("Class probabilities")

        for name, probability in zip(
            CLASS_NAMES, probabilities
        ):
            st.write(f"**{name}: {probability * 100:.2f}%**")
            st.progress(
                float(np.clip(probability, 0.0, 1.0))
            )

        st.subheader("Grad-CAM visualization")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(
                original_image,
                caption="Resized MRI",
                use_container_width=True,
            )

        with col2:
            st.image(
                heatmap,
                caption="Grad-CAM heatmap",
                clamp=True,
                use_container_width=True,
            )

        with col3:
            st.image(
                overlay,
                caption="Grad-CAM overlay",
                use_container_width=True,
            )

        st.caption(
            "A high confidence score does not guarantee correctness. "
            "The heatmap is not a tumour segmentation."
        )

    except Exception as exc:
        st.error(f"Could not analyze this image: {exc}")

else:
    st.info("Upload a JPG, JPEG, or PNG MRI image to begin.")
