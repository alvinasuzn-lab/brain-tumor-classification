
import os
from pathlib import Path

import cv2
import gdown
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# -------------------- PAGE SETTINGS --------------------

st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Brain Tumor MRI Classification")
st.write(
    "Upload a brain MRI image to predict its class "
    "and view a Grad-CAM visualization."
)

# -------------------- MODEL SETTINGS --------------------

MODEL_NAME = "best_brain_tumor_finetuned.keras"
MODEL_ID = "1sO3yNwEIu9oCfP3W_wlC6r07fhv0hEYI"

MODEL_DIR = Path.home() / ".brain_tumor_model"
MODEL_PATH = MODEL_DIR / MODEL_NAME

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary",
]

IMG_SIZE = (224, 224)


# -------------------- LOAD MODEL --------------------

@st.cache_resource
def load_model():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size < 1_000_000:
        url = f"https://drive.google.com/uc?id={MODEL_ID}"

        result = gdown.download(
            url=url,
            output=str(MODEL_PATH),
            quiet=False,
            resume=False,
        )

        if not result or not MODEL_PATH.exists():
            raise RuntimeError(
                "Model download failed. Check that the Google Drive "
                "file is shared as Anyone with the link — Viewer."
            )

    if MODEL_PATH.stat().st_size < 1_000_000:
        raise RuntimeError(
            "The downloaded model file is too small. "
            "Google Drive may have returned an error page instead."
        )

    with open(MODEL_PATH, "rb") as f:
        header = f.read(100)

    if b"Git LFS pointer" in header:
        raise RuntimeError(
            "A Git LFS pointer was downloaded instead of the model."
        )

    return tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False,
    )


# -------------------- IMAGE PREPROCESSING --------------------

def preprocess_image(image):
    image = image.convert("RGB")
    resized = image.resize(IMG_SIZE)

    array = np.asarray(resized, dtype=np.float32)
    batch = np.expand_dims(array, axis=0)

    return batch


# -------------------- GRAD-CAM --------------------

def generate_gradcam(model, image_batch, class_index):
    try:
        # The ResNet50 output is the final convolutional feature map
        # before global average pooling and the classification layers.
        resnet_layer = model.get_layer("resnet50")

        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[
                resnet_layer.output,
                model.output,
            ],
        )

        image_tensor = tf.convert_to_tensor(
            image_batch,
            dtype=tf.float32,
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(
                image_tensor,
                training=False,
            )

            class_score = predictions[:, class_index]

        gradients = tape.gradient(class_score, conv_outputs)

        if gradients is None:
            raise RuntimeError(
                "Gradients could not be calculated for this model."
            )

        # Average gradients across the spatial dimensions.
        weights = tf.reduce_mean(
            gradients,
            axis=(1, 2),
        )

        conv_outputs = conv_outputs[0]
        weights = weights[0]

        # Weighted combination of feature maps.
        heatmap = tf.reduce_sum(
            conv_outputs * weights,
            axis=-1,
        )

        # Keep positive activations and normalize.
        heatmap = tf.maximum(heatmap, 0)
        maximum = tf.reduce_max(heatmap)

        heatmap = heatmap / (maximum + tf.keras.backend.epsilon())
        heatmap = heatmap.numpy()

        # Resize heatmap to match the original MRI dimensions.
        heatmap = cv2.resize(
            heatmap,
            (IMG_SIZE[1], IMG_SIZE[0]),
            interpolation=cv2.INTER_LINEAR,
        )

        # IMPORTANT: Convert grayscale heatmap into a colour heatmap.
        heatmap_uint8 = np.uint8(
            np.clip(heatmap, 0, 1) * 255
        )

        heatmap_color = cv2.applyColorMap(
            heatmap_uint8,
            cv2.COLORMAP_JET,
        )

        # OpenCV uses BGR; Streamlit expects RGB.
        heatmap_color = cv2.cvtColor(
            heatmap_color,
            cv2.COLOR_BGR2RGB,
        )

        # Blend colour heatmap with the original MRI.
        original = np.uint8(
            np.clip(image_batch[0], 0, 255)
        )

        overlay = cv2.addWeighted(
            original,
            0.55,
            heatmap_color,
            0.45,
            0,
        )

        return heatmap_color, overlay

    except Exception as error:
        raise RuntimeError(
            f"Grad-CAM generation failed: {error}"
        ) from error


# -------------------- MAIN APP --------------------

uploaded_file = st.file_uploader(
    "Upload an MRI image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original MRI")
        st.image(image, use_container_width=True)

    with st.spinner("Loading model..."):
        try:
            model = load_model()
        except Exception as error:
            st.error(f"Could not load the model: {error}")
            st.stop()

    image_batch = preprocess_image(image)

    with st.spinner("Classifying MRI..."):
        predictions = model.predict(image_batch, verbose=0)[0]

    predicted_index = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[predicted_index]) * 100

    st.subheader("Prediction")
    st.success(f"Predicted class: **{predicted_class.upper()}**")
    st.metric("Model confidence", f"{confidence:.2f}%")

    st.subheader("Class probabilities")

    for index, class_name in enumerate(CLASS_NAMES):
        st.write(f"**{class_name.title()}** — {predictions[index] * 100:.2f}%")
        st.progress(
            float(np.clip(predictions[index], 0, 1))
        )

    st.divider()
    st.subheader("Grad-CAM Explainability")
    st.write(
        "The coloured regions highlight image features that influence "
        "the model's prediction. Red/yellow indicates stronger "
        "relative activation; blue indicates weaker activation."
    )

    try:
        with st.spinner("Generating colour Grad-CAM..."):
            heatmap_color, overlay = generate_gradcam(
                model,
                image_batch,
                predicted_index,
            )

        col3, col4 = st.columns(2)

        with col3:
            st.write("**Colour heatmap**")
            st.image(
                heatmap_color,
                channels="RGB",
                use_container_width=True,
            )

        with col4:
            st.write("**Heatmap overlaid on MRI**")
            st.image(
                overlay,
                channels="RGB",
                use_container_width=True,
            )

    except Exception as error:
        st.warning(str(error))

    st.caption(
        "Educational/research demonstration only. "
        "This model is not a substitute for professional medical diagnosis."
    )
