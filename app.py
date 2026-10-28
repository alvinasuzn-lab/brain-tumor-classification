
import os
from pathlib import Path

import cv2
import gdown
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Brain Tumor MRI Classification",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Brain Tumor MRI Classification")
st.write(
    "Upload a brain MRI image to classify it and view a "
    "Grad-CAM visualization."
)

IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary",
]

MODEL_ID = "1sO3yNwEIu9oCfP3W_wlC6r07fhv0hEYI"
MODEL_NAME = "best_brain_tumor_finetuned.keras"

MODEL_DIR = Path.home() / ".brain_tumor_model"
MODEL_PATH = MODEL_DIR / MODEL_NAME


# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size < 1_000_000:
        try:
            result = gdown.download(
                id=MODEL_ID,
                output=str(MODEL_PATH),
                quiet=False,
                resume=False,
            )
        except Exception as exc:
            raise RuntimeError(
                "Google Drive download failed. Set the model's sharing "
                "permission to Anyone with the link — Viewer. Details: "
                + str(exc)
            ) from exc

        if not result or not MODEL_PATH.exists():
            raise RuntimeError(
                "Model download failed. Check the Google Drive link "
                "and sharing permissions."
            )

    if MODEL_PATH.stat().st_size < 1_000_000:
        raise RuntimeError(
            "The downloaded model is incomplete or invalid."
        )

    with open(MODEL_PATH, "rb") as file:
        header = file.read(200)

    if b"Git LFS pointer" in header:
        raise RuntimeError(
            "A Git LFS pointer was downloaded instead of the model."
        )

    try:
        model = tf.keras.models.load_model(
            str(MODEL_PATH),
            compile=False,
        )
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow could not load the model. Check the TensorFlow "
            "and Keras versions. Details: " + str(exc)
        ) from exc

    if model.output_shape[-1] != len(CLASS_NAMES):
        raise RuntimeError(
            f"Expected 4 model outputs; received {model.output_shape}."
        )

    return model


# =====================================================
# IMAGE PREPROCESSING
# =====================================================

def preprocess_image(image):
    image = image.convert("RGB").resize(IMG_SIZE)
    array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(array, axis=0)


# =====================================================
# GRAD-CAM
# =====================================================

def generate_gradcam(model, image_batch, class_index):
    try:
        resnet = model.get_layer("resnet50")
    except Exception as exc:
        raise RuntimeError(
            "The model does not contain a layer named 'resnet50'."
        ) from exc

    last_conv = None

    for layer in reversed(resnet.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer
            break

    if last_conv is None:
        raise RuntimeError(
            "No convolutional layer was found inside ResNet50."
        )

    # Build a feature extractor from the same ResNet50 model.
    feature_model = tf.keras.Model(
        inputs=resnet.input,
        outputs=[last_conv.output, resnet.output],
    )

    resnet_index = model.layers.index(resnet)
    classifier_layers = model.layers[resnet_index + 1:]

    if not classifier_layers:
        raise RuntimeError(
            "The model's classification head could not be found."
        )

    image_tensor = tf.convert_to_tensor(
        image_batch,
        dtype=tf.float32,
    )

    with tf.GradientTape() as tape:
        conv_features, resnet_features = feature_model(
            image_tensor,
            training=False,
        )

        tape.watch(conv_features)

        predictions = resnet_features

        for layer in classifier_layers:
            if isinstance(layer, tf.keras.layers.Dropout):
                predictions = layer(predictions, training=False)
            else:
                predictions = layer(predictions)

        score = predictions[:, class_index]

    gradients = tape.gradient(score, conv_features)

    if gradients is None:
        raise RuntimeError(
            "Grad-CAM gradients are unavailable for this model."
        )

    weights = tf.reduce_mean(
        gradients,
        axis=(1, 2),
        keepdims=True,
    )

    heatmap = tf.reduce_sum(
        weights * conv_features,
        axis=-1,
    )[0]

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    heatmap = cv2.resize(
        heatmap,
        (IMG_SIZE[1], IMG_SIZE[0]),
        interpolation=cv2.INTER_LINEAR,
    )

    heatmap_uint8 = np.uint8(
        np.clip(heatmap, 0, 1) * 255
    )

    heatmap_rgb = cv2.cvtColor(
        cv2.applyColorMap(
            heatmap_uint8,
            cv2.COLORMAP_JET,
        ),
        cv2.COLOR_BGR2RGB,
    )

    original = np.uint8(
        np.clip(image_batch[0], 0, 255)
    )

    overlay = cv2.addWeighted(
        original,
        0.60,
        heatmap_rgb,
        0.40,
        0,
    )

    return heatmap_rgb, overlay


# =====================================================
# MAIN APP
# =====================================================

uploaded_file = st.file_uploader(
    "Upload a brain MRI image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    try:
        original_image = Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error("Could not read the uploaded image.")
        st.stop()

    image_batch = preprocess_image(original_image)

    with st.spinner("Loading model..."):
        try:
            model = load_model()
        except Exception as exc:
            st.error(f"Model loading failed: {exc}")
            st.stop()

    with st.spinner("Predicting MRI class..."):
        try:
            probabilities = np.asarray(
                model.predict(image_batch, verbose=0)
            ).reshape(-1)

            if len(probabilities) != len(CLASS_NAMES):
                st.error("The model output does not match the four classes.")
                st.stop()

            predicted_index = int(np.argmax(probabilities))
            predicted_class = CLASS_NAMES[predicted_index]
            confidence = float(probabilities[predicted_index]) * 100

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.stop()

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original MRI")
        st.image(original_image, use_container_width=True)

    with col2:
        st.subheader("Prediction")
        st.success(f"Predicted class: {predicted_class.upper()}")
        st.metric("Confidence", f"{confidence:.2f}%")

        st.write("**Class probabilities**")

        for index, name in enumerate(CLASS_NAMES):
            value = float(probabilities[index])
            st.write(f"{name.title()}: {value * 100:.2f}%")
            st.progress(float(np.clip(value, 0, 1)))

    st.divider()
    st.subheader("Grad-CAM Explainability")

    try:
        with st.spinner("Generating colour heatmap..."):
            heatmap, overlay = generate_gradcam(
                model,
                image_batch,
                predicted_index,
            )

        col3, col4 = st.columns(2)

        with col3:
            st.write("**Colour heatmap**")
            st.image(
                heatmap,
                channels="RGB",
                use_container_width=True,
            )

        with col4:
            st.write("**Heatmap overlay**")
            st.image(
                overlay,
                channels="RGB",
                use_container_width=True,
            )

        st.caption(
            "Grad-CAM highlights regions influencing the prediction. "
            "It does not guarantee accurate tumor localization."
        )

    except Exception as exc:
        st.error(f"Grad-CAM failed: {exc}")

st.caption(
    "For educational and research use only. "
    "Not a substitute for professional medical diagnosis."
)
