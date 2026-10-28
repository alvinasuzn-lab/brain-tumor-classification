
import os
from pathlib import Path

import cv2
import gdown
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Brain Tumor MRI Classification")
st.write("Upload an MRI image to classify it and view Grad-CAM.")

MODEL_ID = "1sO3yNwEIu9oCfP3W_wlC6r07fhv0hEYI"
MODEL_NAME = "best_brain_tumor_finetuned.keras"
MODEL_DIR = Path.home() / ".brain_tumor_model"
MODEL_PATH = MODEL_DIR / MODEL_NAME

IMG_SIZE = (224, 224)
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]


@st.cache_resource
def load_model():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size < 1_000_000:
        result = gdown.download(
            id=MODEL_ID,
            output=str(MODEL_PATH),
            quiet=False,
            resume=False,
        )

        if not result or not MODEL_PATH.exists():
            raise RuntimeError(
                "Model download failed. Set Google Drive sharing to "
                "'Anyone with the link — Viewer'."
            )

    if MODEL_PATH.stat().st_size < 1_000_000:
        raise RuntimeError("Downloaded model file is invalid or incomplete.")

    with open(MODEL_PATH, "rb") as f:
        if b"Git LFS pointer" in f.read(200):
            raise RuntimeError("The file is a Git LFS pointer, not the model.")

    model = tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False,
    )

    if model.output_shape[-1] != 4:
        raise RuntimeError(
            f"Expected 4 class outputs, received {model.output_shape}."
        )

    return model


def preprocess_image(image):
    image = image.convert("RGB").resize(IMG_SIZE)
    array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def generate_gradcam(model, image_batch, class_index):
    # Locate the nested ResNet50 feature extractor.
    resnet = model.get_layer("resnet50")

    # Find the last convolutional layer inside ResNet50.
    last_conv = None
    for layer in reversed(resnet.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer
            break

    if last_conv is None:
        raise RuntimeError("Could not find a convolutional layer in ResNet50.")

    # Return both the last convolutional activations and the ResNet output
    # from the SAME forward pass, keeping the gradient path connected.
    feature_extractor = tf.keras.Model(
        inputs=resnet.input,
        outputs=[last_conv.output, resnet.output],
    )

    resnet_index = model.layers.index(resnet)
    head_layers = model.layers[resnet_index + 1:]

    image_tensor = tf.convert_to_tensor(image_batch, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_features, resnet_output = feature_extractor(
            image_tensor,
            training=False,
        )

        # Explicitly watch the activations so gradients can be calculated.
        tape.watch(conv_features)

        output = resnet_output

        # Apply the original trained classification head.
        for layer in head_layers:
            if isinstance(layer, tf.keras.layers.Dropout):
                output = layer(output, training=False)
            else:
                output = layer(output)

        score = output[:, class_index]

    gradients = tape.gradient(score, conv_features)

    if gradients is None:
        raise RuntimeError(
            "No Grad-CAM gradients were produced. Check the model architecture."
        )

    # Grad-CAM: average gradients to obtain channel weights.
    weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)

    heatmap = tf.reduce_sum(weights * conv_features, axis=-1)[0]
    heatmap = tf.maximum(heatmap, 0)

    maximum = tf.reduce_max(heatmap)
    heatmap = heatmap / (maximum + 1e-8)
    heatmap = heatmap.numpy()

    # Resize and apply a colour map.
    heatmap = cv2.resize(
        heatmap,
        (IMG_SIZE[1], IMG_SIZE[0]),
        interpolation=cv2.INTER_LINEAR,
    )

    heatmap_uint8 = np.uint8(np.clip(heatmap, 0, 1) * 255)
    heatmap_rgb = cv2.cvtColor(
        cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET),
        cv2.COLOR_BGR2RGB,
    )

    original = np.uint8(np.clip(image_batch[0], 0, 255))

    # Make stronger activation regions more visible.
    overlay = cv2.addWeighted(original, 0.50, heatmap_rgb, 0.50, 0)

    return heatmap_rgb, overlay


uploaded = st.file_uploader(
    "Upload brain MRI",
    type=["jpg", "jpeg", "png"],
)

if uploaded is not None:
    try:
        image = Image.open(uploaded).convert("RGB")
    except Exception:
        st.error("Unable to read this image.")
        st.stop()

    image_batch = preprocess_image(image)

    try:
        with st.spinner("Loading trained model..."):
            model = load_model()
    except Exception as e:
        st.error(f"Model loading error: {e}")
        st.stop()

    try:
        with st.spinner("Predicting..."):
            probabilities = np.asarray(
                model.predict(image_batch, verbose=0)
            ).reshape(-1)

        if len(probabilities) != len(CLASS_NAMES):
            st.error("The model output does not match the four class names.")
            st.stop()

        predicted_index = int(np.argmax(probabilities))
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(probabilities[predicted_index]) * 100

    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

    left, right = st.columns(2)

    with left:
        st.subheader("Original MRI")
        st.image(image, use_container_width=True)

    with right:
        st.subheader("Prediction")
        st.success(f"Predicted class: {predicted_class.upper()}")
        st.metric("Confidence", f"{confidence:.2f}%")

        for i, name in enumerate(CLASS_NAMES):
            st.write(f"{name.title()}: {probabilities[i] * 100:.2f}%")
            st.progress(float(np.clip(probabilities[i], 0, 1)))

    st.divider()
    st.subheader("Grad-CAM tumor-region explanation")

    try:
        with st.spinner("Generating Grad-CAM..."):
            heatmap, overlay = generate_gradcam(
                model,
                image_batch,
                predicted_index,
            )

        left, right = st.columns(2)

        with left:
            st.write("**Colour heatmap**")
            st.image(heatmap, channels="RGB", use_container_width=True)

        with right:
            st.write("**Heatmap over MRI**")
            st.image(overlay, channels="RGB", use_container_width=True)

        st.caption(
            "Red/yellow regions indicate stronger relative model activation. "
            "Grad-CAM does not guarantee that these regions are the tumor."
        )

    except Exception as e:
        st.error(f"Grad-CAM error: {e}")

st.caption(
    "Academic demonstration only; not a substitute for medical diagnosis."
)
