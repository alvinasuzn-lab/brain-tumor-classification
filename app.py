
import os
from pathlib import Path

import cv2
import gdown
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Brain Tumor MRI Classification",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Brain Tumor MRI Classification")
st.write(
    "Upload a brain MRI image to obtain a model prediction "
    "and a Grad-CAM visualization."
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


# =========================================================
# 2. DOWNLOAD AND LOAD THE TRAINED MODEL
# =========================================================

@st.cache_resource
def load_model():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Download only if the model is not already available.
    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size < 1_000_000:
        st.write("Downloading the trained model from Google Drive...")

        try:
            result = gdown.download(
                id=MODEL_ID,
                output=str(MODEL_PATH),
                quiet=False,
                resume=False,
            )
        except Exception as error:
            raise RuntimeError(
                "Google Drive download failed. Check the Drive sharing "
                "permission and the gdown version. Details: " + str(error)
            ) from error

        if not result or not MODEL_PATH.exists():
            raise RuntimeError(
                "The model was not downloaded. Set Google Drive access "
                "to Anyone with the link — Viewer."
            )

    # Validate the downloaded file.
    if MODEL_PATH.stat().st_size < 1_000_000:
        raise RuntimeError(
            "The downloaded model is too small to be valid. "
            "Google Drive may have returned an error page."
        )

    with open(MODEL_PATH, "rb") as file:
        header = file.read(200)

    if b"<!DOCTYPE html" in header or b"<html" in header.lower():
        raise RuntimeError(
            "Google Drive returned an HTML page instead of the model. "
            "Check that the shared file can be downloaded."
        )

    try:
        model = tf.keras.models.load_model(
            str(MODEL_PATH),
            compile=False,
        )
    except Exception as error:
        raise RuntimeError(
            "The model file downloaded, but TensorFlow could not load it. "
            "Check TensorFlow/Keras compatibility. Details: " + str(error)
        ) from error

    if model.output_shape[-1] != len(CLASS_NAMES):
        raise RuntimeError(
            f"Expected 4 output classes, but model output is "
            f"{model.output_shape}."
        )

    return model


# =========================================================
# 3. PREPROCESS MRI IMAGE
# =========================================================

def preprocess_image(pil_image):
    resized = pil_image.convert("RGB").resize(IMG_SIZE)

    image_array = np.asarray(resized, dtype=np.float32)

    # Keep pixel values in the 0–255 range, matching the
    # previous app's inference approach.
    image_batch = np.expand_dims(image_array, axis=0)

    return resized, image_batch


# =========================================================
# 4. GENERATE COLOUR GRAD-CAM
# =========================================================

def generate_gradcam(model, image_batch, class_index):
    # This implementation uses the nested ResNet50 and the
    # classification layers after it.
    try:
        resnet_model = model.get_layer("resnet50")
    except Exception as error:
        raise RuntimeError(
            "Could not find the nested layer named 'resnet50'. "
            "Check the model architecture."
        ) from error

    # Find the final convolutional layer automatically.
    conv_layer = None

    for layer in reversed(resnet_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            conv_layer = layer
            break

    if conv_layer is None:
        raise RuntimeError(
            "No Conv2D layer was found inside ResNet50."
        )

    conv_model = tf.keras.Model(
        inputs=resnet_model.inputs,
        outputs=conv_layer.output,
    )

    # Get the classifier layers that follow ResNet50.
    try:
        resnet_position = model.layers.index(resnet_model)
    except ValueError as error:
        raise RuntimeError(
            "Could not identify the ResNet50 position in the model."
        ) from error

    classifier_layers = model.layers[resnet_position + 1:]

    if not classifier_layers:
        raise RuntimeError(
            "No classification layers were found after ResNet50."
        )

    image_tensor = tf.convert_to_tensor(
        image_batch,
        dtype=tf.float32,
    )

    with tf.GradientTape() as tape:
        # Feature maps from the last convolutional layer.
        conv_outputs = conv_model(image_tensor, training=False)

        # Explicitly watch the convolutional activations.
        tape.watch(conv_outputs)

        # Pass activations through the trained classification head.
        predictions = conv_outputs

        for layer in classifier_layers:
            if isinstance(layer, tf.keras.layers.Dropout):
                predictions = layer(predictions, training=False)
            else:
                predictions = layer(predictions)

        class_score = predictions[:, class_index]

    gradients = tape.gradient(class_score, conv_outputs)

    if gradients is None:
        raise RuntimeError(
            "Gradients were not calculated. "
            "The model's architecture may need a custom Grad-CAM method."
        )

    # Average gradients across height and width.
    weights = tf.reduce_mean(
        gradients,
        axis=(1, 2),
        keepdims=True,
    )

    # Weighted combination of convolutional feature maps.
    heatmap = tf.reduce_sum(
        weights * conv_outputs,
        axis=-1,
    )[0]

    heatmap = tf.maximum(heatmap, 0)

    max_value = tf.reduce_max(heatmap)
    heatmap = heatmap / (max_value + 1e-8)

    heatmap = heatmap.numpy()

    # Resize to the image dimensions.
    heatmap = cv2.resize(
        heatmap,
        IMG_SIZE,
        interpolation=cv2.INTER_LINEAR,
    )

    # Convert the grayscale activation map into a COLOUR map.
    heatmap_uint8 = np.uint8(
        np.clip(heatmap, 0, 1) * 255
    )

    heatmap_bgr = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET,
    )

    heatmap_rgb = cv2.cvtColor(
        heatmap_bgr,
        cv2.COLOR_BGR2RGB,
    )

    # Overlay the colour heatmap on the MRI.
    original_rgb = np.uint8(
        np.clip(image_batch[0], 0, 255)
    )

    overlay = cv2.addWeighted(
        original_rgb,
        0.55,
        heatmap_rgb,
        0.45,
        0,
    )

    return heatmap_rgb, overlay


# =========================================================
# 5. USER INTERFACE AND PREDICTION
# =========================================================

uploaded_file = st.file_uploader(
    "Upload a brain MRI image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:

    try:
        original_image = Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error("The uploaded image could not be opened.")
        st.stop()

    resized_image, image_batch = preprocess_image(original_image)

    try:
        with st.spinner("Loading model..."):
            model = load_model()
    except Exception as error:
        st.error(f"Model loading failed: {error}")
        st.stop()

    # ---------------- Prediction ----------------

    try:
        with st.spinner("Analysing MRI image..."):
            probabilities = model.predict(
                image_batch,
                verbose=0,
            )[0]

        probabilities = np.asarray(probabilities).reshape(-1)

        if len(probabilities) != len(CLASS_NAMES):
            st.error(
                f"Expected 4 probabilities, received {len(probabilities)}."
            )
            st.stop()

        predicted_index = int(np.argmax(probabilities))
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(probabilities[predicted_index]) * 100

    except Exception as error:
        st.error(f"Prediction failed: {error}")
        st.stop()

    # ---------------- Display results ----------------

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original MRI")
        st.image(original_image, use_container_width=True)

    with col2:
        st.subheader("Prediction")
        st.success(f"Predicted class: {predicted_class.upper()}")
        st.metric("Model confidence", f"{confidence:.2f}%")

        st.write("**Class probabilities**")

        for index, name in enumerate(CLASS_NAMES):
            probability = float(probabilities[index])
            st.write(f"{name.title()}: {probability * 100:.2f}%")
            st.progress(float(np.clip(probability, 0, 1)))

    # ---------------- Grad-CAM ----------------

    st.divider()
    st.subheader("Grad-CAM Explainability")

    st.write(
        "The heatmap visualizes image regions that influence the "
        "model's prediction. It is an explanatory visualization, "
        "not a medical segmentation or diagnosis."
    )

    try:
        with st.spinner("Generating colour Grad-CAM..."):
            heatmap_rgb, overlay = generate_gradcam(
                model,
                image_batch,
                predicted_index,
            )

        col3, col4 = st.columns(2)

        with col3:
            st.write("**Colour heatmap**")
            st.image(
                heatmap_rgb,
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

    except Exception as error:
        st.error(f"Grad-CAM failed: {error}")

st.caption(
    "For academic demonstration only. This model is not a substitute "
    "for professional medical diagnosis."
)
