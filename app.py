
import os
from pathlib import Path

import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import gdown

from PIL import Image

st.set_page_config(
    page_title="Brain Tumor MRI Classification",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Brain Tumor MRI Classification")
st.write("Upload an MRI image to view the prediction and Grad-CAM.")

st.warning(
    "Educational research tool only. Not validated for medical diagnosis."
)

CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary Tumor",
]

IMG_SIZE = (224, 224)
MODEL_ID = "1sO3yNwEIu9oCfP3W_wlC6r07fhv0hEYI"
MODEL_NAME = "best_brain_tumor_finetuned.keras"
MODEL_PATH = Path.home() / ".brain_tumor_model" / MODEL_NAME


@st.cache_resource
def load_model():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size < 1_000_000:
        result = gdown.download(
            id=MODEL_ID,
            output=str(MODEL_PATH),
            quiet=False,
        )

        if result is None or not MODEL_PATH.exists():
            raise RuntimeError(
                "Model download failed. Check Google Drive sharing."
            )

    return tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False,
    )


def make_gradcam_model(model):
    """
    Create a feature extractor connected to the full model input.
    The convolutional activation and prediction are computed in
    one forward pass through the original model graph.
    """
    backbone = model.get_layer("resnet50")
    last_conv = backbone.get_layer("conv5_block3_3_conv")

    # Reuse the actual layer tensors from the original model graph.
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[
            last_conv.output,
            model.output,
        ],
    )
    return grad_model


def predict_mri(image, model, alpha=0.40):
    original = np.array(image.convert("RGB"))

    resized = cv2.resize(
        original,
        IMG_SIZE,
        interpolation=cv2.INTER_AREA,
    )

    # This keeps RGB pixels in the 0–255 range.
    # Confirm against the original training preprocessing.
    batch = np.expand_dims(
        resized.astype(np.float32),
        axis=0,
    )

    grad_model = make_gradcam_model(model)

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(
            batch,
            training=False,
        )

        predicted_index = tf.argmax(
            predictions[0]
        )

        target_score = predictions[:, predicted_index]

    gradients = tape.gradient(
        target_score,
        conv_output,
    )

    if gradients is None:
        raise RuntimeError(
            "Gradients are disconnected. The model graph needs inspection."
        )

    activations = conv_output[0]
    grads = gradients[0]

    weights = tf.reduce_mean(
        grads,
        axis=(0, 1),
    )

    heatmap = tf.reduce_sum(
        activations * weights,
        axis=-1,
    )

    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.reduce_max(heatmap)

    heatmap = tf.math.divide_no_nan(
        heatmap,
        max_value,
    ).numpy()

    heatmap = cv2.resize(
        heatmap,
        (original.shape[1], original.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )

    heatmap = np.clip(heatmap, 0, 1)

    colored_bgr = cv2.applyColorMap(
        np.uint8(255 * heatmap),
        cv2.COLORMAP_JET,
    )

    colored_rgb = cv2.cvtColor(
        colored_bgr,
        cv2.COLOR_BGR2RGB,
    )

    overlay = cv2.addWeighted(
        original.astype(np.uint8),
        1 - alpha,
        colored_rgb,
        alpha,
        0,
    )

    probabilities = predictions.numpy()[0]

    scores = {
        CLASS_NAMES[i]: float(probabilities[i])
        for i in range(len(CLASS_NAMES))
    }

    return scores, heatmap, colored_rgb, overlay


uploaded_file = st.file_uploader(
    "Upload a brain MRI image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Uploaded MRI")
        st.image(image, caption="Original MRI", width=350)

        alpha = st.slider(
            "Heatmap overlay transparency",
            min_value=0.1,
            max_value=0.8,
            value=0.4,
            step=0.05,
        )

        if st.button("Classify MRI", type="primary"):
            with st.spinner("Analyzing MRI..."):
                model = load_model()
                scores, heatmap, colored, overlay = predict_mri(
                    image,
                    model,
                    alpha=alpha,
                )

            predicted_class = max(scores, key=scores.get)

            st.subheader("Prediction")
            st.success(f"Predicted class: {predicted_class}")
            st.metric(
                "Model confidence",
                f"{scores[predicted_class] * 100:.2f}%",
            )

            st.subheader("Grad-CAM Visualisation")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.image(
                    image,
                    caption="Original MRI",
                    use_container_width=True,
                )

            with col2:
                st.image(
                    colored,
                    caption="Colour heatmap",
                    use_container_width=True,
                )

            with col3:
                st.image(
                    overlay,
                    caption="Heatmap overlay",
                    use_container_width=True,
                )

            st.subheader("Class confidence scores")
            st.bar_chart(scores)

            st.caption(
                "Grad-CAM shows regions influencing the model's prediction; "
                "it does not outline or confirm a tumour. Educational use only."
            )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
