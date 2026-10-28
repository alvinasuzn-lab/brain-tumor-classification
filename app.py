
import os
from pathlib import Path

import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import gdown

from PIL import Image

# --------------------------------------------------
# APP CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Brain Tumor MRI Classification",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Brain Tumor Classification from MRI Images")

st.write(
    "Upload an MRI image to view the predicted class, "
    "confidence scores, and Grad-CAM visualisation."
)

st.warning(
    "Educational research demonstration only. "
    "This model is not validated for medical diagnosis."
)

# IMPORTANT:
# Confirm this order against the class_indices used during training.
CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary Tumor",
]

IMG_SIZE = (224, 224)

MODEL_ID = "1sO3yNwEIu9oCfP3W_wlC6r07fhv0hEYI"
MODEL_NAME = "best_brain_tumor_finetuned.keras"

MODEL_PATH = (
    Path.home()
    / ".brain_tumor_model"
    / MODEL_NAME
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size < 1_000_000:
        url = f"https://drive.google.com/uc?id={MODEL_ID}"

        downloaded_path = gdown.download(
            url=url,
            output=str(MODEL_PATH),
            quiet=False,
        )

        if not downloaded_path or not MODEL_PATH.exists():
            raise RuntimeError(
                "Model download failed. Check that the Google Drive "
                "file is shared as Anyone with the link — Viewer."
            )

    if MODEL_PATH.stat().st_size < 1_000_000:
        raise RuntimeError(
            "Downloaded model file appears incomplete."
        )

    return tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False,
    )

# --------------------------------------------------
# GRAD-CAM
# --------------------------------------------------

def generate_gradcam(model, image_batch, class_index):
    """
    Generates Grad-CAM from the final convolutional layer
    inside the model's ResNet50 backbone.

    The input image must use the same preprocessing as training.
    """

    try:
        resnet = model.get_layer("resnet50")
        last_conv = resnet.get_layer("conv5_block3_3_conv")

        grad_model = tf.keras.Model(
            inputs=resnet.inputs,
            outputs=[
                last_conv.output,
                resnet.output,
            ],
        )

        with tf.GradientTape() as tape:
            # The outer model includes preprocessing layers.
            # Capture the tensor entering ResNet50.
            backbone_input_model = tf.keras.Model(
                inputs=model.inputs,
                outputs=resnet.input,
            )

            backbone_input = backbone_input_model(
                image_batch,
                training=False,
            )

            tape.watch(backbone_input)

            conv_output, backbone_output = grad_model(
                backbone_input,
                training=False,
            )

            # Recompute the classifier output from the backbone
            # using the model's existing trained head.
            x = model.get_layer("gap")(backbone_output)
            x = model.get_layer("dropout")(x, training=False)
            x = model.get_layer("dense_features")(x)
            predictions = model.get_layer("class_predictions")(x)

            target_score = predictions[:, class_index]

        gradients = tape.gradient(target_score, conv_output)

        if gradients is None:
            raise RuntimeError(
                "Gradients could not be calculated for the selected layer."
            )

        weights = tf.reduce_mean(
            gradients,
            axis=(1, 2),
        )

        conv_output = conv_output[0]
        weights = weights[0]

        heatmap = tf.reduce_sum(
            conv_output * weights,
            axis=-1,
        )

        heatmap = tf.maximum(heatmap, 0)
        max_value = tf.reduce_max(heatmap)

        heatmap = tf.math.divide_no_nan(
            heatmap,
            max_value,
        )

        return heatmap.numpy()

    except Exception as error:
        raise RuntimeError(
            f"Grad-CAM generation failed: {error}"
        ) from error

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

def predict_mri(image, model):
    """
    Predicts a class and creates a Grad-CAM heatmap/overlay.

    NOTE: Pixel normalization must be verified against training.
    This version passes RGB pixels in the 0–255 range to the model.
    """

    # Convert to RGB and resize consistently.
    pil_image = image.convert("RGB")
    original = np.array(pil_image)

    resized = cv2.resize(
        original,
        IMG_SIZE,
        interpolation=cv2.INTER_AREA,
    )

    # Keep 0–255 pixel scale here.
    # Do not add ResNet50 preprocess_input unless training used it.
    image_batch = np.expand_dims(
        resized.astype(np.float32),
        axis=0,
    )

    outputs = model(
        image_batch,
        training=False,
    ).numpy()[0]

    # This assumes the model outputs softmax probabilities.
    # Verify the final activation before interpreting these as confidence.
    probabilities = outputs

    if (
        probabilities.ndim != 1
        or len(probabilities) != len(CLASS_NAMES)
        or not np.all(np.isfinite(probabilities))
    ):
        raise ValueError(
            "The model output does not match the four configured classes."
        )

    predicted_index = int(np.argmax(probabilities))

    # Grad-CAM for the predicted class.
    heatmap = generate_gradcam(
        model,
        image_batch,
        predicted_index,
    )

    heatmap = cv2.resize(
        heatmap,
        (original.shape[1], original.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )

    heatmap = np.clip(heatmap, 0, 1)

    # Colour heatmap: OpenCV produces BGR, then convert to RGB.
    colored_bgr = cv2.applyColorMap(
        np.uint8(255 * heatmap),
        cv2.COLORMAP_JET,
    )

    colored_rgb = cv2.cvtColor(
        colored_bgr,
        cv2.COLOR_BGR2RGB,
    )

    # Ensure matching image size and dtype.
    original = np.asarray(original, dtype=np.uint8)
    colored_rgb = np.asarray(colored_rgb, dtype=np.uint8)

    overlay = cv2.addWeighted(
        original,
        0.60,
        colored_rgb,
        0.40,
        0,
    )

    scores = {
        CLASS_NAMES[i]: float(probabilities[i])
        for i in range(len(CLASS_NAMES))
    }

    return scores, heatmap, colored_rgb, overlay

# --------------------------------------------------
# STREAMLIT INTERFACE
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a brain MRI image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Uploaded MRI")
        st.image(image, caption="Original uploaded MRI", width=350)

        if st.button("Classify MRI", type="primary"):
            with st.spinner("Loading model and analyzing MRI..."):
                model = load_model()

                scores, heatmap, colored, overlay = predict_mri(
                    image,
                    model,
                )

            predicted_class = max(scores, key=scores.get)
            confidence = scores[predicted_class] * 100

            st.subheader("Prediction Results")

            result_col, confidence_col = st.columns(2)

            with result_col:
                st.success(
                    f"Predicted class: {predicted_class}"
                )

            with confidence_col:
                st.metric(
                    "Model confidence",
                    f"{confidence:.2f}%",
                )

            st.subheader("Grad-CAM Visualisation")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.image(
                    image,
                    caption="1. Original MRI",
                    use_container_width=True,
                )

            with col2:
                st.image(
                    colored,
                    caption="2. Colour heatmap",
                    use_container_width=True,
                )

            with col3:
                st.image(
                    overlay,
                    caption="3. Heatmap overlay",
                    use_container_width=True,
                )

            st.subheader("Confidence Scores")

            st.bar_chart(scores)

            with st.expander("View exact class scores"):
                for class_name, score in scores.items():
                    st.write(
                        f"**{class_name}:** {score * 100:.2f}%"
                    )

            st.caption(
                "Grad-CAM indicates image regions that influence "
                "the model's prediction; it does not establish "
                "the actual tumour boundaries. This application is "
                "for educational research only and is not a "
                "medical diagnostic system."
            )

    except Exception as error:
        st.error(
            f"Prediction failed: {error}"
        )
        st.info(
            "Check the model download, class order, preprocessing, "
            "and Grad-CAM implementation."
        )
