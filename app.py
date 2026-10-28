
import os
from pathlib import Path

import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import gdown

from PIL import Image


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Brain Tumour MRI Classification",
    page_icon="🧠",
    layout="wide"
)

MODEL_ID = "1sO3yNwEIu9oCfP3W_wlC6r07fhv0hEYI"
MODEL_NAME = "best_brain_tumor_finetuned.keras"
IMG_SIZE = (224, 224)

# IMPORTANT: Confirm these names match your training class_indices.
CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary Tumor"
]


# --------------------------------------------------
# LOAD MODEL AND BUILD GRAD-CAM COMPONENTS
# --------------------------------------------------

@st.cache_resource
def load_model():
    model_dir = Path.home() / ".brain_tumor_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / MODEL_NAME

    if not model_path.exists():
        url = f"https://drive.google.com/uc?id={MODEL_ID}"
        result = gdown.download(
            url,
            str(model_path),
            quiet=False
        )

        if result is None or not model_path.exists():
            raise RuntimeError(
                "Model download failed. Check the Google Drive "
                "file ID and sharing permissions."
            )

    model = tf.keras.models.load_model(
        str(model_path),
        compile=False
    )

    # Actual layer names from your deployed model.
    augmentation_layer = model.get_layer("data_augmentation")
    resnet = model.get_layer("resnet50")
    gap_layer = model.get_layer("gap")
    dropout_layer = model.get_layer("dropout")
    dense_layer = model.get_layer("dense_features")
    classifier_layer = model.get_layer("class_predictions")

    last_conv = resnet.get_layer("conv5_block3_3_conv")

    # This model follows the OUTER model graph and stops
    # at data_augmentation. No nonexistent "add" layer.
    preprocessing_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=augmentation_layer.output
    )

    # This model runs only the ResNet50 backbone.
    grad_model = tf.keras.Model(
        inputs=resnet.inputs,
        outputs=[last_conv.output, resnet.output]
    )

    return (
        model,
        preprocessing_model,
        grad_model,
        gap_layer,
        dropout_layer,
        dense_layer,
        classifier_layer
    )


# --------------------------------------------------
# PREPARE IMAGE
# --------------------------------------------------

def prepare_image(image):
    image = image.convert("RGB")
    original = np.array(image)

    resized = image.resize(IMG_SIZE)
    array = np.asarray(resized, dtype=np.float32)

    batch = np.expand_dims(array, axis=0)

    return original, batch


# --------------------------------------------------
# PREDICTION AND GRAD-CAM
# --------------------------------------------------

def predict_and_gradcam(
    image_batch,
    preprocessing_model,
    grad_model,
    gap_layer,
    dropout_layer,
    dense_layer,
    classifier_layer
):
    image_tensor = tf.convert_to_tensor(
        image_batch,
        dtype=tf.float32
    )

    # Apply the model's own preprocessing/augmentation layer.
    backbone_input = preprocessing_model(
        image_tensor,
        training=False
    )

    with tf.GradientTape() as tape:
        conv_output, backbone_output = grad_model(
            backbone_input,
            training=False
        )

        # Use the trained classifier head.
        x = gap_layer(backbone_output)
        x = dropout_layer(x, training=False)
        x = dense_layer(x)
        scores = classifier_layer(x)

        predicted_index = int(
            tf.argmax(scores[0]).numpy()
        )

        class_score = scores[:, predicted_index]

    gradients = tape.gradient(class_score, conv_output)

    if gradients is None:
        raise RuntimeError(
            "Grad-CAM gradients are unavailable. "
            "Check the model architecture."
        )

    # Average gradients over height and width.
    weights = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    # Weighted sum of feature maps.
    cam = tf.reduce_sum(
        conv_output * weights[:, None, None, :],
        axis=-1
    )

    cam = tf.nn.relu(cam)[0].numpy()

    maximum = float(np.max(cam))
    if maximum > 0:
        cam = cam / maximum

    return scores.numpy()[0], predicted_index, cam


# --------------------------------------------------
# HEATMAP AND OVERLAY
# --------------------------------------------------

def create_visualizations(original_image, heatmap):
    height, width = original_image.shape[:2]

    resized_heatmap = cv2.resize(
        heatmap,
        (width, height)
    )

    heatmap_uint8 = np.uint8(
        np.clip(resized_heatmap, 0, 1) * 255
    )

    heatmap_bgr = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    heatmap_rgb = cv2.cvtColor(
        heatmap_bgr,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        original_image,
        0.6,
        heatmap_rgb,
        0.4,
        0
    )

    return heatmap_rgb, overlay


# --------------------------------------------------
# STREAMLIT APP
# --------------------------------------------------

st.title("🧠 Brain Tumour MRI Classification")

st.write(
    "Upload a brain MRI image to view the model's predicted "
    "class, class scores, and Grad-CAM visualization."
)

st.warning(
    "This is an educational research prototype, not a "
    "medical diagnostic tool."
)

uploaded_file = st.file_uploader(
    "Upload MRI image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        original_image, image_batch = prepare_image(image)

        with st.spinner("Analyzing MRI image..."):
            (
                model,
                preprocessing_model,
                grad_model,
                gap_layer,
                dropout_layer,
                dense_layer,
                classifier_layer
            ) = load_model()

            scores, predicted_index, heatmap = predict_and_gradcam(
                image_batch,
                preprocessing_model,
                grad_model,
                gap_layer,
                dropout_layer,
                dense_layer,
                classifier_layer
            )

            # Softmax conversion only if classifier output
            # is not already softmax probabilities.
            activation_name = getattr(
                classifier_layer.activation,
                "__name__",
                ""
            )

            if activation_name == "softmax":
                probabilities = scores
            else:
                probabilities = tf.nn.softmax(scores).numpy()

            # Keep class scores within a valid display range.
            probabilities = np.asarray(
                probabilities,
                dtype=np.float64
            )

            heatmap_rgb, overlay = create_visualizations(
                original_image,
                heatmap
            )

        st.success("Analysis completed!")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Predicted class",
                CLASS_NAMES[predicted_index]
            )

        with col2:
            st.metric(
                "Model confidence",
                f"{probabilities[predicted_index] * 100:.2f}%"
            )

        st.subheader("MRI and Grad-CAM Results")

        image_col1, image_col2, image_col3 = st.columns(3)

        with image_col1:
            st.image(
                original_image,
                caption="Original MRI",
                use_container_width=True
            )

        with image_col2:
            st.image(
                heatmap_rgb,
                caption="Grad-CAM Heatmap",
                use_container_width=True
            )

        with image_col3:
            st.image(
                overlay,
                caption="Heatmap Overlay",
                use_container_width=True
            )

        st.caption(
            "Grad-CAM highlights image regions that influence "
            "the selected class score. It does not confirm the "
            "presence or location of a tumour."
        )

        st.subheader("Class Scores")

        for index, class_name in enumerate(CLASS_NAMES):
            value = float(probabilities[index])

            st.write(f"**{class_name}: {value * 100:.2f}%**")
            st.progress(float(np.clip(value, 0.0, 1.0)))

        with st.expander("Technical details"):
            st.write("Input size: 224 × 224 × 3")
            st.write("Backbone: ResNet50")
            st.write("Grad-CAM layer: conv5_block3_3_conv")
            st.write("Grad-CAM map size: 7 × 7")
            st.write("Model output shape:", model.output_shape)

    except Exception as error:
        st.error("The image could not be analyzed.")
        st.exception(error)
