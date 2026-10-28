
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

# Confirm this order matches the class_indices used in training.
CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary Tumor"
]


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    model_dir = Path.home() / ".brain_tumor_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / MODEL_NAME

    if not model_path.exists() or model_path.stat().st_size == 0:
        if model_path.exists():
            model_path.unlink()

        with st.spinner("Downloading trained model from Google Drive..."):
            result = gdown.download(
                id=MODEL_ID,
                output=str(model_path),
                quiet=False,
                fuzzy=True
            )

        if (
            result is None
            or not model_path.exists()
            or model_path.stat().st_size == 0
        ):
            raise RuntimeError(
                "Model download failed. Check the Google Drive file ID "
                "and ensure the file is shared as Anyone with the link - Viewer."
            )

    model = tf.keras.models.load_model(
        str(model_path),
        compile=False
    )

    required_layers = [
        "data_augmentation",
        "resnet50",
        "gap",
        "dropout",
        "dense_features",
        "class_predictions"
    ]

    missing = [
        name for name in required_layers
        if not any(layer.name == name for layer in model.layers)
    ]

    if missing:
        raise ValueError(
            "Expected layers are missing from the model: "
            + ", ".join(missing)
        )

    augmentation = model.get_layer("data_augmentation")
    resnet = model.get_layer("resnet50")

    # Preserve the outer model's preprocessing path.
    preprocessing_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=augmentation.output
    )

    # Final convolution inside ResNet50.
    last_conv = resnet.get_layer("conv5_block3_3_conv")

    grad_model = tf.keras.Model(
        inputs=resnet.inputs,
        outputs=[
            last_conv.output,
            resnet.output
        ]
    )

    return (
        model,
        preprocessing_model,
        grad_model,
        model.get_layer("gap"),
        model.get_layer("dropout"),
        model.get_layer("dense_features"),
        model.get_layer("class_predictions")
    )


# --------------------------------------------------
# PREPARE IMAGE
# --------------------------------------------------

def prepare_image(image):
    image = image.convert("RGB")

    original = np.asarray(
        image,
        dtype=np.uint8
    )

    resized = image.resize(
        IMG_SIZE,
        Image.Resampling.BILINEAR
    )

    image_batch = np.asarray(
        resized,
        dtype=np.float32
    )[None, ...]

    return original, image_batch


# --------------------------------------------------
# PROBABILITY CONVERSION
# --------------------------------------------------

def get_probabilities(scores, classifier_layer):
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)

    if not np.all(np.isfinite(scores)):
        raise ValueError("Model produced non-finite output scores.")

    if len(scores) != len(CLASS_NAMES):
        raise ValueError(
            f"Expected {len(CLASS_NAMES)} class scores, "
            f"received {len(scores)}."
        )

    activation_name = getattr(
        classifier_layer.activation,
        "__name__",
        ""
    )

    if activation_name == "softmax":
        probabilities = scores
    else:
        probabilities = tf.nn.softmax(scores).numpy()

    return np.asarray(probabilities, dtype=np.float64)


# --------------------------------------------------
# PREDICTION AND GRAD-CAM
# --------------------------------------------------

def predict_and_gradcam(
    image_batch,
    model,
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

    # Full model's actual prediction.
    full_scores = model(
        image_tensor,
        training=False
    ).numpy()[0]

    full_probabilities = get_probabilities(
        full_scores,
        classifier_layer
    )

    full_predicted_index = int(
        np.argmax(full_probabilities)
    )

    # Reconstruct the path used to calculate gradients.
    backbone_input = preprocessing_model(
        image_tensor,
        training=False
    )

    with tf.GradientTape() as tape:
        conv_output, backbone_output = grad_model(
            backbone_input,
            training=False
        )

        # Reuse the original trained head layers.
        x = gap_layer(backbone_output)
        x = dropout_layer(x, training=False)
        x = dense_layer(x)
        head_scores = classifier_layer(x)

        # Target the full model's predicted class.
        target_score = head_scores[:, full_predicted_index]

    gradients = tape.gradient(
        target_score,
        conv_output
    )

    if gradients is None:
        raise RuntimeError(
            "Grad-CAM gradients are unavailable. "
            "The convolutional layer may not be connected to the target score."
        )

    weights = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    cam = tf.reduce_sum(
        conv_output * weights[:, None, None, :],
        axis=-1
    )

    cam = tf.nn.relu(cam)[0].numpy()

    if not np.all(np.isfinite(cam)):
        raise RuntimeError("Grad-CAM contains invalid values.")

    maximum = float(np.max(cam))

    if maximum <= 1e-8:
        raise RuntimeError(
            "No positive Grad-CAM activation was produced for the "
            "selected class. The heatmap cannot be displayed reliably."
        )

    cam = cam / maximum

    # Check whether the reconstructed head agrees with the full model.
    head_probabilities = get_probabilities(
        head_scores.numpy()[0],
        classifier_layer
    )

    head_predicted_index = int(
        np.argmax(head_probabilities)
    )

    outputs_match = np.allclose(
        full_probabilities,
        head_probabilities,
        rtol=1e-3,
        atol=1e-4
    )

    return (
        full_probabilities,
        full_predicted_index,
        cam,
        outputs_match,
        head_predicted_index
    )


# --------------------------------------------------
# HEATMAP AND OVERLAY
# --------------------------------------------------

def create_visualizations(original_image, heatmap):
    height, width = original_image.shape[:2]

    resized_heatmap = cv2.resize(
        heatmap,
        (width, height),
        interpolation=cv2.INTER_LINEAR
    )

    resized_heatmap = np.clip(
        resized_heatmap,
        0.0,
        1.0
    )

    heatmap_uint8 = np.uint8(
        resized_heatmap * 255
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
        0.60,
        heatmap_rgb,
        0.40,
        0
    )

    return heatmap_rgb, overlay


# --------------------------------------------------
# STREAMLIT USER INTERFACE
# --------------------------------------------------

st.title("🧠 Brain Tumour MRI Classification")

st.write(
    "Upload a brain MRI image to view the model prediction, "
    "class scores and Grad-CAM visualization."
)

st.warning(
    "Educational research prototype only. This application "
    "is not a medical diagnostic tool."
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

            (
                probabilities,
                predicted_index,
                heatmap,
                outputs_match,
                head_predicted_index
            ) = predict_and_gradcam(
                image_batch,
                model,
                preprocessing_model,
                grad_model,
                gap_layer,
                dropout_layer,
                dense_layer,
                classifier_layer
            )

            heatmap_rgb, overlay = create_visualizations(
                original_image,
                heatmap
            )

        st.success("Analysis completed.")

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

        if not outputs_match:
            st.error(
                "Warning: the reconstructed classifier does not match "
                "the full model. The displayed prediction is from the "
                "full model, but this Grad-CAM may not faithfully explain "
                "that prediction. Do not use it as a validated explanation."
            )

            st.write(
                "Class predicted by the Grad-CAM classifier path:",
                CLASS_NAMES[head_predicted_index]
            )

        st.subheader("MRI and Grad-CAM Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(
                original_image,
                caption="Original MRI",
                use_container_width=True
            )

        with col2:
            st.image(
                heatmap_rgb,
                caption="Grad-CAM Heatmap",
                use_container_width=True
            )

        with col3:
            st.image(
                overlay,
                caption="Heatmap Overlay",
                use_container_width=True
            )

        st.caption(
            "Grad-CAM shows regions associated with a selected class score. "
            "It does not establish tumour boundaries or prove accurate "
            "tumour localization."
        )

        st.subheader("Class Scores")

        for index, class_name in enumerate(CLASS_NAMES):
            value = float(probabilities[index])

            st.write(
                f"**{class_name}: {value * 100:.2f}%**"
            )

            st.progress(
                float(np.clip(value, 0.0, 1.0))
            )

        with st.expander("Technical details"):
            st.write("Input size: 224 × 224 × 3")
            st.write("Backbone: ResNet50")
            st.write("Grad-CAM layer: conv5_block3_3_conv")
            st.write("Expected feature map: approximately 7 × 7")
            st.write("Model output shape:", model.output_shape)
            st.write("Prediction and reconstructed head match:", outputs_match)

    except Exception as error:
        st.error("The image could not be analyzed.")
        st.exception(error)

else:
    st.info("Upload an MRI image to begin.")
