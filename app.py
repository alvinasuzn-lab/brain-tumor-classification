
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

# These must match the class order used during training.
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

        with st.spinner("Downloading trained model..."):
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
                "Model download failed. Check the Google Drive ID "
                "and set sharing to Anyone with the link - Viewer."
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
            "Expected model layers were not found: "
            + ", ".join(missing)
        )

    resnet = model.get_layer("resnet50")
    augmentation = model.get_layer("data_augmentation")

    # Match the model's own input-to-backbone preprocessing path.
    preprocessing_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=augmentation.output
    )

    # Final convolutional operation inside ResNet50.
    # This layer produces spatial feature maps for Grad-CAM.
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

    # This is the exact image displayed and used for the overlay.
    original = np.asarray(image, dtype=np.uint8)

    resized = image.resize(
        IMG_SIZE,
        Image.Resampling.BILINEAR
    )

    batch = np.asarray(
        resized,
        dtype=np.float32
    )[None, ...]

    return original, batch


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

    # Match the preprocessing path used by the trained model.
    backbone_input = preprocessing_model(
        image_tensor,
        training=False
    )

    with tf.GradientTape() as tape:
        conv_output, backbone_output = grad_model(
            backbone_input,
            training=False
        )

        # Reuse the trained classification head.
        x = gap_layer(backbone_output)
        x = dropout_layer(x, training=False)
        x = dense_layer(x)
        scores = classifier_layer(x)

        # Explain the class with the highest model score.
        predicted_index = int(
            tf.argmax(scores[0]).numpy()
        )

        target_score = scores[:, predicted_index]

    gradients = tape.gradient(
        target_score,
        conv_output
    )

    if gradients is None:
        raise RuntimeError(
            "Grad-CAM gradients are unavailable. "
            "Check the model graph and layer connections."
        )

    # Standard Grad-CAM channel weights.
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
        raise RuntimeError("Grad-CAM produced invalid values.")

    maximum = float(np.max(cam))

    if maximum <= 1e-8:
        raise RuntimeError(
            "Grad-CAM has no positive activation for this class. "
            "The heatmap cannot be meaningfully displayed."
        )

    cam = cam / maximum

    # Verify that the reconstructed classifier agrees with
    # the original model's prediction.
    direct_output = np.asarray(
        model.predict(image_batch, verbose=0)[0]
    )

    reconstructed_output = scores.numpy()[0]

    if not np.allclose(
        direct_output,
        reconstructed_output,
        rtol=1e-3,
        atol=1e-4
    ):
        raise RuntimeError(
            "The reconstructed classifier does not match the "
            "original model output. Grad-CAM was stopped to avoid "
            "showing an explanation for a potentially different "
            "prediction path."
        )

    # Convert output to probabilities for display.
    activation_name = getattr(
        classifier_layer.activation,
        "__name__",
        ""
    )

    if activation_name == "softmax":
        probabilities = reconstructed_output
    else:
        probabilities = tf.nn.softmax(
            reconstructed_output
        ).numpy()

    if (
        not np.all(np.isfinite(probabilities))
        or len(probabilities) != len(CLASS_NAMES)
        or np.any(probabilities < 0)
        or not np.isclose(
            np.sum(probabilities),
            1.0,
            atol=1e-3
        )
    ):
        raise RuntimeError(
            "The model output could not be interpreted as "
            "four valid class probabilities."
        )

    # The prediction and Grad-CAM target must agree.
    if predicted_index != int(np.argmax(probabilities)):
        raise RuntimeError(
            "The target class does not match the displayed prediction."
        )

    return probabilities, predicted_index, cam


# --------------------------------------------------
# HEATMAP AND OVERLAY
# --------------------------------------------------

def create_visualizations(original_image, heatmap):
    height, width = original_image.shape[:2]

    # Resizing changes only the display resolution, not activation.
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

    # Original image and heatmap are both RGB.
    overlay = cv2.addWeighted(
        original_image,
        0.60,
        heatmap_rgb,
        0.40,
        0
    )

    return heatmap_rgb, overlay


# --------------------------------------------------
# STREAMLIT APP
# --------------------------------------------------

st.title("🧠 Brain Tumour MRI Classification")

st.write(
    "Upload a brain MRI image to view the predicted class, "
    "class scores and Grad-CAM visualization."
)

st.warning(
    "Educational research prototype only. This application "
    "does not provide a medical diagnosis."
)

uploaded_file = st.file_uploader(
    "Upload MRI image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        original_image, image_batch = prepare_image(image)

        with st.spinner("Loading model and analyzing MRI..."):
            (
                model,
                preprocessing_model,
                grad_model,
                gap_layer,
                dropout_layer,
                dense_layer,
                classifier_layer
            ) = load_model()

            probabilities, predicted_index, heatmap = (
                predict_and_gradcam(
                    image_batch,
                    model,
                    preprocessing_model,
                    grad_model,
                    gap_layer,
                    dropout_layer,
                    dense_layer,
                    classifier_layer
                )
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
            "Red/yellow areas generally indicate stronger positive "
            "Grad-CAM activation relative to this heatmap's maximum. "
            "They do not establish tumour boundaries or prove that "
            "the highlighted area caused the prediction."
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

    except Exception as error:
        st.error("The image could not be analyzed.")
        st.exception(error)

else:
    st.info("Upload an MRI image to begin.")
