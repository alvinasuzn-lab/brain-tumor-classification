
import os
from pathlib import Path

import streamlit as st
import numpy as np
import tensorflow as tf
import cv2

from PIL import Image
import gdown


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Brain Tumour MRI Classification",
    page_icon="🧠",
    layout="wide"
)

MODEL_ID = "1sO3yNwEIu9oCfP3W_wlC6r07fhv0hEYI"
MODEL_NAME = "best_brain_tumor_finetuned.keras"

CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary Tumor"
]

IMG_SIZE = (224, 224)


# --------------------------------------------------
# 2. LOAD THE TRAINED MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    model_dir = Path.home() / ".brain_tumor_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / MODEL_NAME

    if not model_path.exists():
        url = f"https://drive.google.com/uc?id={MODEL_ID}"

        downloaded = gdown.download(
            url,
            str(model_path),
            quiet=False
        )

        if downloaded is None or not model_path.exists():
            raise RuntimeError(
                "Model download failed. Check the Google Drive "
                "file ID and its sharing permissions."
            )

    model = tf.keras.models.load_model(
        str(model_path),
        compile=False
    )

    # Obtain the existing ResNet50 backbone.
    resnet = model.get_layer("resnet50")

    # Obtain the final convolutional layer.
    last_conv = resnet.get_layer("conv5_block3_3_conv")

    # Build Grad-CAM model from the ResNet50 submodel.
    grad_model = tf.keras.Model(
        inputs=resnet.inputs,
        outputs=[last_conv.output, resnet.output]
    )

    # Preserve the original model's preprocessing graph.
    # This output is connected to the outer model inputs.
    preprocessing_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=model.get_layer("add").output
    )

    # Reuse the actual trained classification head.
    gap_layer = model.get_layer("gap")
    dropout_layer = model.get_layer("dropout")
    dense_layer = model.get_layer("dense_features")
    classifier_layer = model.get_layer("class_predictions")

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
# 3. PREPARE THE IMAGE
# --------------------------------------------------

def prepare_image(image):
    image = image.convert("RGB")
    original = np.array(image)

    resized = image.resize(IMG_SIZE)

    # Keep pixel range at 0-255.
    # The model's own preprocessing graph is used below.
    image_array = np.asarray(
        resized,
        dtype=np.float32
    )

    image_batch = np.expand_dims(image_array, axis=0)

    return original, image_batch


# --------------------------------------------------
# 4. GENERATE GRAD-CAM
# --------------------------------------------------

def generate_gradcam(
    image_batch,
    preprocessing_model,
    grad_model,
    gap_layer,
    dropout_layer,
    dense_layer,
    classifier_layer,
    class_index
):
    image_tensor = tf.convert_to_tensor(
        image_batch,
        dtype=tf.float32
    )

    with tf.GradientTape() as tape:

        # Follow the original model's preprocessing graph.
        backbone_input = preprocessing_model(
            image_tensor,
            training=False
        )

        # Obtain convolutional features and backbone output.
        conv_output, backbone_output = grad_model(
            backbone_input,
            training=False
        )

        # Run the original trained classification head.
        x = gap_layer(backbone_output)
        x = dropout_layer(x, training=False)
        x = dense_layer(x)
        predictions = classifier_layer(x)

        # Score for the selected class.
        class_score = predictions[:, class_index]

    # Calculate gradients of the class score
    # with respect to the convolutional feature maps.
    gradients = tape.gradient(
        class_score,
        conv_output
    )

    if gradients is None:
        raise RuntimeError(
            "Grad-CAM gradients could not be calculated."
        )

    # Average gradients across spatial dimensions.
    weights = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    # Weight each feature map by its gradient importance.
    cam = tf.reduce_sum(
        conv_output * weights[:, tf.newaxis, tf.newaxis, :],
        axis=-1
    )

    # Keep positive activations.
    cam = tf.nn.relu(cam)

    # Normalize the heatmap.
    cam = cam[0].numpy()

    maximum = np.max(cam)

    if maximum > 0:
        cam = cam / maximum

    return cam, predictions.numpy()[0]


# --------------------------------------------------
# 5. CREATE THE HEATMAP OVERLAY
# --------------------------------------------------

def create_overlay(original_image, heatmap):
    height, width = original_image.shape[:2]

    heatmap_resized = cv2.resize(
        heatmap,
        (width, height)
    )

    heatmap_uint8 = np.uint8(
        255 * np.clip(heatmap_resized, 0, 1)
    )

    colored_heatmap = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    # OpenCV produces BGR; convert to RGB.
    colored_heatmap = cv2.cvtColor(
        colored_heatmap,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        original_image,
        0.6,
        colored_heatmap,
        0.4,
        0
    )

    return colored_heatmap, overlay


# --------------------------------------------------
# 6. STREAMLIT USER INTERFACE
# --------------------------------------------------

st.title("🧠 Brain Tumour MRI Classification")

st.write(
    "Upload a brain MRI image to obtain a model prediction "
    "and a Grad-CAM visualization of the regions influencing "
    "the prediction."
)

st.warning(
    "Educational research prototype only. "
    "Predictions are not a medical diagnosis."
)

uploaded_file = st.file_uploader(
    "Upload a brain MRI image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file)

        original_image, image_batch = prepare_image(image)

        st.subheader("Uploaded MRI")
        st.image(
            original_image,
            caption="Original MRI image",
            use_container_width=True
        )

        with st.spinner("Loading model and analyzing image..."):

            (
                model,
                preprocessing_model,
                grad_model,
                gap_layer,
                dropout_layer,
                dense_layer,
                classifier_layer
            ) = load_model()

            # Generate Grad-CAM and get scores from the same
            # backbone/head forward pass.
            preliminary_cam, scores = generate_gradcam(
                image_batch,
                preprocessing_model,
                grad_model,
                gap_layer,
                dropout_layer,
                dense_layer,
                classifier_layer,
                class_index=0
            )

            # If the model outputs logits, convert them to
            # probabilities for display.
            activation_name = getattr(
                classifier_layer.activation,
                "__name__",
                ""
            )

            if activation_name == "softmax":
                probabilities = scores
            else:
                probabilities = tf.nn.softmax(scores).numpy()

            predicted_index = int(np.argmax(probabilities))
            predicted_class = CLASS_NAMES[predicted_index]
            confidence = float(probabilities[predicted_index]) * 100

            # Generate the heatmap for the predicted class.
            heatmap, _ = generate_gradcam(
                image_batch,
                preprocessing_model,
                grad_model,
                gap_layer,
                dropout_layer,
                dense_layer,
                classifier_layer,
                class_index=predicted_index
            )

            colored_heatmap, overlay = create_overlay(
                original_image,
                heatmap
            )

        st.success("Analysis completed")

        st.subheader("Prediction")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Predicted class",
                predicted_class
            )

        with col2:
            st.metric(
                "Model confidence",
                f"{confidence:.2f}%"
            )

        st.subheader("Grad-CAM Visualization")

        image_col1, image_col2, image_col3 = st.columns(3)

        with image_col1:
            st.image(
                original_image,
                caption="Original MRI",
                use_container_width=True
            )

        with image_col2:
            st.image(
                colored_heatmap,
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
            "Warm colours indicate regions with stronger positive "
            "Grad-CAM responses for the predicted class. "
            "The heatmap is an explanation aid, not proof of a tumour."
        )

        st.subheader("Class Scores")

        for i, class_name in enumerate(CLASS_NAMES):
            score_percent = float(probabilities[i]) * 100

            st.write(
                f"**{class_name}:** {score_percent:.2f}%"
            )

            st.progress(
                float(np.clip(probabilities[i], 0, 1))
            )

        with st.expander("Technical information"):
            st.write("Input size: 224 × 224 × 3")
            st.write("Backbone: ResNet50")
            st.write("Grad-CAM layer: conv5_block3_3_conv")
            st.write("Grad-CAM map size: 7 × 7")
            st.write(
                "Model output shape:",
                model.output_shape
            )

    except Exception as error:
        st.error(
            "The image could not be analyzed. "
            "Check the error details below."
        )

        st.exception(error)
