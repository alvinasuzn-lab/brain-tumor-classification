
import io
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input


# =====================================================
# 1. CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Brain Tumor MRI Classification",
    page_icon="🧠",
    layout="wide"
)

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

IMG_SIZE = (224, 224)

MODEL_PATH = (
    Path(__file__).parent
    / "best_brain_tumor_finetuned.keras"
)


# =====================================================
# 2. LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}. "
            "Keep it in the same folder as app.py."
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    required_layers = [
        "resnet50",
        "gap",
        "dropout",
        "dense_features",
        "class_predictions"
    ]

    for name in required_layers:
        try:
            model.get_layer(name)
        except ValueError as exc:
            raise ValueError(
                f"Required layer '{name}' was not found."
            ) from exc

    return model


# =====================================================
# 3. STANDARD GRAD-CAM
# =====================================================

def generate_gradcam(image_array, model, class_index):
    """
    Standard Grad-CAM using the final convolutional layer.
    Returns the grayscale heatmap, colour map, overlay,
    and an optional error message.
    """

    base = model.get_layer("resnet50")
    conv_layer = base.get_layer("conv5_block3_out")
    classifier = model.get_layer("class_predictions")

    # The feature model exposes the convolutional feature map
    # and the final output of the ResNet50 feature extractor.
    feature_model = tf.keras.Model(
        inputs=base.input,
        outputs=[
            conv_layer.output,
            base.output
        ]
    )

    batch = image_array.astype(np.float32)[None, ...]

    # Match the ResNet50 preprocessing used in the nested branch.
    processed = preprocess_input(batch)

    with tf.GradientTape() as tape:
        conv_outputs, base_outputs = feature_model(
            processed,
            training=False
        )

        # Use the existing trained classification head.
        x = model.get_layer("gap")(base_outputs)
        x = model.get_layer("dropout")(x, training=False)
        x = model.get_layer("dense_features")(x)

        # Calculate the class logit before the final activation.
        # This assumes the classifier is a standard Dense layer.
        if not isinstance(classifier, tf.keras.layers.Dense):
            return None, None, None, (
                "The output layer is not a Dense layer. "
                "Grad-CAM needs to be adapted to this model."
            )

        logits = tf.linalg.matmul(x, classifier.kernel)

        if classifier.use_bias:
            logits = tf.nn.bias_add(logits, classifier.bias)

        class_score = logits[:, class_index]

    gradients = tape.gradient(class_score, conv_outputs)

    if gradients is None:
        return None, None, None, (
            "Gradients could not be calculated. "
            "The model's computational graph needs inspection."
        )

    # Average the gradients over the spatial dimensions.
    channel_weights = tf.reduce_mean(
        gradients,
        axis=(1, 2),
        keepdims=True
    )

    # Weighted sum of the convolutional feature maps.
    cam = tf.reduce_sum(
        conv_outputs * channel_weights,
        axis=-1
    )[0]

    # Standard Grad-CAM retains positive contributions.
    cam = tf.nn.relu(cam)
    cam = cam.numpy()

    if not np.all(np.isfinite(cam)):
        return None, None, None, (
            "The heatmap contains invalid numerical values."
        )

    maximum = float(np.max(cam))

    if maximum < 1e-8:
        return None, None, None, (
            "No meaningful positive Grad-CAM activation was found. "
            "The prediction is available, but the explanation "
            "cannot be displayed reliably."
        )

    # Normalize the raw activation.
    cam = cam / maximum

    # Resize to the original image dimensions.
    cam = cv2.resize(
        cam,
        (image_array.shape[1], image_array.shape[0]),
        interpolation=cv2.INTER_LINEAR
    )

    cam = np.clip(cam, 0.0, 1.0)

    # Grayscale image for examining the activation itself.
    grayscale = np.uint8(cam * 255)

    # Colour visualization.
    coloured_bgr = cv2.applyColorMap(
        grayscale,
        cv2.COLORMAP_JET
    )

    coloured = cv2.cvtColor(
        coloured_bgr,
        cv2.COLOR_BGR2RGB
    )

    # Overlay on the original MRI.
    overlay = cv2.addWeighted(
        image_array,
        0.65,
        coloured,
        0.35,
        0
    )

    return grayscale, coloured, overlay, None


# =====================================================
# 4. PREDICT CLASS
# =====================================================

def predict_mri(image, model):
    image = image.convert("RGB").resize(IMG_SIZE)
    original = np.array(image, dtype=np.uint8)

    # Predict using the complete saved model.
    batch = original.astype(np.float32)[None, ...]

    probabilities = model.predict(
        batch,
        verbose=0
    )[0]

    if len(probabilities) != len(CLASS_NAMES):
        raise ValueError(
            "Expected four output scores. "
            "Check the model's output and class order."
        )

    predicted_index = int(np.argmax(probabilities))

    grayscale, coloured, overlay, cam_error = generate_gradcam(
        original,
        model,
        predicted_index
    )

    scores = {
        CLASS_NAMES[i]: float(probabilities[i])
        for i in range(len(CLASS_NAMES))
    }

    return (
        original,
        scores,
        grayscale,
        coloured,
        overlay,
        cam_error
    )


# =====================================================
# 5. STREAMLIT INTERFACE
# =====================================================

st.title("🧠 Brain Tumor MRI Classification")

st.write(
    "Upload a brain MRI to view the predicted class, "
    "class scores, and Grad-CAM visualization."
)

st.warning(
    "Educational project only. This model is not a medical "
    "diagnostic tool. Grad-CAM highlights image features "
    "influencing a prediction; it does not establish tumour "
    "boundaries or replace professional assessment."
)

uploaded_file = st.file_uploader(
    "Upload a brain MRI image",
    type=["jpg", "jpeg", "png"]
)


# =====================================================
# 6. CLASSIFICATION AND VISUALIZATION
# =====================================================

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Uploaded MRI")
        st.image(image, width=350)

        if st.button(
            "Classify MRI",
            type="primary",
            use_container_width=True
        ):
            with st.spinner("Analysing MRI..."):
                model = load_model()

                (
                    original,
                    scores,
                    grayscale,
                    coloured,
                    overlay,
                    cam_error
                ) = predict_mri(image, model)

            predicted_class = max(scores, key=scores.get)
            confidence = scores[predicted_class] * 100

            st.divider()
            st.subheader("Prediction Results")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Predicted class",
                    predicted_class.title()
                )

            with col2:
                st.metric(
                    "Model output score",
                    f"{confidence:.2f}%"
                )

            st.caption(
                "The displayed score is the model's output, "
                "not a guarantee of diagnostic accuracy."
            )

            st.subheader("Class Scores")

            st.bar_chart({
                name.title(): value * 100
                for name, value in scores.items()
            })

            for name, value in scores.items():
                st.write(
                    f"**{name.title()}:** {value * 100:.2f}%"
                )

            st.divider()
            st.subheader("Grad-CAM Visualization")

            if cam_error is not None:
                st.warning(cam_error)

            else:
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.image(
                        original,
                        caption="Original MRI",
                        use_container_width=True
                    )

                with col2:
                    st.image(
                        grayscale,
                        caption="Grayscale Activation Map",
                        clamp=True,
                        use_container_width=True
                    )

                with col3:
                    st.image(
                        overlay,
                        caption="Grad-CAM Overlay",
                        use_container_width=True
                    )

                # Optional colour heatmap.
                with st.expander("View colour heatmap"):
                    st.image(
                        coloured,
                        caption="Grad-CAM Heatmap",
                        use_container_width=True
                    )

                # Download overlay.
                buffer = io.BytesIO()

                Image.fromarray(overlay).save(
                    buffer,
                    format="PNG"
                )

                st.download_button(
                    "Download Grad-CAM Overlay",
                    data=buffer.getvalue(),
                    file_name="gradcam_overlay.png",
                    mime="image/png"
                )

    except Exception as error:
        st.error("An error occurred while processing the MRI.")
        st.exception(error)
