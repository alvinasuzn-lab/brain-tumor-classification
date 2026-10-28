
import io
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

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


# ==================================================
# 2. LOAD MODEL
# ==================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}. "
            "Place the model file in the same folder as app.py."
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
                f"Required model layer '{name}' was not found."
            ) from exc

    return model


# ==================================================
# 3. GRAD-CAM
# ==================================================

def generate_gradcam(image_array, model, class_index):
    """
    Generate Grad-CAM from an intermediate ResNet50 layer.

    Uses conv4_block6_out for a higher spatial resolution
    than conv5_block3_out.
    """

    base = model.get_layer("resnet50")

    conv_layer = base.get_layer("conv4_block6_out")

    feature_model = tf.keras.Model(
        inputs=base.input,
        outputs=[
            conv_layer.output,
            base.output
        ]
    )

    batch = image_array.astype(np.float32)[None, ...]

    # Preprocess for the nested ResNet50 feature extractor.
    processed = preprocess_input(batch)

    with tf.GradientTape() as tape:
        conv_outputs, base_outputs = feature_model(
            processed,
            training=False
        )

        # Follow the original trained classification head.
        x = model.get_layer("gap")(base_outputs)
        x = model.get_layer("dropout")(x, training=False)
        x = model.get_layer("dense_features")(x)
        predictions = model.get_layer("class_predictions")(x)

        class_score = predictions[:, class_index]

    gradients = tape.gradient(class_score, conv_outputs)

    if gradients is None:
        raise ValueError(
            "Gradients could not be calculated. "
            "Please check the saved model architecture."
        )

    # Calculate channel importance.
    weights = tf.reduce_mean(
        gradients,
        axis=(1, 2),
        keepdims=True
    )

    # Weighted combination of feature maps.
    heatmap = tf.reduce_sum(
        conv_outputs * weights,
        axis=-1
    )[0]

    heatmap = tf.maximum(heatmap, 0)

    max_value = float(tf.reduce_max(heatmap).numpy())

    if max_value < 1e-8:
        raise ValueError(
            "Grad-CAM activation is nearly zero for this image."
        )

    heatmap = heatmap / max_value
    heatmap = heatmap.numpy()

    # Resize the feature map to the original image dimensions.
    heatmap = cv2.resize(
        heatmap,
        (image_array.shape[1], image_array.shape[0]),
        interpolation=cv2.INTER_CUBIC
    )

    heatmap = np.clip(heatmap, 0, 1)

    # Improve visual contrast; this does not add model information.
    heatmap = np.power(heatmap, 0.7)

    # Create the coloured heatmap.
    colored_heatmap = cv2.applyColorMap(
        np.uint8(255 * heatmap),
        cv2.COLORMAP_JET
    )

    colored_heatmap = cv2.cvtColor(
        colored_heatmap,
        cv2.COLOR_BGR2RGB
    )

    # Blend the heatmap and the original MRI.
    overlay = cv2.addWeighted(
        image_array,
        0.65,
        colored_heatmap,
        0.35,
        0
    )

    return heatmap, colored_heatmap, overlay


# ==================================================
# 4. CLASSIFICATION
# ==================================================

def predict_mri(image, model):
    image = image.convert("RGB").resize(IMG_SIZE)

    original = np.array(image, dtype=np.uint8)

    # The complete model receives raw RGB pixels.
    # Its own preprocessing layer is expected to handle
    # preprocessing during prediction.
    batch = original.astype(np.float32)[None, ...]

    probabilities = model.predict(
        batch,
        verbose=0
    )[0]

    if len(probabilities) != len(CLASS_NAMES):
        raise ValueError(
            "The model output does not contain four class scores. "
            "Check the model and class order."
        )

    predicted_index = int(np.argmax(probabilities))

    heatmap, colored_heatmap, overlay = generate_gradcam(
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
        heatmap,
        colored_heatmap,
        overlay
    )


# ==================================================
# 5. STREAMLIT INTERFACE
# ==================================================

st.title("🧠 Brain Tumor MRI Classification")

st.write(
    "Upload a brain MRI image to obtain a model prediction "
    "and view the regions influencing that prediction."
)

st.info(
    "This is an educational machine-learning project. "
    "It is not a medical diagnostic tool. Consult a qualified "
    "healthcare professional for medical interpretation."
)

uploaded_file = st.file_uploader(
    "Upload an MRI image",
    type=["jpg", "jpeg", "png"]
)


# ==================================================
# 6. DISPLAY UPLOADED IMAGE
# ==================================================

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Uploaded MRI")

        st.image(
            image,
            caption="Original uploaded image",
            width=350
        )

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
                    heatmap,
                    colored_heatmap,
                    overlay
                ) = predict_mri(image, model)

            predicted_class = max(
                scores,
                key=scores.get
            )

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
                "The score is the model's output, not a guarantee "
                "that the prediction is medically correct."
            )

            st.subheader("Class Scores")

            score_percentages = {
                name.title(): value * 100
                for name, value in scores.items()
            }

            st.bar_chart(score_percentages)

            for name, value in scores.items():
                st.write(
                    f"**{name.title()}:** {value * 100:.2f}%"
                )

            # ------------------------------------------
            # GRAD-CAM RESULTS
            # ------------------------------------------

            st.divider()
            st.subheader("Grad-CAM Visualizations")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.image(
                    original,
                    caption="Original MRI",
                    use_container_width=True
                )

            with col2:
                st.image(
                    colored_heatmap,
                    caption="Grad-CAM Heatmap",
                    use_container_width=True
                )

            with col3:
                st.image(
                    overlay,
                    caption="Grad-CAM Overlay",
                    use_container_width=True
                )

            st.caption(
                "Red/yellow regions indicate stronger relative "
                "Grad-CAM activation; blue regions indicate weaker "
                "activation. These colours do not identify tumour "
                "boundaries or prove that a region contains a tumour."
            )

            # ------------------------------------------
            # DOWNLOAD OVERLAY
            # ------------------------------------------

            st.subheader("Download Result")

            output_buffer = io.BytesIO()

            Image.fromarray(overlay).save(
                output_buffer,
                format="PNG"
            )

            st.download_button(
                label="Download Grad-CAM Overlay",
                data=output_buffer.getvalue(),
                file_name="gradcam_overlay.png",
                mime="image/png"
            )

    except Exception as error:
        st.error("An error occurred while processing the image.")
        st.exception(error)
