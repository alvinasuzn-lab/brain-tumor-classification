
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
            f"Cannot find {MODEL_PATH.name}. "
            "Place it in the same folder as app.py."
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    required_layers = [
        "resnet50",
        "gap",
        "dropout",
        "dense_features",
        "class_predictions"
    ]

    for layer_name in required_layers:
        try:
            model.get_layer(layer_name)
        except ValueError as exc:
            raise ValueError(
                f"Required model layer '{layer_name}' is missing."
            ) from exc

    return model


# ==================================================
# 3. GRAD-CAM
# ==================================================

def generate_gradcam(image_array, model, class_index):
    """
    Generate Grad-CAM using the final convolutional layer
    of the trained ResNet50 model.
    """

    base = model.get_layer("resnet50")
    conv_layer = base.get_layer("conv5_block3_out")

    feature_model = tf.keras.Model(
        inputs=base.input,
        outputs=[
            conv_layer.output,
            base.output
        ]
    )

    batch = image_array.astype(np.float32)[None, ...]

    # Match preprocessing inside the ResNet50 branch.
    processed = preprocess_input(batch)

    with tf.GradientTape() as tape:
        conv_outputs, base_outputs = feature_model(
            processed,
            training=False
        )

        # Use the trained classification head.
        x = model.get_layer("gap")(base_outputs)
        x = model.get_layer("dropout")(x, training=False)
        x = model.get_layer("dense_features")(x)
        predictions = model.get_layer("class_predictions")(x)

        # Use the predicted class score.
        # Log transformation can help with very small gradients.
        class_score = tf.math.log(
            tf.clip_by_value(
                predictions[:, class_index],
                1e-7,
                1.0
            )
        )

    gradients = tape.gradient(class_score, conv_outputs)

    if gradients is None:
        raise ValueError(
            "Gradients could not be calculated. "
            "Check the model architecture."
        )

    # Calculate the importance of each feature-map channel.
    weights = tf.reduce_mean(
        gradients,
        axis=(1, 2),
        keepdims=True
    )

    heatmap = tf.reduce_sum(
        conv_outputs * weights,
        axis=-1
    )[0]

    # Retain positive class contributions.
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap.numpy()

    max_value = float(np.max(heatmap))

    if not np.isfinite(max_value) or max_value < 1e-8:
        # Never manufacture a heatmap when activation is absent.
        blank_heatmap = np.zeros(
            image_array.shape[:2],
            dtype=np.float32
        )

        blank_colour = np.zeros_like(image_array)

        return None, blank_colour, image_array.copy()

    heatmap = heatmap / max_value

    # Resize the heatmap to the MRI dimensions.
    heatmap = cv2.resize(
        heatmap,
        (image_array.shape[1], image_array.shape[0]),
        interpolation=cv2.INTER_CUBIC
    )

    heatmap = np.clip(heatmap, 0, 1)

    colored_heatmap = cv2.applyColorMap(
        np.uint8(255 * heatmap),
        cv2.COLORMAP_JET
    )

    colored_heatmap = cv2.cvtColor(
        colored_heatmap,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        image_array,
        0.65,
        colored_heatmap,
        0.35,
        0
    )

    return heatmap, colored_heatmap, overlay


# ==================================================
# 4. PREDICT MRI CLASS
# ==================================================

def predict_mri(image, model):
    image = image.convert("RGB").resize(IMG_SIZE)

    original = np.array(image, dtype=np.uint8)

    batch = original.astype(np.float32)[None, ...]

    # The full model receives raw RGB pixels, as during training.
    probabilities = model.predict(
        batch,
        verbose=0
    )[0]

    if len(probabilities) != len(CLASS_NAMES):
        raise ValueError(
            "Expected four output scores. Check model class order."
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

    return original, scores, heatmap, colored_heatmap, overlay


# ==================================================
# 5. STREAMLIT USER INTERFACE
# ==================================================

st.title("🧠 Brain Tumor MRI Classification")

st.write(
    "Upload a brain MRI image to view the predicted class, "
    "model output scores, and Grad-CAM visualization."
)

st.warning(
    "Educational project demonstration only. This model is "
    "not a medical diagnostic tool and must not replace "
    "professional medical assessment."
)

uploaded_file = st.file_uploader(
    "Upload a brain MRI image",
    type=["jpg", "jpeg", "png"]
)


# ==================================================
# 6. PROCESS UPLOADED IMAGE
# ==================================================

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Uploaded MRI")

        st.image(
            image,
            caption="Original MRI",
            width=350
        )

        if st.button(
            "Classify MRI",
            type="primary",
            use_container_width=True
        ):

            with st.spinner("Loading model and analysing MRI..."):
                model = load_model()

                (
                    original,
                    scores,
                    heatmap,
                    colored_heatmap,
                    overlay
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
                "This score represents the model's output, "
                "not the probability that a medical diagnosis "
                "is correct."
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
            # GRAD-CAM VISUALIZATIONS
            # ------------------------------------------

            st.divider()
            st.subheader("Grad-CAM Visualizations")

            if heatmap is None:
                st.warning(
                    "The model produced a class prediction, "
                    "but no meaningful positive Grad-CAM "
                    "activation was obtained for this image. "
                    "The explanation is unavailable."
                )

                st.image(
                    original,
                    caption="Original MRI",
                    use_container_width=True
                )

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
                    "Grad-CAM activation. They do not prove that "
                    "a region contains a tumour or define its boundary."
                )

                # Download overlay.
                output_buffer = io.BytesIO()

                Image.fromarray(overlay).save(
                    output_buffer,
                    format="PNG"
                )

                st.download_button(
                    "Download Grad-CAM Overlay",
                    data=output_buffer.getvalue(),
                    file_name="gradcam_overlay.png",
                    mime="image/png"
                )

    except Exception as error:
        st.error("An error occurred while processing the MRI.")
        st.exception(error)
