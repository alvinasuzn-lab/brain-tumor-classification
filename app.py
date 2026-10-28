
import io
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

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


# =========================================================
# 2. LOAD MODEL
# =========================================================

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
                f"Required model layer '{name}' was not found."
            ) from exc

    return model


# =========================================================
# 3. GENERATE GRAD-CAM
# =========================================================

def generate_gradcam(image_array, model, class_index):
    """
    Generate Grad-CAM using the model's existing ResNet50
    feature extractor and trained classification head.

    The target score is calculated before the final Dense
    layer's activation to reduce softmax saturation.
    """

    base = model.get_layer("resnet50")
    conv_layer = base.get_layer("conv5_block3_out")
    classifier = model.get_layer("class_predictions")

    # This model returns the convolutional feature map and
    # the final output of the ResNet50 feature extractor.
    feature_model = tf.keras.Model(
        inputs=base.input,
        outputs=[
            conv_layer.output,
            base.output
        ]
    )

    batch = image_array.astype(np.float32)[None, ...]

    # Match the preprocessing used by the ResNet50 branch.
    processed = preprocess_input(batch)

    with tf.GradientTape() as tape:

        conv_outputs, base_outputs = feature_model(
            processed,
            training=False
        )

        # Use the model's original trained head.
        x = model.get_layer("gap")(base_outputs)
        x = model.get_layer("dropout")(x, training=False)
        x = model.get_layer("dense_features")(x)

        # Calculate the pre-activation score of the selected
        # class directly from the trained Dense layer weights.
        # This avoids taking gradients through a saturated softmax.
        if not isinstance(classifier, tf.keras.layers.Dense):
            raise TypeError(
                "The class_predictions layer is not a Dense layer. "
                "The Grad-CAM calculation needs adjustment for this model."
            )

        logits = tf.linalg.matmul(x, classifier.kernel)

        if classifier.use_bias:
            logits = tf.nn.bias_add(logits, classifier.bias)

        class_score = logits[:, class_index]

    gradients = tape.gradient(class_score, conv_outputs)

    if gradients is None:
        return None, None, None, (
            "Gradients are unavailable. The model's "
            "classification graph could not be connected."
        )

    # Standard Grad-CAM channel weights.
    weights = tf.reduce_mean(
        gradients,
        axis=(1, 2),
        keepdims=True
    )

    # Weighted sum of feature maps.
    raw_heatmap = tf.reduce_sum(
        conv_outputs * weights,
        axis=-1
    )[0]

    # Standard Grad-CAM retains positive contributions.
    positive_heatmap = tf.maximum(raw_heatmap, 0)
    heatmap = positive_heatmap.numpy()

    # If there is no positive activation, do not create a
    # fake heatmap. Return an explanation-unavailable status.
    max_value = float(np.max(heatmap))

    if not np.isfinite(max_value) or max_value < 1e-8:
        return None, None, None, (
            "The gradients were calculated, but no meaningful "
            "positive Grad-CAM activation was produced. "
            "The class prediction is still available; the "
            "explanation is unavailable for this image."
        )

    heatmap = heatmap / max_value

    # Resize to the MRI image dimensions.
    heatmap = cv2.resize(
        heatmap,
        (image_array.shape[1], image_array.shape[0]),
        interpolation=cv2.INTER_CUBIC
    )

    heatmap = np.clip(heatmap, 0, 1)

    # Colour map for visualization only.
    coloured = cv2.applyColorMap(
        np.uint8(255 * heatmap),
        cv2.COLORMAP_JET
    )

    coloured = cv2.cvtColor(
        coloured,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        image_array,
        0.65,
        coloured,
        0.35,
        0
    )

    return heatmap, coloured, overlay, None


# =========================================================
# 4. PREDICT MRI CLASS
# =========================================================

def predict_mri(image, model):
    image = image.convert("RGB").resize(IMG_SIZE)

    original = np.array(image, dtype=np.uint8)

    # The complete saved model receives raw RGB pixels,
    # matching the existing model's input pipeline.
    batch = original.astype(np.float32)[None, ...]

    probabilities = model.predict(
        batch,
        verbose=0
    )[0]

    if len(probabilities) != len(CLASS_NAMES):
        raise ValueError(
            "The model must output four class scores. "
            "Please check the model and class order."
        )

    predicted_index = int(np.argmax(probabilities))

    heatmap, coloured, overlay, explanation_error = (
        generate_gradcam(
            original,
            model,
            predicted_index
        )
    )

    scores = {
        CLASS_NAMES[i]: float(probabilities[i])
        for i in range(len(CLASS_NAMES))
    }

    return (
        original,
        scores,
        heatmap,
        coloured,
        overlay,
        explanation_error
    )


# =========================================================
# 5. STREAMLIT PAGE
# =========================================================

st.title("🧠 Brain Tumor MRI Classification")

st.write(
    "Upload a brain MRI image to view the predicted class, "
    "model output scores, and Grad-CAM explanation."
)

st.warning(
    "Educational project demonstration only. This model "
    "is not a medical diagnostic tool and must not replace "
    "professional medical assessment."
)

uploaded_file = st.file_uploader(
    "Upload a brain MRI image",
    type=["jpg", "jpeg", "png"]
)


# =========================================================
# 6. DISPLAY IMAGE AND RUN PREDICTION
# =========================================================

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

            with st.spinner("Analysing MRI..."):

                model = load_model()

                (
                    original,
                    scores,
                    heatmap,
                    coloured,
                    overlay,
                    explanation_error
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
                "The displayed score is the model's output. "
                "It is not a guarantee of diagnostic accuracy."
            )

            st.subheader("Class Scores")

            percentages = {
                name.title(): value * 100
                for name, value in scores.items()
            }

            st.bar_chart(percentages)

            for name, value in scores.items():
                st.write(
                    f"**{name.title()}:** {value * 100:.2f}%"
                )

            # ---------------------------------------------
            # GRAD-CAM RESULTS
            # ---------------------------------------------

            st.divider()
            st.subheader("Grad-CAM Visualization")

            if explanation_error is not None:
                st.warning(explanation_error)

                st.image(
                    original,
                    caption="Original MRI — explanation unavailable",
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
                        coloured,
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
                    "activation in the Grad-CAM visualization. "
                    "They do not prove tumour location or boundaries."
                )

                # Download overlay.
                buffer = io.BytesIO()

                Image.fromarray(overlay).save(
                    buffer,
                    format="PNG"
                )

                st.download_button(
                    label="Download Grad-CAM Overlay",
                    data=buffer.getvalue(),
                    file_name="gradcam_overlay.png",
                    mime="image/png"
                )

    except Exception as error:
        st.error("An error occurred while processing the MRI.")
        st.exception(error)
