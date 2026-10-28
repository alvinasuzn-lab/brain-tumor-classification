
import io
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input


# ==================================================
# 1. CONFIGURATION
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
            f"Missing model file: {MODEL_PATH.name}. "
            "Keep the model beside app.py."
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
                f"Required model layer '{name}' is missing."
            ) from exc

    return model


# ==================================================
# 3. LAYERCAM VISUALIZATION
# ==================================================

def generate_layercam(image_array, model, class_index):
    """
    LayerCAM uses spatially varying positive gradients
    multiplied by convolutional feature activations.
    """

    base = model.get_layer("resnet50")

    # Intermediate layer with higher spatial resolution.
    conv_layer = base.get_layer("conv4_block6_out")

    feature_model = tf.keras.Model(
        inputs=base.input,
        outputs=[
            conv_layer.output,
            base.output
        ]
    )

    batch = image_array.astype(np.float32)[None, ...]
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

        classifier = model.get_layer("class_predictions")

        if not isinstance(classifier, tf.keras.layers.Dense):
            raise ValueError(
                "Expected class_predictions to be a Dense layer."
            )

        # Pre-activation score for the selected class.
        logits = tf.linalg.matmul(x, classifier.kernel)

        if classifier.use_bias:
            logits = tf.nn.bias_add(logits, classifier.bias)

        class_score = logits[:, class_index]

    gradients = tape.gradient(class_score, conv_outputs)

    if gradients is None:
        return None, None, None

    # LayerCAM: use positive gradients at each spatial location.
    positive_gradients = tf.nn.relu(gradients)

    # Weight each activation by its local positive gradient.
    layercam = tf.reduce_sum(
        positive_gradients * tf.nn.relu(conv_outputs),
        axis=-1
    )[0]

    layercam = layercam.numpy()

    if not np.all(np.isfinite(layercam)):
        return None, None, None

    max_value = float(np.max(layercam))

    if max_value < 1e-8:
        return None, None, None

    layercam = layercam / max_value

    # Resize to image dimensions.
    layercam = cv2.resize(
        layercam,
        (image_array.shape[1], image_array.shape[0]),
        interpolation=cv2.INTER_LINEAR
    )

    layercam = np.clip(layercam, 0, 1)

    # Colour visualization.
    coloured = cv2.applyColorMap(
        np.uint8(layercam * 255),
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

    return layercam, coloured, overlay


# ==================================================
# 4. PREDICT CLASS
# ==================================================

def predict_mri(image, model):
    image = image.convert("RGB").resize(IMG_SIZE)

    original = np.array(image, dtype=np.uint8)

    batch = original.astype(np.float32)[None, ...]

    # Use the same input pipeline as the trained full model.
    probabilities = model.predict(
        batch,
        verbose=0
    )[0]

    if len(probabilities) != len(CLASS_NAMES):
        raise ValueError(
            "Expected four class scores. Check class ordering."
        )

    predicted_index = int(np.argmax(probabilities))

    layercam, coloured, overlay = generate_layercam(
        original,
        model,
        predicted_index
    )

    scores = {
        CLASS_NAMES[i]: float(probabilities[i])
        for i in range(len(CLASS_NAMES))
    }

    return original, scores, layercam, coloured, overlay


# ==================================================
# 5. USER INTERFACE
# ==================================================

st.title("🧠 Brain Tumor MRI Classification")

st.write(
    "Upload a brain MRI to view the predicted class, "
    "class scores, and LayerCAM visualization."
)

st.warning(
    "Educational project only. This model is not a medical "
    "diagnostic tool. The visualization does not establish "
    "tumour boundaries or replace expert assessment."
)

uploaded_file = st.file_uploader(
    "Upload brain MRI",
    type=["jpg", "jpeg", "png"]
)


# ==================================================
# 6. RUN CLASSIFICATION
# ==================================================

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
                    layercam,
                    coloured,
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
                "This is the model's output score, not a guarantee "
                "of diagnostic accuracy."
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
            st.subheader("LayerCAM Visualization")

            if layercam is None:
                st.warning(
                    "A valid LayerCAM explanation could not be "
                    "generated for this image. The class prediction "
                    "is available, but the explanation is unavailable."
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
                        coloured,
                        caption="LayerCAM Heatmap",
                        use_container_width=True
                    )

                with col3:
                    st.image(
                        overlay,
                        caption="LayerCAM Overlay",
                        use_container_width=True
                    )

                st.caption(
                    "Red and yellow indicate stronger relative "
                    "LayerCAM activation. They do not guarantee "
                    "that the highlighted region is tumour tissue."
                )

                buffer = io.BytesIO()
                Image.fromarray(overlay).save(
                    buffer,
                    format="PNG"
                )

                st.download_button(
                    "Download LayerCAM Overlay",
                    data=buffer.getvalue(),
                    file_name="layercam_overlay.png",
                    mime="image/png"
                )

    except Exception as error:
        st.error("An error occurred while processing the MRI.")
        st.exception(error)
