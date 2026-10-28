
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

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE = (224, 224)

MODEL_PATH = (
    Path(__file__).parent / "best_brain_tumor_finetuned.keras"
)


# =====================================================
# 2. LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find {MODEL_PATH.name}. "
            "Keep the model beside app.py."
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    for name in [
        "resnet50", "gap", "dropout",
        "dense_features", "class_predictions"
    ]:
        try:
            model.get_layer(name)
        except ValueError as exc:
            raise ValueError(
                f"Required model layer '{name}' is missing."
            ) from exc

    return model


# =====================================================
# 3. GRAYSCALE GRAD-CAM
# =====================================================

def generate_gradcam(image_array, model, class_index):
    """
    Standard Grad-CAM using a higher-resolution intermediate
    ResNet50 layer. Returns grayscale activation and overlay.
    """

    base = model.get_layer("resnet50")
    conv_layer = base.get_layer("conv4_block6_out")
    classifier = model.get_layer("class_predictions")

    feature_model = tf.keras.Model(
        inputs=base.input,
        outputs=[conv_layer.output, base.output]
    )

    batch = image_array.astype(np.float32)[None, ...]
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

        if not isinstance(classifier, tf.keras.layers.Dense):
            raise ValueError(
                "The class_predictions layer is not Dense."
            )

        logits = tf.linalg.matmul(x, classifier.kernel)

        if classifier.use_bias:
            logits = tf.nn.bias_add(logits, classifier.bias)

        class_score = logits[:, class_index]

    gradients = tape.gradient(class_score, conv_outputs)

    if gradients is None:
        return None, None, (
            "Grad-CAM gradients are unavailable. "
            "The model's gradient connection needs inspection."
        )

    # Standard Grad-CAM channel weights.
    weights = tf.reduce_mean(
        gradients,
        axis=(1, 2),
        keepdims=True
    )

    cam = tf.reduce_sum(
        conv_outputs * weights,
        axis=-1
    )[0]

    cam = tf.nn.relu(cam).numpy()

    if not np.all(np.isfinite(cam)):
        return None, None, "Invalid numerical values in the heatmap."

    maximum = float(cam.max())

    if maximum < 1e-8:
        return None, None, (
            "No meaningful positive Grad-CAM activation was found "
            "for this image."
        )

    cam = cam / maximum

    # Resize smoothly to the MRI dimensions.
    cam = cv2.resize(
        cam,
        (image_array.shape[1], image_array.shape[0]),
        interpolation=cv2.INTER_CUBIC
    )

    cam = np.clip(cam, 0, 1)

    # Improve display contrast using only existing activation values.
    # This changes appearance, not the model's localization.
    positive = cam[cam > 0]

    if positive.size > 0:
        low = float(np.percentile(positive, 5))
        high = float(np.percentile(positive, 99))

        if high > low:
            cam = np.clip((cam - low) / (high - low), 0, 1)

    # Convert activation into a visible grayscale image.
    grayscale_heatmap = np.uint8(cam * 255)

    # Create a grayscale MRI and overlay.
    grayscale_mri = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2GRAY
    )

    overlay = cv2.addWeighted(
        grayscale_mri,
        0.65,
        grayscale_heatmap,
        0.35,
        0
    )

    return grayscale_heatmap, overlay, None


# =====================================================
# 4. PREDICTION
# =====================================================

def predict_mri(image, model):
    image = image.convert("RGB").resize(IMG_SIZE)
    original = np.array(image, dtype=np.uint8)

    probabilities = model.predict(
        original.astype(np.float32)[None, ...],
        verbose=0
    )[0]

    if len(probabilities) != len(CLASS_NAMES):
        raise ValueError(
            "The model did not return four class scores."
        )

    predicted_index = int(np.argmax(probabilities))

    heatmap, overlay, cam_error = generate_gradcam(
        original,
        model,
        predicted_index
    )

    scores = {
        CLASS_NAMES[i]: float(probabilities[i])
        for i in range(len(CLASS_NAMES))
    }

    return original, scores, heatmap, overlay, cam_error


# =====================================================
# 5. STREAMLIT INTERFACE
# =====================================================

st.title("🧠 Brain Tumor MRI Classification")

st.write(
    "Upload a brain MRI to view the predicted class, "
    "class scores, and grayscale Grad-CAM."
)

st.warning(
    "Educational project only. This is not a medical diagnostic "
    "tool. Grad-CAM does not establish tumour boundaries."
)

uploaded_file = st.file_uploader(
    "Upload brain MRI",
    type=["jpg", "jpeg", "png"]
)


# =====================================================
# 6. DISPLAY RESULTS
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

                original, scores, heatmap, overlay, cam_error = (
                    predict_mri(image, model)
                )

            predicted_class = max(scores, key=scores.get)
            confidence = scores[predicted_class] * 100

            st.divider()
            st.subheader("Prediction Results")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Predicted class", predicted_class.title())

            with col2:
                st.metric("Model output score", f"{confidence:.2f}%")

            st.caption(
                "The score is the model's output, not a guarantee "
                "of medical diagnostic accuracy."
            )

            st.subheader("Class Scores")

            st.bar_chart({
                name.title(): value * 100
                for name, value in scores.items()
            })

            for name, value in scores.items():
                st.write(f"**{name.title()}:** {value * 100:.2f}%")

            st.divider()
            st.subheader("Grayscale Grad-CAM")

            if cam_error:
                st.warning(cam_error)
                st.image(
                    original,
                    caption="Original MRI — heatmap unavailable",
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
                        heatmap,
                        caption="Grayscale Activation",
                        clamp=True,
                        use_container_width=True
                    )

                with col3:
                    st.image(
                        overlay,
                        caption="Grayscale Overlay",
                        clamp=True,
                        use_container_width=True
                    )

                buffer = io.BytesIO()
                Image.fromarray(overlay).save(buffer, format="PNG")

                st.download_button(
                    "Download Grayscale Overlay",
                    data=buffer.getvalue(),
                    file_name="grayscale_gradcam.png",
                    mime="image/png"
                )

    except Exception as error:
        st.error("An error occurred while processing the MRI.")
        st.exception(error)
