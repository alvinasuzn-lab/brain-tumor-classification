
from pathlib import Path
import io

import cv2
import gdown
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input


# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Brain MRI Classifier",
    page_icon="🧠",
    layout="wide",
)

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

MODEL_ID = "1sO3yNwEIu9oCfP3W_wlC6r07fhv0hEYI"
MODEL_NAME = "best_brain_tumor_finetuned.keras"

MODEL_DIR = Path.home() / ".brain_tumor_model"
MODEL_PATH = MODEL_DIR / MODEL_NAME

st.title("🧠 Brain MRI Classification")
st.caption("Fine-tuned ResNet50 | Four-class classification")

st.write(
    "Upload a brain MRI image to obtain a model prediction. "
    "Grad-CAM is optional and provides an approximate explanation."
)


# =====================================================
# DOWNLOAD AND LOAD MODEL
# =====================================================
@st.cache_resource(show_spinner="Preparing the trained model...")
def load_model():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Reuse an existing model if it loads successfully.
    if MODEL_PATH.is_file() and MODEL_PATH.stat().st_size > 1_000_000:
        try:
            with MODEL_PATH.open("rb") as file:
                header = file.read(200)

            if not header.startswith(
                b"version https://git-lfs.github.com/spec/v1"
            ):
                return tf.keras.models.load_model(
                    str(MODEL_PATH), compile=False
                )
        except Exception:
            pass

        MODEL_PATH.unlink(missing_ok=True)

    # Download the model from Google Drive.
    url = f"https://drive.google.com/uc?id={MODEL_ID}"

    try:
        result = gdown.download(
            url=url,
            output=str(MODEL_PATH),
            quiet=False,
            resume=False,
        )
    except Exception as error:
        MODEL_PATH.unlink(missing_ok=True)
        raise RuntimeError(
            f"Google Drive download failed: {error}"
        ) from error

    if not result or not MODEL_PATH.is_file():
        MODEL_PATH.unlink(missing_ok=True)
        raise RuntimeError(
            "Google Drive did not provide the model file. "
            "Check the sharing permissions and download availability."
        )

    # Check for a likely incomplete download or an error page.
    if MODEL_PATH.stat().st_size < 1_000_000:
        MODEL_PATH.unlink(missing_ok=True)
        raise RuntimeError(
            "The downloaded file is too small to be the model. "
            "Google Drive may have returned an access or quota error."
        )

    with MODEL_PATH.open("rb") as file:
        header = file.read(200)

    if header.startswith(b"version https://git-lfs.github.com/spec/v1"):
        MODEL_PATH.unlink(missing_ok=True)
        raise RuntimeError(
            "The downloaded file is a Git LFS pointer, not the actual model."
        )

    try:
        return tf.keras.models.load_model(
            str(MODEL_PATH), compile=False
        )
    except Exception as error:
        MODEL_PATH.unlink(missing_ok=True)
        raise RuntimeError(
            "The downloaded file could not be loaded as a Keras model."
        ) from error


try:
    model = load_model()
except Exception as error:
    st.error("Unable to prepare the trained model.")
    st.code(str(error))
    st.info(
        "Check that Google Drive General access is set to "
        "'Anyone with the link — Viewer'. If access is correct, "
        "Google Drive may be blocking automated downloads or "
        "enforcing a download limit."
    )
    st.stop()


# =====================================================
# PREPROCESS IMAGE
# =====================================================
def prepare_image(image):
    image = image.convert("RGB").resize((224, 224))
    array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(array, axis=0)


# =====================================================
# PREDICTION
# =====================================================
def predict_image(batch):
    predictions = model.predict(batch, verbose=0)[0]

    if len(predictions) != len(CLASS_NAMES):
        raise ValueError(
            f"Expected 4 class scores, received {len(predictions)}."
        )

    index = int(np.argmax(predictions))
    return CLASS_NAMES[index], predictions, index


# =====================================================
# GRAD-CAM
# =====================================================
def generate_gradcam(batch, class_index):
    try:
        backbone = model.get_layer("resnet50")
        layer_names = {layer.name for layer in backbone.layers}

        if "conv5_block3_out" in layer_names:
            target_name = "conv5_block3_out"
        elif "conv4_block6_out" in layer_names:
            target_name = "conv4_block6_out"
        else:
            return None, None

        feature_model = tf.keras.Model(
            inputs=backbone.input,
            outputs=[
                backbone.get_layer(target_name).output,
                backbone.output,
            ],
        )

        gap = model.get_layer("gap")
        dropout = model.get_layer("dropout")
        dense = model.get_layer("dense_features")
        output_layer = model.get_layer("class_predictions")

        processed = preprocess_input(
            tf.convert_to_tensor(batch, dtype=tf.float32)
        )

        with tf.GradientTape() as tape:
            conv_features, backbone_output = feature_model(
                processed, training=False
            )

            pooled = gap(backbone_output)
            dropped = dropout(pooled, training=False)
            features = dense(dropped)

            logits = tf.linalg.matmul(features, output_layer.kernel)

            if output_layer.use_bias:
                logits = logits + output_layer.bias

            score = logits[:, class_index]

        gradients = tape.gradient(score, conv_features)

        if gradients is None:
            return None, None

        weights = tf.reduce_mean(
            gradients, axis=(1, 2), keepdims=True
        )

        cam = tf.reduce_sum(
            weights * conv_features, axis=-1
        )[0]

        cam = tf.nn.relu(cam).numpy()

        if not np.isfinite(cam).all() or cam.max() <= 1e-8:
            return None, None

        cam = cam / (cam.max() + 1e-8)
        cam = cv2.resize(cam, (224, 224))

        activation = np.uint8(np.clip(cam, 0, 1) * 255)

        original = batch[0].astype(np.uint8)
        gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)

        overlay = cv2.addWeighted(
            gray, 0.65, activation, 0.35, 0
        )

        return activation, overlay

    except Exception as error:
        st.warning(f"Grad-CAM could not be generated: {error}")
        return None, None


# =====================================================
# USER INTERFACE
# =====================================================
uploaded_file = st.file_uploader(
    "Upload brain MRI",
    type=["png", "jpg", "jpeg"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded MRI", width=400)

        show_gradcam = st.checkbox(
            "Generate grayscale Grad-CAM",
            value=False,
        )

        if st.button("Analyze MRI", type="primary"):
            with st.spinner("Analyzing MRI..."):
                batch = prepare_image(image)
                predicted_class, scores, index = predict_image(batch)

            confidence = float(scores[index])

            st.divider()
            st.subheader("Prediction")
            st.success(f"Predicted class: {predicted_class.upper()}")
            st.metric("Model score", f"{confidence * 100:.2f}%")

            st.caption(
                "This is a model prediction, not a medical diagnosis."
            )

            with st.expander("View all class scores"):
                for name, score in zip(CLASS_NAMES, scores):
                    st.write(f"**{name.capitalize()}**")
                    st.progress(float(np.clip(score, 0, 1)))
                    st.caption(f"{float(score) * 100:.2f}%")

            if show_gradcam:
                with st.spinner("Generating Grad-CAM..."):
                    activation, overlay = generate_gradcam(batch, index)

                if activation is not None:
                    st.subheader("Grayscale Grad-CAM")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.image(image, caption="Original MRI")

                    with col2:
                        st.image(
                            activation,
                            caption="Activation map",
                            clamp=True,
                        )

                    with col3:
                        st.image(
                            overlay,
                            caption="Grayscale overlay",
                            clamp=True,
                        )

                    buffer = io.BytesIO()
                    Image.fromarray(overlay).save(buffer, format="PNG")

                    st.download_button(
                        "Download Grad-CAM",
                        data=buffer.getvalue(),
                        file_name="brain_mri_gradcam.png",
                        mime="image/png",
                    )

                    st.info(
                        "Grad-CAM highlights approximate regions that "
                        "influenced the prediction. It does not show "
                        "confirmed tumour boundaries."
                    )
                else:
                    st.warning(
                        "Prediction succeeded, but Grad-CAM "
                        "could not be generated."
                    )

    except Exception as error:
        st.error("Unable to process the MRI image.")
        st.exception(error)


st.divider()
st.caption("Academic project | Not intended for clinical diagnosis")
