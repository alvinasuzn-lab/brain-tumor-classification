from pathlib import Path

import cv2
import gdown
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# --------------------------------------------------
# CONFIGURATION (matches the training notebook)
# --------------------------------------------------

st.set_page_config(
    page_title="Brain Tumour MRI Classification",
    page_icon="🧠",
    layout="wide",
)

MODEL_ID = "1sO3yNwEIu9oCfP3W_wlC6r07fhv0hEYI"
MODEL_NAME = "best_brain_tumor_finetuned.keras"
IMG_SIZE = (224, 224)

# Folder names sorted alphabetically = class_names printed in the notebook:
# ['glioma', 'meningioma', 'notumor', 'pituitary']
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary Tumor"]

# The saved model was built as:
#   input -> data_augmentation -> preprocess_input -> resnet50 -> gap
#         -> dropout -> dense_features -> class_predictions
# So the model itself expects RAW 0-255 RGB pixels.
GRADCAM_LAYER = "conv5_block3_out"  # final ResNet50 feature map (feeds GAP)
USE_LOGIT_TARGET = True             # explain the pre-softmax class score


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource(show_spinner="Downloading and loading model (first run only)...")
def load_model():
    model_dir = Path.home() / ".brain_tumor_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / MODEL_NAME

    if not model_path.exists() or model_path.stat().st_size < 1_000_000:
        if model_path.exists():
            model_path.unlink()
        gdown.download(
            id=MODEL_ID, output=str(model_path), quiet=False, fuzzy=True
        )

    if not model_path.exists() or model_path.stat().st_size < 1_000_000:
        raise RuntimeError(
            "Model download failed. Check the Google Drive file ID and that "
            "the file is shared as 'Anyone with the link - Viewer'."
        )

    model = tf.keras.models.load_model(str(model_path), compile=False)

    for name in [
        "data_augmentation", "resnet50", "gap",
        "dropout", "dense_features", "class_predictions",
    ]:
        model.get_layer(name)  # raises a clear error if a layer is missing

    resnet = model.get_layer("resnet50")
    conv_layer = resnet.get_layer(GRADCAM_LAYER)

    # Backbone that returns (final conv feature map, backbone output)
    grad_model = tf.keras.Model(
        inputs=resnet.inputs,
        outputs=[conv_layer.output, resnet.output],
    )

    head = {
        "gap": model.get_layer("gap"),
        "dropout": model.get_layer("dropout"),
        "dense": model.get_layer("dense_features"),
        "classifier": model.get_layer("class_predictions"),
    }
    return model, grad_model, head


# --------------------------------------------------
# IMAGE PREPARATION
# --------------------------------------------------

def prepare_image(image: Image.Image):
    """Returns the original RGB image and a 1x224x224x3 RAW (0-255) batch.

    Resizing uses tf.image.resize (bilinear), the same method used by
    image_dataset_from_directory during training.
    """
    original = np.asarray(image.convert("RGB"), dtype=np.uint8)
    resized = tf.image.resize(
        tf.convert_to_tensor(original, dtype=tf.float32), IMG_SIZE
    ).numpy()
    return original, resized[None, ...].astype(np.float32)


def backbone_preprocess(raw_batch):
    # Same function used inside the model during training.
    return tf.keras.applications.resnet50.preprocess_input(raw_batch)


# --------------------------------------------------
# PREDICTION + GRAD-CAM
# --------------------------------------------------

def predict_and_gradcam(raw_batch, model, grad_model, head):
    x_raw = tf.convert_to_tensor(raw_batch, dtype=tf.float32)

    # 1) The real prediction comes from the full saved model.
    probabilities = model(x_raw, training=False).numpy()[0]
    predicted_index = int(np.argmax(probabilities))

    # 2) Grad-CAM path: preprocess_input -> backbone -> same head layers.
    #    (data_augmentation does nothing at inference time.)
    backbone_input = backbone_preprocess(x_raw)

    with tf.GradientTape() as tape:
        conv_output, backbone_output = grad_model(backbone_input, training=False)

        h = head["gap"](backbone_output)
        h = head["dropout"](h, training=False)
        h = head["dense"](h)
        cam_probs = head["classifier"](h)

        classifier = head["classifier"]
        can_use_logits = (
            USE_LOGIT_TARGET
            and isinstance(classifier, tf.keras.layers.Dense)
            and getattr(classifier.activation, "__name__", "")
            in ("softmax", "linear")
        )
        if can_use_logits:
            logits = tf.matmul(h, classifier.kernel) + classifier.bias
            target_score = logits[:, predicted_index]
        else:
            logits = None
            target_score = cam_probs[:, predicted_index]

    gradients = tape.gradient(target_score, conv_output)
    if gradients is None:
        raise RuntimeError("Gradients could not be calculated.")

    weights = tf.reduce_mean(gradients, axis=(1, 2))
    cam = tf.reduce_sum(conv_output * weights[:, None, None, :], axis=-1)
    cam = tf.nn.relu(cam)[0].numpy()

    peak = float(cam.max())
    if not np.all(np.isfinite(cam)) or peak <= 1e-8:
        raise RuntimeError("No positive Grad-CAM activation for this class.")
    cam = cam / peak

    # 3) Safety check: the Grad-CAM path must reproduce the real output.
    paths_match = bool(
        np.allclose(probabilities, cam_probs.numpy()[0], rtol=1e-3, atol=1e-4)
    )
    if logits is not None:
        paths_match = paths_match and bool(
            np.allclose(
                probabilities,
                tf.nn.softmax(logits[0]).numpy(),
                rtol=1e-3,
                atol=1e-4,
            )
        )
    max_diff = float(np.max(np.abs(probabilities - cam_probs.numpy()[0])))

    return probabilities, predicted_index, cam, paths_match, max_diff


# --------------------------------------------------
# HEATMAP AND OVERLAY
# --------------------------------------------------

def create_visualizations(original_image, heatmap):
    h, w = original_image.shape[:2]
    big = cv2.resize(heatmap.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    big = np.clip(big, 0.0, 1.0)

    heat_bgr = cv2.applyColorMap(np.uint8(big * 255), cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original_image, 0.60, heat_rgb, 0.40, 0)
    return heat_rgb, overlay


# --------------------------------------------------
# USER INTERFACE
# --------------------------------------------------

st.title("🧠 Brain Tumour MRI Classification")
st.write(
    "Upload a brain MRI image to view the model prediction, class scores "
    "and Grad-CAM visualization."
)
st.warning(
    "Educational research prototype only. This application is not a "
    "medical diagnostic tool."
)

uploaded_file = st.file_uploader("Upload MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        original_image, raw_batch = prepare_image(image)

        with st.spinner("Analyzing MRI image..."):
            model, grad_model, head = load_model()
            probabilities, predicted_index, heatmap, paths_match, max_diff = (
                predict_and_gradcam(raw_batch, model, grad_model, head)
            )
            heatmap_rgb, overlay = create_visualizations(original_image, heatmap)

        st.success("Analysis completed.")

        c1, c2 = st.columns(2)
        c1.metric("Predicted class", CLASS_NAMES[predicted_index])
        c2.metric("Model confidence", f"{probabilities[predicted_index] * 100:.2f}%")

        st.subheader("MRI and Grad-CAM Results")

        if not paths_match:
            st.error(
                "The Grad-CAM path does not reproduce the model output "
                f"(max difference {max_diff:.5f}), so the heatmap is hidden. "
                "The prediction above comes from the full model."
            )
            st.image(original_image, caption="Original MRI", width=350)
        else:
            col1, col2, col3 = st.columns(3)
            col1.image(original_image, caption="Original MRI", use_container_width=True)
            col2.image(heatmap_rgb, caption="Grad-CAM Heatmap", use_container_width=True)
            col3.image(overlay, caption="Heatmap Overlay", use_container_width=True)
            st.caption(
                f"Grad-CAM shows regions that influenced the predicted class score "
                f"at {heatmap.shape[0]} x {heatmap.shape[1]} resolution, upsampled for "
                "display. It does not outline the tumour and does not prove "
                "accurate tumour localization."
            )

        st.subheader("Class Scores")
        for i, name in enumerate(CLASS_NAMES):
            value = float(probabilities[i])
            st.write(f"**{name}: {value * 100:.2f}%**")
            st.progress(float(np.clip(value, 0.0, 1.0)))

        with st.expander("Technical details"):
            st.write("Input: raw 0-255 RGB, resized to 224 x 224")
            st.write("Model pipeline: augmentation (off at inference) -> preprocess_input -> ResNet50")
            st.write("Grad-CAM layer:", GRADCAM_LAYER)
            st.write(
                "Gradient target:",
                "pre-softmax class score" if USE_LOGIT_TARGET else "class probability",
            )
            st.write("Grad-CAM path matches model output:", paths_match)
            st.write(f"Max probability difference: {max_diff:.2e}")

    except Exception as error:
        st.error("The image could not be analyzed.")
        st.exception(error)
else:
    st.info("Upload an MRI image to begin.")
