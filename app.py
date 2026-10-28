\from pathlib import Path

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

# Alphabetical folder order: ['glioma', 'meningioma', 'notumor', 'pituitary']
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary Tumor"]

# Model pipeline: input -> data_augmentation -> preprocess_input -> resnet50
#                 -> gap -> dropout -> dense_features -> class_predictions
# The saved model expects RAW 0-255 RGB pixels.

# Grad-CAM layer: final ResNet50 feature map (feeds GAP)
CAM_LAYERS = {
    "conv5_block3_out (7x7)": "conv5_block3_out",
}
USE_LOGIT_TARGET = True  # explain the pre-softmax class score


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
        gdown.download(id=MODEL_ID, output=str(model_path), quiet=False, fuzzy=True)

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

    # One backbone-with-feature-map model per candidate layer.
    # Each returns (feature map, backbone output).
    grad_models = {}
    for label, layer_name in CAM_LAYERS.items():
        try:
            conv_layer = resnet.get_layer(layer_name)
        except ValueError:
            continue
        grad_models[label] = tf.keras.Model(
            inputs=resnet.inputs,
            outputs=[conv_layer.output, resnet.output],
        )

    if not grad_models:
        raise RuntimeError("None of the Grad-CAM layers were found in resnet50.")

    head = {
        "gap": model.get_layer("gap"),
        "dropout": model.get_layer("dropout"),
        "dense": model.get_layer("dense_features"),
        "classifier": model.get_layer("class_predictions"),
    }
    return model, grad_models, head


# --------------------------------------------------
# IMAGE PREPARATION
# --------------------------------------------------

def prepare_image(image: Image.Image):
    """Returns the original RGB image and a 1x224x224x3 RAW (0-255) batch."""
    original = np.asarray(image.convert("RGB"), dtype=np.uint8)
    resized = tf.image.resize(
        tf.convert_to_tensor(original, dtype=tf.float32), IMG_SIZE
    ).numpy()
    return original, resized[None, ...].astype(np.float32)


def backbone_preprocess(raw_batch):
    return tf.keras.applications.resnet50.preprocess_input(raw_batch)


# --------------------------------------------------
# PREDICTION + GRAD-CAM / GRAD-CAM++
# --------------------------------------------------

def compute_cam(conv_output, gradients, method):
    """Return an (H, W) class activation map from feature maps and gradients."""
    if method == "Grad-CAM++":
        g2 = tf.square(gradients)
        g3 = g2 * gradients
        denom = 2.0 * g2 + tf.reduce_sum(conv_output * g3, axis=(1, 2), keepdims=True)
        denom = tf.where(tf.abs(denom) > 1e-12, denom, tf.ones_like(denom))
        alpha = g2 / denom
        weights = tf.reduce_sum(alpha * tf.nn.relu(gradients), axis=(1, 2))
    else:  # standard Grad-CAM
        weights = tf.reduce_mean(gradients, axis=(1, 2))

    cam = tf.reduce_sum(conv_output * weights[:, None, None, :], axis=-1)
    return tf.nn.relu(cam)[0].numpy()


def predict_and_gradcam(raw_batch, model, grad_models, head, layer_label, method):
    x_raw = tf.convert_to_tensor(raw_batch, dtype=tf.float32)

    # 1) The real prediction comes from the full saved model.
    probabilities = model(x_raw, training=False).numpy()[0]
    predicted_index = int(np.argmax(probabilities))

    # 2) CAM path: preprocess_input -> backbone -> same head layers.
    grad_model = grad_models[layer_label]
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
            and getattr(classifier.activation, "__name__", "") in ("softmax", "linear")
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

    cam = compute_cam(conv_output, gradients, method)

    peak = float(cam.max())
    if not np.all(np.isfinite(cam)) or peak <= 1e-8:
        raise RuntimeError("No positive activation for this class.")
    cam = cam / peak

    # 3) Safety check: the CAM path must reproduce the real output.
    paths_match = bool(
        np.allclose(probabilities, cam_probs.numpy()[0], rtol=1e-3, atol=1e-4)
    )
    if logits is not None:
        paths_match = paths_match and bool(
            np.allclose(
                probabilities, tf.nn.softmax(logits[0]).numpy(), rtol=1e-3, atol=1e-4
            )
        )
    max_diff = float(np.max(np.abs(probabilities - cam_probs.numpy()[0])))

    return probabilities, predicted_index, cam, paths_match, max_diff


# --------------------------------------------------
# HEAD / BRAIN MASK
# --------------------------------------------------

def head_mask(original_rgb):
    """Soft 0-1 mask of the imaged head, so heat is never drawn on the black
    background or at the image border (a known CNN padding artefact)."""
    gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    thr = max(8.0, 0.12 * float(np.percentile(gray, 99)))
    binary = (gray > thr).astype(np.uint8) * 255

    h, w = binary.shape
    k = max(3, int(min(h, w) * 0.03) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # keep the largest connected component
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        binary = ((labels == largest) * 255).astype(np.uint8)

    # fill holes (flood the outside, invert)
    flood = binary.copy()
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 255)
    binary = binary | cv2.bitwise_not(flood)

    area = float(binary.astype(bool).mean())
    if area < 0.05 or area > 0.98:  # unreliable mask -> do not use it
        return np.ones((h, w), np.float32)

    soft = cv2.GaussianBlur(binary.astype(np.float32) / 255.0, (0, 0), sigmaX=max(h, w) / 80)
    return np.clip(soft, 0.0, 1.0)


# --------------------------------------------------
# HEATMAP AND OVERLAY
# --------------------------------------------------

def _normalize(x):
    peak = float(x.max())
    return x / peak if peak > 1e-6 else x


def create_visualizations(original_image, cam, use_mask, focus, single_region, min_level):
    h, w = original_image.shape[:2]

    # smooth upsampling
    big = cv2.resize(cam.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    big = np.clip(big, 0.0, 1.0)
    big = cv2.GaussianBlur(big, (0, 0), sigmaX=max(h, w) / 100)

    # remove border / background activation
    if use_mask:
        big = big * head_mask(original_image)
    big = _normalize(big)

    # focus: drop weak, diffuse activation so only the strongest evidence stays
    if focus > 0:
        big = np.clip((big - focus) / (1.0 - focus), 0.0, 1.0)

    # keep only the connected region that contains the strongest activation
    if single_region and big.max() > 0:
        n, labels = cv2.connectedComponents((big > 0).astype(np.uint8), connectivity=8)
        py, px = np.unravel_index(np.argmax(big), big.shape)
        big = big * (labels == labels[py, px])

    # soften edges of the remaining blob
    big = cv2.GaussianBlur(big, (0, 0), sigmaX=max(h, w) / 60)
    big = _normalize(big)

    heat_bgr = cv2.applyColorMap(np.uint8(big * 255), cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)

    alpha = np.clip((big - min_level) / (1.0 - min_level + 1e-6), 0.0, 1.0)[..., None] * 0.65
    overlay = original_image.astype(np.float32) * (1 - alpha) + heat_rgb.astype(np.float32) * alpha
    return heat_rgb, np.uint8(np.clip(overlay, 0, 255))


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

with st.sidebar:
    st.header("Grad-CAM settings")
    method = st.selectbox("Method", ["Grad-CAM", "Grad-CAM++"], index=0)
    layer_label = next(iter(CAM_LAYERS))
    use_mask = st.checkbox("Restrict heat to the head region", value=True)
    focus = st.slider("Focus (hide weak activation)", 0.0, 0.8, 0.45, 0.05,
                      help="Higher = only the strongest evidence is shown.")
    single_region = st.checkbox("Show only the strongest region", value=True,
                                help="Untick to also see secondary regions.")
    min_level = st.slider("Overlay threshold", 0.0, 0.9, 0.15, 0.05,
                          help="Activations below this level are not coloured in the overlay.")

uploaded_file = st.file_uploader("Upload MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        original_image, raw_batch = prepare_image(image)

        with st.spinner("Analyzing MRI image..."):
            model, grad_models, head = load_model()
            if layer_label not in grad_models:
                layer_label = next(iter(grad_models))
            probabilities, predicted_index, cam, paths_match, max_diff = (
                predict_and_gradcam(raw_batch, model, grad_models, head, layer_label, method)
            )
            heatmap_rgb, overlay = create_visualizations(
                original_image, cam, use_mask, focus, single_region, min_level
            )

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
            col2.image(heatmap_rgb, caption=f"{method} Heatmap", use_container_width=True)
            col3.image(overlay, caption="Heatmap Overlay", use_container_width=True)
            st.caption(
                f"{method} shows regions that influenced the predicted class score "
                f"at {cam.shape[0]} x {cam.shape[1]} resolution, upsampled for display. "
                "It does not outline the tumour and does not prove accurate tumour "
                "localization."
            )

        st.subheader("Class Scores")
        for i, name in enumerate(CLASS_NAMES):
            value = float(probabilities[i])
            st.write(f"**{name}: {value * 100:.2f}%**")
            st.progress(float(np.clip(value, 0.0, 1.0)))

        with st.expander("Technical details"):
            st.write("Input: raw 0-255 RGB, resized to 224 x 224")
            st.write("Model pipeline: augmentation (off at inference) -> preprocess_input -> ResNet50")
            st.write("CAM method:", method)
            st.write("CAM layer:", CAM_LAYERS[layer_label], f"({cam.shape[0]}x{cam.shape[1]})")
            st.write("Head-region mask applied:", use_mask)
            st.write("Focus:", focus, "| strongest region only:", single_region)
            st.write(
                "Gradient target:",
                "pre-softmax class score" if USE_LOGIT_TARGET else "class probability",
            )
            st.write("CAM path matches model output:", paths_match)
            st.write(f"Max probability difference: {max_diff:.2e}")

    except Exception as error:
        st.error("The image could not be analyzed.")
        st.exception(error)
else:
    st.info("Upload an MRI image to begin.")
