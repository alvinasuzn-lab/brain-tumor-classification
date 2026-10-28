import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from tensorflow.keras.applications.resnet50 import preprocess_input

st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="wide"
)

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
MODEL_PATH = Path(__file__).parent / "best_brain_tumor_finetuned.keras"
IMG_SIZE = (224, 224)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. "
            "Place best_brain_tumor_finetuned.keras beside app.py."
        )
    return tf.keras.models.load_model(MODEL_PATH)


def generate_gradcam(image_array, model, class_index):
    """Generate Grad-CAM using the final ResNet50 convolutional layer."""
    resnet = model.get_layer("resnet50")
    conv_layer = resnet.get_layer("conv5_block3_out")

    feature_model = tf.keras.Model(
        inputs=resnet.input,
        outputs=conv_layer.output
    )

    batch = image_array.astype(np.float32)[None, ...]
    processed = preprocess_input(batch)

    with tf.GradientTape() as tape:
        conv_output = feature_model(processed, training=False)
        tape.watch(conv_output)

        # Continue through the model's existing classification head.
        x = tf.keras.layers.GlobalAveragePooling2D()(conv_output)
        x = model.get_layer("dropout")(x, training=False)
        x = model.get_layer("dense_features")(x)
        predictions = model.get_layer("class_predictions")(x)
        score = predictions[:, class_index]

    gradients = tape.gradient(score, conv_output)

    if gradients is None:
        raise ValueError("Could not calculate Grad-CAM gradients.")

    weights = tf.reduce_mean(gradients, axis=(1, 2))
    heatmap = tf.reduce_sum(
        conv_output * weights[:, None, None, :],
        axis=-1
    )[0]

    heatmap = tf.maximum(heatmap, 0)
    maximum = float(tf.reduce_max(heatmap).numpy())

    if maximum <= 1e-8:
        raise ValueError("Grad-CAM produced no positive activation.")

    heatmap = heatmap / maximum
    heatmap = cv2.resize(
        heatmap.numpy(), IMG_SIZE, interpolation=cv2.INTER_LINEAR
    )
    heatmap = np.clip(heatmap, 0, 1)

    colored = cv2.applyColorMap(
        np.uint8(255 * heatmap), cv2.COLORMAP_JET
    )
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(
        image_array, 0.60, colored, 0.40, 0
    )

    return heatmap, overlay


def predict_mri(image, model):
    image = image.convert("RGB").resize(IMG_SIZE)
    original = np.array(image, dtype=np.uint8)

    # The complete model receives raw RGB pixels, as in training.
    probabilities = model.predict(
        original.astype(np.float32)[None, ...],
        verbose=0
    )[0]

    predicted_index = int(np.argmax(probabilities))

    heatmap, overlay = generate_gradcam(
        original, model, predicted_index
    )

    scores = {
        CLASS_NAMES[i]: float(probabilities[i])
        for i in range(len(CLASS_NAMES))
    }

    return original, scores, heatmap, overlay


st.title("🧠 Brain Tumor Classification from MRI Images")

st.write(
    "Upload an MRI image to view the predicted class, confidence scores, "
    "and Grad-CAM visualization."
)

st.warning(
    "Educational project demonstration only. This model is not validated "
    "for medical diagnosis. Grad-CAM highlights image regions influencing "
    "the model; it does not confirm or segment a tumour."
)

uploaded_file = st.file_uploader(
    "Upload a brain MRI image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.subheader("Uploaded MRI")
        st.image(image, use_container_width=True)

        if st.button("Classify MRI", type="primary"):
            with st.spinner("Generating prediction and Grad-CAM..."):
                model = load_model()
                original, scores, heatmap, overlay = predict_mri(
                    image, model
                )

            predicted_class = max(scores, key=scores.get)
            confidence = scores[predicted_class]

            st.subheader("Prediction Results")
            st.success(
                f"Predicted class: {predicted_class} "
                f"(model confidence: {confidence:.2%})"
            )

            st.write("Model confidence scores:")
            st.bar_chart(scores)

            st.subheader("Grad-CAM Visualizations")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.caption("Original MRI")
                st.image(original, use_container_width=True)

            with col2:
                st.caption("Grad-CAM Heatmap")
                st.image(heatmap, clamp=True, use_container_width=True)

            with col3:
                st.caption("Grad-CAM Overlay")
                st.image(overlay, use_container_width=True)

    except Exception as error:
        st.error(f"Prediction failed: {error}")
