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

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

def predict_mri(image, model):
    image = image.convert("RGB").resize((224, 224))
    original = np.array(image).astype(np.uint8)
    batch = original.astype(np.float32)[None, ...]

    probabilities = model.predict(batch, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))

    base = model.get_layer("resnet50")
    feature_model = tf.keras.Model(
        inputs=base.input,
        outputs=[
            base.get_layer("conv5_block3_out").output,
            base.output
        ]
    )

    processed = preprocess_input(batch)

    with tf.GradientTape() as tape:
        conv_outputs, base_outputs = feature_model(
            processed, training=False
        )
        x = model.get_layer("gap")(base_outputs)
        x = model.get_layer("dropout")(x, training=False)
        x = model.get_layer("dense_features")(x)
        predictions = model.get_layer("class_predictions")(x)
        score = predictions[:, predicted_index]

    gradients = tape.gradient(score, conv_outputs)
    weights = tf.reduce_mean(gradients, axis=(1, 2))

    heatmap = tf.reduce_sum(
        conv_outputs * weights[:, None, None, :],
        axis=-1
    )[0]
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = cv2.resize(heatmap.numpy(), (224, 224))

    colored = cv2.applyColorMap(
        np.uint8(255 * heatmap), cv2.COLORMAP_JET
    )
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original, 0.6, colored, 0.4, 0)

    scores = {
        CLASS_NAMES[i]: float(probabilities[i])
        for i in range(len(CLASS_NAMES))
    }

    return scores, overlay

st.title("🧠 Brain Tumor Classification from MRI Images")
st.write(
    "Upload an MRI image to view the model's predicted class "
    "and Grad-CAM visualization."
)

st.warning(
    "Educational project demonstration only. This model is not "
    "validated for medical diagnosis."
)

uploaded_file = st.file_uploader(
    "Upload a brain MRI image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    left, right = st.columns(2)
    with left:
        st.subheader("Uploaded MRI")
        st.image(image, use_container_width=True)

    if st.button("Classify MRI", type="primary"):
        try:
            with st.spinner("Analyzing image..."):
                model = load_model()
                scores, overlay = predict_mri(image, model)

            predicted_class = max(scores, key=scores.get)

            with right:
                st.subheader("Grad-CAM Visualization")
                st.image(overlay, use_container_width=True)

            st.subheader("Prediction Results")
            st.success(f"Predicted class: {predicted_class}")
            st.write("Model confidence scores:")
            st.bar_chart(scores)

        except Exception as error:
            st.error(f"Prediction failed: {error}")
