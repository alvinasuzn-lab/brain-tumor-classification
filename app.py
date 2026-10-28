
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

from PIL import Image
from pathlib import Path
from tensorflow.keras.applications.resnet50 import preprocess_input


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

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


# --------------------------------------------------
# 2. LOAD TRAINED MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}. "
            "Make sure the model is in the same repository "
            "folder as app.py."
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
        except ValueError:
            raise ValueError(
                f"Expected model layer '{layer_name}' was not found. "
                "Check the layer names in your saved model."
            )

    return model


# --------------------------------------------------
# 3. GENERATE GRAD-CAM HEATMAP
# --------------------------------------------------

def generate_gradcam(image_array, model, class_index):
    """
    Generate Grad-CAM for the predicted class.

    image_array:
        RGB image with shape (224, 224, 3), values 0-255.

    class_index:
        Index of the predicted class.
    """

    base = model.get_layer("resnet50")

    conv_layer = base.get_layer("conv5_block3_out")

    # Return both the convolutional feature map and
    # the output of the ResNet50 feature extractor.
    feature_model = tf.keras.Model(
        inputs=base.input,
        outputs=[
            conv_layer.output,
            base.output
        ]
    )

    batch = image_array.astype(np.float32)[None, ...]

    # The nested ResNet50 branch expects its own
    # ResNet50 preprocessing.
    processed = preprocess_input(batch)

    with tf.GradientTape() as tape:
        conv_outputs, base_outputs = feature_model(
            processed,
            training=False
        )

        # Use the original trained classification head.
        x = model.get_layer("gap")(base_outputs)
        x = model.get_layer("dropout")(x, training=False)
        x = model.get_layer("dense_features")(x)
        predictions = model.get_layer("class_predictions")(x)

        class_score = predictions[:, class_index]

    gradients = tape.gradient(class_score, conv_outputs)

    if gradients is None:
        raise ValueError(
            "Could not calculate Grad-CAM gradients. "
            "Please check the model architecture."
        )

    # Average gradients spatially to obtain channel weights.
    weights = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    # Weighted combination of convolutional feature maps.
    heatmap = tf.reduce_sum(
        conv_outputs * weights[:, None, None, :],
        axis=-1
    )[0]

    # Retain positive activations.
    heatmap = tf.maximum(heatmap, 0)

    max_value = float(tf.reduce_max(heatmap).numpy())

    if max_value < 1e-8:
        raise ValueError(
            "The Grad-CAM activation is nearly zero "
            "for this image."
        )

    heatmap = heatmap / max_value
    heatmap = heatmap.numpy()

    # Resize to the MRI dimensions.
    heatmap = cv2.resize(
        heatmap,
        (image_array.shape[1], image_array.shape[0]),
        interpolation=cv2.INTER_CUBIC
    )

    heatmap = np.clip(heatmap, 0, 1)

    # Create a colour heatmap.
    colored_heatmap = cv2.applyColorMap(
        np.uint8(255 * heatmap),
        cv2.COLORMAP_JET
    )

    colored_heatmap = cv2.cvtColor(
        colored_heatmap,
        cv2.COLOR_BGR2RGB
    )

    # Blend the heatmap with the original MRI.
    overlay = cv2.addWeighted(
        image_array,
        0.65,
        colored_heatmap,
        0.35,
        0
    )

    return heatmap, overlay


# --------------------------------------------------
# 4. PREDICT MRI CLASS
# --------------------------------------------------

def predict_mri(image, model):
    image = image.convert("RGB").resize(IMG_SIZE)

    original = np.array(image, dtype=np.uint8)

    batch = original.astype(np.float32)[None, ...]

    # The complete saved model receives raw RGB pixels.
    # Its own preprocessing layer is expected to handle
    # preprocessing during prediction.
    probabilities = model.predict(
        batch,
        verbose=0
    )[0]

    if len(probabilities) != len(CLASS_NAMES):
        raise ValueError(
            "The model output does not match the four "
            "expected classes. Check the class order."
        )

    predicted_index = int(np.argmax(probabilities))

    heatmap, overlay = generate_gradcam(
        original,
        model,
        predicted_index
    )

    scores = {
        CLASS_NAMES[i]: float(probabilities[i])
        for i in range(len(CLASS_NAMES))
    }

    return original, scores, heatmap, overlay


# --------------------------------------------------
# 5. STREAMLIT USER INTERFACE
# --------------------------------------------------

st.title("🧠 Brain Tumor MRI Classification")

st.write(
    "Upload a brain MRI image to view the model's "
    "predicted class, confidence scores, and Grad-CAM "
    "visualization."
)

st.warning(
    "Educational project demonstration only. "
    "This AI model is not a medical diagnostic tool. "
    "Predictions and Grad-CAM visualizations must not "
    "be used as a substitute for professional medical advice."
)

uploaded_file = st.file_uploader(
    "Upload a brain MRI image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# 6. DISPLAY IMAGE AND RUN CLASSIFICATION
# --------------------------------------------------

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Uploaded MRI")

        st.image(
            image,
            caption="Original uploaded image",
            width=400
        )

        if st.button(
            "Classify MRI",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "Loading model and analysing image..."
            ):

                model = load_model()

                original, scores, heatmap, overlay = (
                    predict_mri(image, model)
                )

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
                    predicted_class.replace("_", " ").title()
                )

            with col2:
                st.metric(
                    "Model confidence",
                    f"{confidence:.2f}%"
                )

            st.caption(
                "Confidence is the model's output score, "
                "not the probability that a clinical diagnosis "
                "is correct."
            )

            st.subheader("Class Scores")

            score_percentages = {
                name: value * 100
                for name, value in scores.items()
            }

            st.bar_chart(score_percentages)

            st.write("Exact model scores:")

            for name, value in scores.items():
                st.write(f"**{name.title()}:** {value * 100:.2f}%")

            # ------------------------------------------
            # GRAD-CAM VISUALIZATIONS
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
                    heatmap,
                    caption="Grad-CAM Heatmap",
                    clamp=True,
                    use_container_width=True
                )

            with col3:
                st.image(
                    overlay,
                    caption="Grad-CAM Overlay",
                    use_container_width=True
                )

            st.info(
                "Grad-CAM highlights image regions that influence "
                "the model's selected class. Red and yellow indicate "
                "stronger relative activation in this heatmap; they "
                "do not prove the location of a tumour. The heatmap "
                "is not a tumour segmentation mask."
            )

            # ------------------------------------------
            # DOWNLOAD RESULTS
            # ------------------------------------------

            st.subheader("Save Visualization")

            overlay_image = Image.fromarray(overlay)

            import io

            buffer = io.BytesIO()
            overlay_image.save(buffer, format="PNG")

            st.download_button(
                label="Download Grad-CAM Overlay",
                data=buffer.getvalue(),
                file_name="gradcam_overlay.png",
                mime="image/png"
            )

    except Exception as error:
        st.error(
            "Something went wrong while processing the MRI."
        )

        st.exception(error)
