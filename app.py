
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import gdown

from PIL import Image
from pathlib import Path


# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="Brain Tumor Classification",
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

# Google Drive ID of your trained model
MODEL_ID = "1sO3yNwEIu9oCfP3W_wlC6r07fhv0hEYI"

# Save the model in Streamlit Cloud's writable home directory
MODEL_PATH = (
    Path.home()
    / ".brain_tumor_model"
    / "best_brain_tumor_finetuned.keras"
)


# --------------------------------------------------
# LOAD MODEL FROM GOOGLE DRIVE
# --------------------------------------------------

@st.cache_resource
def load_model():

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size == 0:

        with st.spinner("Downloading trained model from Google Drive..."):

            if MODEL_PATH.exists():
                MODEL_PATH.unlink()

            downloaded = gdown.download(
                id=MODEL_ID,
                output=str(MODEL_PATH),
                quiet=False,
                fuzzy=True
            )

        if (
            downloaded is None
            or not MODEL_PATH.exists()
            or MODEL_PATH.stat().st_size == 0
        ):
            raise FileNotFoundError(
                "Model download failed. Check the Google Drive file ID "
                "and make sure the file is shared as "
                "'Anyone with the link – Viewer'."
            )

    model = tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False
    )

    # Check that the expected layers exist
    required_layers = [
        "data_augmentation",
        "resnet50",
        "gap",
        "dropout",
        "dense_features",
        "class_predictions"
    ]

    missing_layers = [
        name for name in required_layers
        if not any(layer.name == name for layer in model.layers)
    ]

    if missing_layers:
        raise ValueError(
            "The loaded model does not have the expected layers: "
            + ", ".join(missing_layers)
        )

    return model


# --------------------------------------------------
# GRAD-CAM
# --------------------------------------------------

def make_gradcam(image_array, model, class_index):

    # Get the same preprocessing/augmentation layer used in training
    augmentation_layer = model.get_layer("data_augmentation")

    preprocessing_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=augmentation_layer.output
    )

    # Get the ResNet50 backbone
    resnet = model.get_layer("resnet50")

    # Last convolutional feature map
    conv_layer = resnet.get_layer("conv5_block3_out")

    feature_model = tf.keras.Model(
        inputs=resnet.input,
        outputs=conv_layer.output
    )

    # Use RGB pixel values in the same range as model prediction
    batch = image_array.astype(np.float32)[None, ...]

    # Apply the model's own preprocessing layer
    processed = preprocessing_model(batch, training=False)

    with tf.GradientTape() as tape:

        # Extract convolutional feature maps
        conv_outputs = feature_model(
            processed,
            training=False
        )

        # Reuse the original trained classification head
        x = model.get_layer("gap")(conv_outputs)
        x = model.get_layer("dropout")(x, training=False)
        x = model.get_layer("dense_features")(x)
        predictions = model.get_layer("class_predictions")(x)

        # Score for the class being explained
        class_score = predictions[:, class_index]

    # Calculate gradients of the selected class score
    gradients = tape.gradient(class_score, conv_outputs)

    if gradients is None:
        raise ValueError(
            "Grad-CAM gradients could not be calculated."
        )

    # Average gradients across the spatial dimensions
    channel_weights = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    # Weight feature maps using the gradients
    heatmap = tf.reduce_sum(
        conv_outputs
        * channel_weights[:, None, None, :],
        axis=-1
    )[0]

    # Keep positive activations
    heatmap = tf.maximum(heatmap, 0)

    heatmap_max = float(tf.reduce_max(heatmap).numpy())

    if not np.isfinite(heatmap_max) or heatmap_max <= 1e-8:
        raise ValueError(
            "Grad-CAM produced no meaningful positive activation "
            "for this image."
        )

    heatmap = heatmap / heatmap_max
    heatmap = heatmap.numpy()

    # Resize heatmap to match the displayed MRI image
    heatmap = cv2.resize(
        heatmap,
        (image_array.shape[1], image_array.shape[0]),
        interpolation=cv2.INTER_LINEAR
    )

    heatmap = np.clip(heatmap, 0, 1)

    # Convert heatmap to RGB for display
    colored_heatmap = cv2.applyColorMap(
        np.uint8(255 * heatmap),
        cv2.COLORMAP_JET
    )

    colored_heatmap = cv2.cvtColor(
        colored_heatmap,
        cv2.COLOR_BGR2RGB
    )

    # Overlay heatmap on the original MRI
    overlay = cv2.addWeighted(
        image_array,
        0.60,
        colored_heatmap,
        0.40,
        0
    )

    return heatmap, colored_heatmap, overlay


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

def predict_mri(uploaded_image, model):

    # Convert image to RGB and resize
    image = uploaded_image.convert("RGB").resize(IMG_SIZE)

    original = np.array(image, dtype=np.uint8)

    # The complete trained model handles its own preprocessing
    batch = original.astype(np.float32)[None, ...]

    predictions = model.predict(
        batch,
        verbose=0
    )

    probabilities = np.asarray(predictions[0], dtype=np.float32)

    if len(probabilities) != len(CLASS_NAMES):
        raise ValueError(
            f"Expected {len(CLASS_NAMES)} class scores, "
            f"but received {len(probabilities)}."
        )

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("The model returned invalid prediction scores.")

    # If the final layer is softmax, these should be probabilities
    if (
        np.any(probabilities < 0)
        or np.any(probabilities > 1)
        or not np.isclose(np.sum(probabilities), 1.0, atol=1e-3)
    ):
        raise ValueError(
            "The model output does not look like softmax probabilities. "
            "Check the final layer and the original training setup."
        )

    predicted_index = int(np.argmax(probabilities))

    # Explain the predicted class
    heatmap, colored_heatmap, overlay = make_gradcam(
        original,
        model,
        predicted_index
    )

    scores = {
        CLASS_NAMES[i]: float(probabilities[i])
        for i in range(len(CLASS_NAMES))
    }

    return (
        scores,
        predicted_index,
        heatmap,
        colored_heatmap,
        overlay,
        original
    )


# --------------------------------------------------
# STREAMLIT USER INTERFACE
# --------------------------------------------------

st.title("🧠 Brain Tumor Classification")
st.write(
    "Upload a brain MRI image to obtain a model prediction "
    "and a Grad-CAM visualization."
)

st.info(
    "This is an educational AI project, not a medical diagnostic tool."
)

uploaded_file = st.file_uploader(
    "Upload a brain MRI image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    try:
        uploaded_image = Image.open(uploaded_file)

        st.subheader("Uploaded MRI")
        st.image(
            uploaded_image,
            caption="Input image",
            width=350
        )

        if st.button("Classify MRI", type="primary"):

            with st.spinner("Loading model and analysing image..."):

                model = load_model()

                (
                    scores,
                    predicted_index,
                    heatmap,
                    colored_heatmap,
                    overlay,
                    original
                ) = predict_mri(uploaded_image, model)

            predicted_class = CLASS_NAMES[predicted_index]
            confidence = scores[predicted_class] * 100

            st.divider()
            st.subheader("Prediction")

            st.success(
                f"Predicted class: {predicted_class.upper()}"
            )

            st.metric(
                label="Model confidence",
                value=f"{confidence:.2f}%"
            )

            st.caption(
                "Confidence is the model's output score, "
                "not a measure of medical certainty."
            )

            st.subheader("Class scores")

            for class_name, score in scores.items():

                st.write(
                    f"**{class_name.upper()}** — {score * 100:.2f}%"
                )

                st.progress(
                    float(np.clip(score, 0, 1))
                )

            st.divider()
            st.subheader("Grad-CAM Visualization")

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
                    caption="Grad-CAM heatmap",
                    use_container_width=True
                )

            with col3:
                st.image(
                    overlay,
                    caption="Heatmap overlay",
                    use_container_width=True
                )

            st.caption(
                "Grad-CAM highlights image regions that influence "
                "the selected class score. It does not prove that "
                "a tumour is present or accurately outline a tumour."
            )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
        st.info(
            "If the error concerns downloading the model, verify "
            "that the Google Drive file ID is correct and the file "
            "is accessible to anyone with the link."
        )

else:
    st.write(
        "Supported formats: JPG, JPEG and PNG."
    )
