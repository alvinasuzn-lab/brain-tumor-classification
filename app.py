from pathlib import Path

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input

MODEL_PATH = Path(__file__).parent / "best_brain_tumor_finetuned.keras"
DRIVE_FILE_ID = "1sO3yNwEIu9oCfP3W_wlC6r07fhv0hEYI"

@st.cache_resource
def load_model():
    import gdown

    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size < 100_000_000:
        result = gdown.download(
            id=DRIVE_FILE_ID,
            output=str(MODEL_PATH),
            quiet=False
        )

        if not result or not MODEL_PATH.exists():
            raise RuntimeError(
                "Model download failed. Check Google Drive sharing settings."
            )

    return tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False
    )
