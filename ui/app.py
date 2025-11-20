import os
import tempfile
import sys
# Add project root to PYTHONPATH
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import streamlit as st
import torch

from src.inference import InferenceEngine
from src.gradcam import overlay_heatmap
from src.utils import CLASS_NAMES

# Config

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

MODEL_PATH = "ml/models/dental_classifier.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def get_engine():
    return InferenceEngine(weights_path=MODEL_PATH, device=DEVICE)

engine = get_engine()

# --------- UI Layout ---------
st.set_page_config(page_title="Dental AI Classifier", layout="wide")
st.title("Dental Imaging Classifier + Grad-CAM")
st.caption(f"Device: **{DEVICE}** | Model: EfficientNet-B0 (4 classes)")

uploaded_file = st.file_uploader(
    "Upload a dental image (X-ray or intraoral photo)", 
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    # Save to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Run inference  and Grad-CAM
    result = engine.predict_with_gradcam(tmp_path)
    class_id = result["class_id"]
    class_name = result["class_name"]
    confidence = result["confidence"]
    probs = result["probs"]
    cam = result["cam"]
    image_tensor = result["image_tensor"]  # on CPU already

    # Build Grad-CAM visuals
    orig_img, heatmap, overlay = overlay_heatmap(image_tensor, cam)

    # Layout: 3 columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Original")
        st.image(orig_img, use_container_width=True)

    with col2:
        st.subheader("Grad-CAM Heatmap")
        st.image(heatmap, use_container_width=True)

    with col3:
        st.subheader("Overlay")
        st.image(overlay, use_container_width=True)

    # Prediction info
    st.markdown("---")
    st.subheader("Prediction")

    st.markdown(
        f"**Predicted class:** `{class_name}`  \n"
        f"**Confidence:** `{confidence:.3f}`"
    )

    # Show full class probability table
    st.subheader("Class probabilities")
    f"**Gum Inflammation:** {probs[0]:.3f}"
    f"**Healthy:** {probs[1]:.3f}"
    f"**Plaque:** {probs[2]:.3f}"
    f"**Unknown:** {probs[3]:.3f}"

