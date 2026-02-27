import streamlit as st
import torch
import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from model import CLIP_GCN_LearnableAdj

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Cotton Disease Classification", layout="centered")
st.title("🌿 Cotton Disease Classification (CLIP + GCN)")

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Class Names (Must match training order)
# -----------------------------
CLASS_NAMES = [
    "Alternaria Leaf",
    "Bacterial Blight",
    "Fusarium Wilt",
    "Healthy Leaf",
    "Verticillium Wilt"
]

# -----------------------------
# Model Path
# -----------------------------
model_path = os.path.join(os.path.dirname(__file__), "best_clip_gcn_model.pth")

# -----------------------------
# Load CLIP (cached)
# -----------------------------
@st.cache_resource
def load_clip():
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()
    return clip_model, clip_processor

clip_model, clip_processor = load_clip()

# -----------------------------
# Load GCN Model (cached)
# -----------------------------
@st.cache_resource
def load_gcn():
    model = CLIP_GCN_LearnableAdj(num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

model = load_gcn()

# -----------------------------
# Image Upload
# -----------------------------
uploaded_file = st.file_uploader("Upload Cotton Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # -----------------------------
    # CLIP Feature Extraction
    # -----------------------------
    inputs = clip_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        features = clip_model.get_image_features(pixel_values=inputs["pixel_values"])

    # Ensure proper shape and dtype
    if features.dim() == 3:  # [batch, 1, 512]
        features = features.squeeze(1)
    features = features.float().to(device)  # float32 & device

    # -----------------------------
    # Prediction
    # -----------------------------
    with torch.no_grad():
        output = model(features)
        prediction = torch.argmax(output, dim=1).item()

    st.success(f"Predicted Class: {CLASS_NAMES[prediction]}")
