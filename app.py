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
# Class Names
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
# Load CLIP
# -----------------------------
@st.cache_resource
def load_clip():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.to(device)
    model.eval()
    return model, processor

clip_model, clip_processor = load_clip()

# -----------------------------
# Load GCN
# -----------------------------
@st.cache_resource
def load_gcn():
    model = CLIP_GCN_LearnableAdj(num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
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

    # CLIP preprocessing
    inputs = clip_processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Extract features safely
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)

    # -----------------------------
    # SAFELY handle tuple or non-tensor
    # -----------------------------
    if isinstance(features, tuple):
        features = features[0]

    # ensure tensor, float and correct device
    features = features.to(device=device, dtype=torch.float)

    # -----------------------------
    # Prediction
    # -----------------------------
    with torch.no_grad():
        output = model(features)
        prediction = torch.argmax(output, dim=1).item()

    st.success(f"Predicted Class: {CLASS_NAMES[prediction]}")
