import streamlit as st
import torch
from PIL import Image
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from model import CLIP_GCN_LearnableAdj

st.title("🌿 Cotton Disease Classification (CLIP + GCN)")

device = torch.device("cpu")

CLASS_NAMES = [
    "Alternaria Leaf",
    "Bacterial Blight",
    "Fusarium Wilt",
    "Healthy Leaf",
    "Verticillium Wilt"
]

# Load CLIP
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

# Load GCN model
model = CLIP_GCN_LearnableAdj(num_classes=len(CLASS_NAMES))
model.load_state_dict(torch.load("best_clip_gcn_model.pth", map_location=device))
model.eval()

uploaded_file = st.file_uploader("Upload Cotton Leaf Image", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with torch.no_grad():
        inputs = clip_processor(images=image, return_tensors="pt")
        features = clip_model.get_image_features(**inputs)
        features = features.float()

        output = model(features)
        prediction = torch.argmax(output, dim=1).item()

    st.success(f"Predicted Class: {CLASS_NAMES[prediction]}")
