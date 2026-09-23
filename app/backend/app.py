import os
import sys
from pathlib import Path

# Add project root to sys.path to allow importing from src
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import re
import torch
import torch.nn as nn
from torchvision import models, transforms
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import pytesseract
import numpy as np

from src.models.multimodal import MultimodalModelV2
from src.explainability.visual_evidence import GradCAM, overlay_heatmap, image_to_base64
from src.explainability.text_evidence import extract_text_evidence
from src.explainability.explanation_rules import generate_explanation

app = Flask(__name__)
CORS(app)

# ==========================================
# CONFIG & PATHS
# ==========================================
CLASS_NAMES = [
    "Hard to Close",
    "Default Choice",
    "False Hierarchy",
    "Nagging",
    "Privacy Zuckering"
]
NUM_CLASSES = len(CLASS_NAMES)
THRESHOLD = 0.35

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

TFIDF_PATH = PROJECT_ROOT / "models" / "tfidf_vectorizer.joblib"
IMAGE_MODEL_PATH = PROJECT_ROOT / "models" / "image" / "best_resnet50_image_only.pth"
MULTIMODAL_MODEL_PATH = PROJECT_ROOT / "models" / "multimodal" / "best_multimodal_v2.pth"

# ==========================================
# LOAD MODELS
# ==========================================
print("Loading TF-IDF Vectorizer...")
vectorizer = joblib.load(TFIDF_PATH)

print("Loading Image Encoder...")
image_model = models.resnet50(weights=None)
image_model.fc = nn.Linear(image_model.fc.in_features, NUM_CLASSES)
image_checkpoint = torch.load(IMAGE_MODEL_PATH, map_location=DEVICE, weights_only=False)
image_model.load_state_dict(image_checkpoint["model_state_dict"])

# Remove final classification layer
image_encoder = nn.Sequential(*list(image_model.children())[:-1])

print("Loading Multimodal Model...")
model = MultimodalModelV2(image_encoder)
multimodal_checkpoint = torch.load(MULTIMODAL_MODEL_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(multimodal_checkpoint["model_state_dict"])
model = model.to(DEVICE)
model.eval()

# Setup GradCAM on the last convolutional layer of ResNet50
target_layer = model.image_encoder[7][-1].conv3  # layer4[-1].conv3
grad_cam = GradCAM(model, target_layer)

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# UTILS
# ==========================================
def clean_ocr_text(text):
    text = str(text)
    text = text.replace("\\n", " ").replace("\\t", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text.strip()

# ==========================================
# ENDPOINTS
# ==========================================
@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
        
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
        
    try:
        # Load and convert image
        original_image = Image.open(file).convert("RGB")
        
        # 1. OCR Extraction & Cleaning
        raw_text = pytesseract.image_to_string(original_image, config="--psm 11")
        clean_text = clean_ocr_text(raw_text)
        
        # 2. Text Features
        text_features_np = vectorizer.transform([clean_text]).toarray()
        text_features = torch.tensor(text_features_np, dtype=torch.float32).to(DEVICE)
        
        # 3. Image Features
        image_tensor = image_transform(original_image).unsqueeze(0).to(DEVICE)
        
        # 4. Inference
        with torch.no_grad():
            outputs = model(image_tensor, text_features)
            probabilities = torch.sigmoid(outputs)[0].cpu().numpy()
            
        # 5. Extract Predictions
        predictions = []
        for i, class_name in enumerate(CLASS_NAMES):
            confidence = float(probabilities[i])
            if confidence >= THRESHOLD:
                predictions.append({
                    "class_index": i,
                    "label": class_name,
                    "confidence": confidence
                })
                
        # Sort predictions by confidence
        predictions = sorted(predictions, key=lambda x: x["confidence"], reverse=True)
        
        if not predictions:
            # If no predictions above threshold, take the highest one
            best_idx = int(np.argmax(probabilities))
            predictions.append({
                "class_index": best_idx,
                "label": CLASS_NAMES[best_idx],
                "confidence": float(probabilities[best_idx])
            })
            
        # 6. Explainability for all detected patterns
        patterns_data = {}
        for p in predictions:
            idx = p["class_index"]
            lbl = p["label"]
            conf = p["confidence"]
            
            # --- Visual Evidence (Grad-CAM) ---
            heatmap = grad_cam.generate_heatmap(image_tensor, text_features, target_class=idx)
            overlayed_img = overlay_heatmap(original_image, heatmap)
            vis_b64 = f"data:image/jpeg;base64,{image_to_base64(overlayed_img)}"
            
            # --- Textual Evidence ---
            txt_ev = extract_text_evidence(model, image_tensor, text_features, vectorizer, target_class=idx)
            
            # --- Explanation ---
            exp = generate_explanation(lbl, conf, txt_ev)
            
            patterns_data[lbl] = {
                "label": lbl,
                "confidence": conf,
                "class_index": idx,
                "visual_evidence": vis_b64,
                "text_evidence": txt_ev,
                "explanation": exp
            }
            
        top_prediction = predictions[0]
        top_label = top_prediction["label"]
        
        response = {
            "predictions": [{"label": p["label"], "confidence": p["confidence"], "class_index": p["class_index"]} for p in predictions],
            "top_prediction": top_label,
            "patterns": patterns_data,
            # Backwards compatibility fields for single-pattern consumers:
            "visual_evidence": patterns_data[top_label]["visual_evidence"],
            "text_evidence": patterns_data[top_label]["text_evidence"],
            "explanation": patterns_data[top_label]["explanation"]
        }
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
