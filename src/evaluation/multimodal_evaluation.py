import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score
)
from torchvision import models, transforms


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = PROJECT_ROOT / "dataset" / "processed" / "images"
TEST_CSV = PROJECT_ROOT / "dataset" / "processed" / "text" / "test.csv"

TFIDF_PATH = PROJECT_ROOT / "models" / "tfidf_vectorizer.joblib"

IMAGE_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "image"
    / "best_resnet50_image_only.pth"
)

MULTIMODAL_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "multimodal"
    / "best_multimodal_v2.pth"
)

RESULTS_DIR = PROJECT_ROOT / "src" / "evaluation" / "results"

RESULTS_PATH = (
    RESULTS_DIR / "multimodal_v2_results.json"
)


# ============================================================
# SETTINGS
# ============================================================

CLASS_NAMES = [
    "hard_to_close",
    "default_choice",
    "false_hierarchy",
    "nagging",
    "privacy_zuckering"
]

NUM_CLASSES = len(CLASS_NAMES)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")


# ============================================================
# MULTIMODAL MODEL V2
# ============================================================

class MultimodalModelV2(nn.Module):

    def __init__(self, image_encoder):
        super().__init__()

        self.image_encoder = image_encoder

        # Image projection: 2048 -> 512
        self.image_projection = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Text projection: 7102 -> 256
        self.text_projection = nn.Sequential(
            nn.Linear(7102, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Fusion: 512 + 256 -> 256 -> 5
        self.fusion = nn.Sequential(
            nn.Linear(512 + 256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 5)
        )

    def forward(self, images, text_features):

        image_features = self.image_encoder(images)

        image_features = torch.flatten(
            image_features,
            1
        )

        image_features = self.image_projection(
            image_features
        )

        text_features = self.text_projection(
            text_features
        )

        combined = torch.cat(
            [image_features, text_features],
            dim=1
        )

        outputs = self.fusion(combined)

        return outputs


# ============================================================
# LOAD TEST DATA
# ============================================================

test_df = pd.read_csv(TEST_CSV)

print(f"Test samples: {len(test_df)}")

print("\nColumns:")
print(test_df.columns.tolist())


# ============================================================
# LOAD TF-IDF VECTORIZER
# ============================================================

vectorizer = joblib.load(TFIDF_PATH)

print(
    f"\nTF-IDF features: "
    f"{len(vectorizer.get_feature_names_out())}"
)


# ============================================================
# IMAGE TRANSFORM
# ============================================================

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD IMAGE MODEL
# ============================================================

image_model = models.resnet50(weights=None)

image_model.fc = nn.Linear(
    image_model.fc.in_features,
    NUM_CLASSES
)

image_checkpoint = torch.load(
    IMAGE_MODEL_PATH,
    map_location=device,
    weights_only=False
)

image_model.load_state_dict(
    image_checkpoint["model_state_dict"]
)

# Remove final classification layer
image_encoder = nn.Sequential(
    *list(image_model.children())[:-1]
)


# ============================================================
# LOAD MULTIMODAL MODEL
# ============================================================

model = MultimodalModelV2(
    image_encoder
)

multimodal_checkpoint = torch.load(
    MULTIMODAL_MODEL_PATH,
    map_location=device,
    weights_only=False
)

model.load_state_dict(
    multimodal_checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()

print("\nMultimodal V2 model loaded successfully.")


# ============================================================
# EVALUATION
# ============================================================

true_labels = []
predictions = []


with torch.no_grad():

    for _, row in test_df.iterrows():

        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        image_path = IMAGE_DIR / row["image"]

        image = Image.open(
            image_path
        ).convert("RGB")

        image = image_transform(image)

        image = image.unsqueeze(0).to(device)

        # ----------------------------------------------------
        # TEXT
        # ----------------------------------------------------

        text = str(row["clean_text"])

        text_features = vectorizer.transform(
            [text]
        ).toarray()

        text_features = torch.tensor(
            text_features,
            dtype=torch.float32
        ).to(device)

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        outputs = model(
            image,
            text_features
        )

        probabilities = torch.sigmoid(
            outputs
        )

        prediction = (
            probabilities >= 0.5
        ).int().cpu().numpy()[0]

        # ----------------------------------------------------
        # TRUE LABEL
        # ----------------------------------------------------

        true_label = row[
            CLASS_NAMES
        ].values.astype(int)

        true_labels.append(true_label)
        predictions.append(prediction)


# ============================================================
# NUMPY ARRAYS
# ============================================================

true_labels = np.array(true_labels)

predictions = np.array(predictions)


# ============================================================
# METRICS
# ============================================================

micro_precision = precision_score(
    true_labels,
    predictions,
    average="micro",
    zero_division=0
)

micro_recall = recall_score(
    true_labels,
    predictions,
    average="micro",
    zero_division=0
)

micro_f1 = f1_score(
    true_labels,
    predictions,
    average="micro",
    zero_division=0
)

macro_f1 = f1_score(
    true_labels,
    predictions,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    true_labels,
    predictions,
    average="weighted",
    zero_division=0
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("DECEPTIUI - MULTIMODAL V2 TEST RESULTS")
print("=" * 60)

print(
    f"Micro Precision : {micro_precision:.4f}"
)

print(
    f"Micro Recall    : {micro_recall:.4f}"
)

print(
    f"Micro F1        : {micro_f1:.4f}"
)

print(
    f"Macro F1        : {macro_f1:.4f}"
)

print(
    f"Weighted F1     : {weighted_f1:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        true_labels,
        predictions,
        target_names=CLASS_NAMES,
        zero_division=0
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

results = {
    "model": "multimodal_v2",
    "test_samples": len(test_df),
    "micro_precision": float(micro_precision),
    "micro_recall": float(micro_recall),
    "micro_f1": float(micro_f1),
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1)
}

with open(
    RESULTS_PATH,
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )

print(
    f"\nResults saved to:\n{RESULTS_PATH}"
)