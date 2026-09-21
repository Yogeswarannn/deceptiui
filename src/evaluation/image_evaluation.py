"""
DeceptiUI - Image Model Evaluation

Evaluates the trained ResNet-50 image-only model
on the test split.

Outputs:
- Micro Precision
- Micro Recall
- Micro F1
- Macro F1
- Weighted F1
- Per-class Precision / Recall / F1
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

from src.models.image_baseline import load_model


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LABELS_PATH = PROJECT_ROOT / "dataset" / "processed" / "labels.csv"
IMAGE_DIR = PROJECT_ROOT / "dataset" / "processed" / "images"

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "image"
    / "best_resnet50_image_only.pth"
)

RESULTS_DIR = PROJECT_ROOT / "src" / "evaluation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = RESULTS_DIR / "image_baseline_results.json"


# ============================================================
# CONFIG
# ============================================================

CLASS_NAMES = [
    "Hard to Close",
    "Default Choice",
    "False Hierarchy",
    "Nagging",
    "Privacy Zuckering",
]

LABEL_COLUMNS = [
    "hard_to_close",
    "default_choice",
    "false_hierarchy",
    "nagging",
    "privacy_zuckering",
]

BATCH_SIZE = 32
THRESHOLD = 0.5


# ============================================================
# DATASET
# ============================================================

class ImageDataset(Dataset):

    def __init__(self, dataframe, image_dir, transform=None):

        self.df = dataframe.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        image_path = self.image_dir / row["image"]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        labels = torch.tensor(
            row[LABEL_COLUMNS].values.astype(np.float32)
        )

        return image, labels


# ============================================================
# TRANSFORMS
# ============================================================

def get_eval_transform():

    return transforms.Compose([
        transforms.Resize((224, 224)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print("=" * 60)
    print("DeceptiUI - Image Model Evaluation")
    print("=" * 60)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            f"Labels file not found:\n{LABELS_PATH}"
        )

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Image directory not found:\n{IMAGE_DIR}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found:\n{MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Load labels
    # --------------------------------------------------------

    df = pd.read_csv(LABELS_PATH)

    # Use only test split
    test_df = df[df["split"] == "test"].copy()

    print(f"\nTest images: {len(test_df)}")

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    transform = get_eval_transform()

    test_dataset = ImageDataset(
        dataframe=test_df,
        image_dir=IMAGE_DIR,
        transform=transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading model...")

    model, checkpoint = load_model(
        MODEL_PATH,
        device=device,
    )

    model.eval()

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    all_labels = []
    all_probabilities = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(images)

            probabilities = torch.sigmoid(outputs)

            all_labels.append(labels.numpy())
            all_probabilities.append(
                probabilities.cpu().numpy()
            )

    y_true = np.concatenate(all_labels, axis=0)
    y_prob = np.concatenate(all_probabilities, axis=0)

    # Convert probabilities to binary predictions
    y_pred = (y_prob >= THRESHOLD).astype(int)

    # ========================================================
    # METRICS
    # ========================================================

    micro_precision = precision_score(
        y_true,
        y_pred,
        average="micro",
        zero_division=0,
    )

    micro_recall = recall_score(
        y_true,
        y_pred,
        average="micro",
        zero_division=0,
    )

    micro_f1 = f1_score(
        y_true,
        y_pred,
        average="micro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    # ========================================================
    # PRINT OVERALL RESULTS
    # ========================================================

    print("\n" + "=" * 60)
    print("OVERALL TEST RESULTS")
    print("=" * 60)

    print(f"Micro Precision : {micro_precision:.4f}")
    print(f"Micro Recall    : {micro_recall:.4f}")
    print(f"Micro F1        : {micro_f1:.4f}")
    print(f"Macro F1        : {macro_f1:.4f}")
    print(f"Weighted F1     : {weighted_f1:.4f}")

    # ========================================================
    # PER-CLASS RESULTS
    # ========================================================

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        zero_division=0,
        output_dict=True,
    )

    print("\n" + "=" * 60)
    print("PER-CLASS RESULTS")
    print("=" * 60)

    for class_name in CLASS_NAMES:

        metrics = report[class_name]

        print(
            f"{class_name:<22} "
            f"Precision={metrics['precision']:.4f}  "
            f"Recall={metrics['recall']:.4f}  "
            f"F1={metrics['f1-score']:.4f}"
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results = {
        "model": "ResNet-50 Image-Only",
        "checkpoint": str(MODEL_PATH),
        "test_samples": int(len(test_df)),
        "threshold": THRESHOLD,

        "overall": {
            "micro_precision": float(micro_precision),
            "micro_recall": float(micro_recall),
            "micro_f1": float(micro_f1),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
        },

        "per_class": {},

    }

    for class_name in CLASS_NAMES:

        metrics = report[class_name]

        results["per_class"][class_name] = {
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1": float(metrics["f1-score"]),
            "support": int(metrics["support"]),
        }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:

        json.dump(
            results,
            f,
            indent=4,
        )

    print("\n" + "=" * 60)
    print(f"Results saved to:")
    print(RESULTS_PATH)
    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()