"""
DeceptiUI - Text Baseline Evaluation

Evaluates the trained TF-IDF + One-vs-Rest Logistic Regression
text-only model on the test split.

No PyTorch is required.
"""

import csv
import json
from pathlib import Path

import joblib
import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "text"
    / "test.csv"
)

VECTORIZER_PATH = (
    PROJECT_ROOT
    / "models"
    / "tfidf_vectorizer.joblib"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "text_baseline_model.joblib"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "src"
    / "evaluation"
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_PATH = (
    RESULTS_DIR
    / "text_baseline_results.json"
)


# ============================================================
# CLASSES
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

THRESHOLD = 0.5


# ============================================================
# LOAD CSV WITHOUT PANDAS
# ============================================================

def load_test_data():

    texts = []
    labels = []

    with open(
        TEST_PATH,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        required_columns = ["clean_text"] + LABEL_COLUMNS

        for column in required_columns:

            if column not in reader.fieldnames:
                raise ValueError(
                    f"Column '{column}' not found in test.csv"
                )

        for row in reader:

            text = row["clean_text"]

            if text is None:
                text = ""

            texts.append(text)

            label_row = []

            for column in LABEL_COLUMNS:

                value = row[column].strip()

                label_row.append(
                    int(float(value))
                )

            labels.append(label_row)

    return texts, np.array(labels, dtype=int)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("DeceptiUI - Text Baseline Evaluation")
    print("=" * 60)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Test dataset not found:\n{TEST_PATH}"
        )

    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            f"TF-IDF vectorizer not found:\n{VECTORIZER_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Text model not found:\n{MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    print("\nLoading test dataset...")

    texts, y_true = load_test_data()

    print(f"Test samples: {len(texts)}")

    # --------------------------------------------------------
    # Load TF-IDF vectorizer
    # --------------------------------------------------------

    print("\nLoading TF-IDF vectorizer...")

    vectorizer = joblib.load(VECTORIZER_PATH)

    # --------------------------------------------------------
    # Load trained classifier
    # --------------------------------------------------------

    print("Loading text model...")

    model = joblib.load(MODEL_PATH)

    # --------------------------------------------------------
    # Transform text
    # --------------------------------------------------------

    print("\nTransforming test text...")

    X_test = vectorizer.transform(texts)

    print(
        f"TF-IDF matrix shape: {X_test.shape}"
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    print("\nGenerating predictions...")

    probabilities = model.predict_proba(X_test)

    # sklearn OneVsRestClassifier normally returns
    # a NumPy array of shape:
    #
    # (samples, classes)

    probabilities = np.asarray(probabilities)

    if probabilities.ndim != 2:

        raise ValueError(
            f"Unexpected probability shape: "
            f"{probabilities.shape}"
        )

    y_pred = (
        probabilities >= THRESHOLD
    ).astype(int)

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Print overall results
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL TEXT-ONLY TEST RESULTS")
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

    # --------------------------------------------------------
    # Per-class results
    # --------------------------------------------------------

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        zero_division=0,
        output_dict=True,
    )

    print("\n" + "=" * 60)
    print("PER-CLASS TEST RESULTS")
    print("=" * 60)

    for class_name in CLASS_NAMES:

        metrics = report[class_name]

        print(
            f"{class_name:<22}"
            f"Precision={metrics['precision']:.4f}  "
            f"Recall={metrics['recall']:.4f}  "
            f"F1={metrics['f1-score']:.4f}"
        )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results = {

        "model": "TF-IDF + One-vs-Rest Logistic Regression",

        "test_samples": len(texts),

        "threshold": THRESHOLD,

        "overall": {
            "micro_precision": float(
                micro_precision
            ),
            "micro_recall": float(
                micro_recall
            ),
            "micro_f1": float(
                micro_f1
            ),
            "macro_f1": float(
                macro_f1
            ),
            "weighted_f1": float(
                weighted_f1
            ),
        },

        "per_class": {},
    }

    for class_name in CLASS_NAMES:

        metrics = report[class_name]

        results["per_class"][class_name] = {

            "precision": float(
                metrics["precision"]
            ),

            "recall": float(
                metrics["recall"]
            ),

            "f1": float(
                metrics["f1-score"]
            ),

            "support": int(
                metrics["support"]
            ),
        }

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    print("\n" + "=" * 60)
    print("RESULTS SAVED")
    print("=" * 60)

    print(RESULTS_PATH)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()