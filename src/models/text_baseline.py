import os
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, f1_score


# ==========================================
# PATHS
# ==========================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "processed",
    "text"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

os.makedirs(MODEL_DIR, exist_ok=True)


# ==========================================
# LOAD DATA
# ==========================================

print("Loading datasets...")

train_df = pd.read_csv(
    os.path.join(DATA_DIR, "train.csv")
)

val_df = pd.read_csv(
    os.path.join(DATA_DIR, "validation.csv")
)

test_df = pd.read_csv(
    os.path.join(DATA_DIR, "test.csv")
)

print(f"Train:      {len(train_df)}")
print(f"Validation: {len(val_df)}")
print(f"Test:       {len(test_df)}")


# ==========================================
# TEXT
# ==========================================

X_train_text = train_df["clean_text"].fillna("")
X_val_text = val_df["clean_text"].fillna("")
X_test_text = test_df["clean_text"].fillna("")


# ==========================================
# LABELS
# ==========================================

LABELS = [
    "hard_to_close",
    "default_choice",
    "false_hierarchy",
    "nagging",
    "privacy_zuckering"
]

Y_train = train_df[LABELS]
Y_val = val_df[LABELS]
Y_test = test_df[LABELS]


# ==========================================
# TF-IDF
# ==========================================

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    max_features=20000,
    sublinear_tf=True
)

X_train = vectorizer.fit_transform(X_train_text)

X_val = vectorizer.transform(X_val_text)

X_test = vectorizer.transform(X_test_text)

print("Training feature shape:", X_train.shape)
print("Validation feature shape:", X_val.shape)
print("Test feature shape:", X_test.shape)


# ==========================================
# MODEL
# ==========================================

print("\nTraining Logistic Regression...")

model = OneVsRestClassifier(
    LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    )
)

model.fit(
    X_train,
    Y_train
)


# ==========================================
# VALIDATION
# ==========================================

print("\nEvaluating on validation set...")

val_predictions = model.predict(X_val)


print("\nValidation Classification Report:")
print(
    classification_report(
        Y_val,
        val_predictions,
        target_names=LABELS,
        zero_division=0
    )
)


val_micro_f1 = f1_score(
    Y_val,
    val_predictions,
    average="micro",
    zero_division=0
)

val_macro_f1 = f1_score(
    Y_val,
    val_predictions,
    average="macro",
    zero_division=0
)

print(
    f"Validation Micro F1: {val_micro_f1:.4f}"
)

print(
    f"Validation Macro F1: {val_macro_f1:.4f}"
)


# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(
    model,
    os.path.join(
        MODEL_DIR,
        "text_baseline_model.joblib"
    )
)

joblib.dump(
    vectorizer,
    os.path.join(
        MODEL_DIR,
        "tfidf_vectorizer.joblib"
    )
)


print("\nModel saved.")

print(
    os.path.join(
        MODEL_DIR,
        "text_baseline_model.joblib"
    )
)

print(
    os.path.join(
        MODEL_DIR,
        "tfidf_vectorizer.joblib"
    )
)