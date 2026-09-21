import os
import re
import pandas as pd


# -----------------------------
# PATHS
# -----------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

INPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "processed",
    "labels_with_ocr.csv"
)

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "processed",
    "cleaned_ocr.csv"
)


# -----------------------------
# LOAD DATA
# -----------------------------

df = pd.read_csv(INPUT_FILE)

print("Original rows:", len(df))


# -----------------------------
# CLEAN OCR TEXT
# -----------------------------

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    # Convert escaped newlines/tabs into spaces
    text = text.replace("\\n", " ")
    text = text.replace("\\t", " ")

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove isolated control characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


df["clean_text"] = df["ocr_text"].apply(clean_text)


# -----------------------------
# TEXT STATISTICS
# -----------------------------

df["text_length"] = df["clean_text"].str.len()

df["has_text"] = (
    df["clean_text"].str.strip() != ""
).astype(int)


# -----------------------------
# SAVE
# -----------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)


# -----------------------------
# SUMMARY
# -----------------------------

print("\n" + "=" * 50)
print("OCR CLEANING COMPLETE")
print("=" * 50)

print("Rows:", len(df))

print(
    "Images with text:",
    df["has_text"].sum()
)

print(
    "Images without text:",
    len(df) - df["has_text"].sum()
)

print(
    "Average text length:",
    round(df["text_length"].mean(), 2)
)

print(
    "Median text length:",
    round(df["text_length"].median(), 2)
)

print(
    "\nSaved to:"
)

print(OUTPUT_FILE)