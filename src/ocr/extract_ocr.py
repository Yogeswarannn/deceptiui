import os
import json
import zipfile
import re

import pandas as pd
import pytesseract
from PIL import Image
from io import BytesIO


# -----------------------------
# PATHS
# -----------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

ZIP_PATH = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "original",
    "ContextRico-DP.zip"
)

GROUNDTRUTH_PATH = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "original",
    "groundtrue.json"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "processed"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# OUR 5 DARK-PATTERN CLASSES
# -----------------------------

TARGET_CLASSES = [
    "Hard to Close",
    "Default Choice",
    "False Hierarchy",
    "Nagging",
    "Privacy Zuckering"
]


# -----------------------------
# LOAD GROUND TRUTH
# -----------------------------

print("Loading ground truth...")

with open(GROUNDTRUTH_PATH, "r", encoding="utf-8") as f:
    groundtruth = json.load(f)

print(f"Ground-truth entries: {len(groundtruth)}")


# -----------------------------
# SELECT OUR 5 CLASSES
# -----------------------------

selected_images = {}

for image_name, annotation in groundtruth.items():

    labels = annotation.get("labels", [])

    selected_labels = [
        label
        for label in labels
        if label in TARGET_CLASSES
    ]

    if selected_labels:

        selected_images[image_name] = {
            "labels": selected_labels
        }


print(f"Selected images: {len(selected_images)}")


# -----------------------------
# OPEN ZIP
# -----------------------------

print("\nOpening ContextRico-DP.zip...")

zip_file = zipfile.ZipFile(ZIP_PATH)

# Create mapping:
# filename -> path inside ZIP

image_paths = {}

for name in zip_file.namelist():

    if name.lower().endswith(
        (".jpg", ".jpeg", ".png", ".webp")
    ):

        filename = os.path.basename(name)

        image_paths[filename] = name


print(f"Images found in ZIP: {len(image_paths)}")


# -----------------------------
# OCR
# -----------------------------

results = []

total = len(selected_images)

print("\nStarting OCR...\n")


for count, (image_name, info) in enumerate(
    selected_images.items(),
    start=1
):

    print(f"[{count}/{total}] {image_name}")

    if image_name not in image_paths:

        print("   WARNING: image not found")

        continue

    zip_path = image_paths[image_name]

    image_bytes = zip_file.read(zip_path)

    image = Image.open(
        BytesIO(image_bytes)
    ).convert("RGB")


    # -------------------------
    # OCR
    # -------------------------

    text = pytesseract.image_to_string(
        image,
        config="--psm 11"
    )


    # Clean OCR text

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    text = text.strip()


    # -------------------------
    # MULTI-HOT LABELS
    # -------------------------

    row = {
        "image": image_name,
        "ocr_text": text
    }

    for class_name in TARGET_CLASSES:

        column_name = (
            class_name
            .lower()
            .replace(" ", "_")
        )

        row[column_name] = int(
            class_name in info["labels"]
        )


    results.append(row)


# -----------------------------
# SAVE DATASET
# -----------------------------

df = pd.DataFrame(results)


output_file = os.path.join(
    OUTPUT_DIR,
    "labels_with_ocr.csv"
)

df.to_csv(
    output_file,
    index=False,
    encoding="utf-8"
)


# -----------------------------
# SAVE CLASS MAPPING
# -----------------------------

class_mapping = {
    class_name: index
    for index, class_name
    in enumerate(TARGET_CLASSES)
}


mapping_file = os.path.join(
    OUTPUT_DIR,
    "class_mapping.json"
)

with open(
    mapping_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        class_mapping,
        f,
        indent=4
    )


# -----------------------------
# SUMMARY
# -----------------------------

print("\n" + "=" * 50)

print("OCR COMPLETE")

print("=" * 50)

print(f"Images processed : {len(df)}")

print(
    f"Images with text : "
    f"{(df['ocr_text'].str.len() > 0).sum()}"
)

print(
    f"Images without text : "
    f"{(df['ocr_text'].str.len() == 0).sum()}"
)

print(
    f"\nSaved to:\n{output_file}"
)