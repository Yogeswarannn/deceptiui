import os
import pandas as pd


# --------------------------------
# PATHS
# --------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

CLEANED_FILE = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "processed",
    "cleaned_ocr.csv"
)

MANIFEST_FILE = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "processed",
    "labels.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "processed",
    "text"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------
# LOAD DATA
# --------------------------------

print("Loading OCR dataset...")

ocr_df = pd.read_csv(CLEANED_FILE)

print("OCR rows:", len(ocr_df))


print("Loading dataset split manifest...")

manifest_df = pd.read_csv(MANIFEST_FILE)

print("Manifest rows:", len(manifest_df))


# --------------------------------
# KEEP ONLY WHAT WE NEED
# --------------------------------

manifest_df = manifest_df[
    [
        "image",
        "split"
    ]
]


# --------------------------------
# MERGE OCR + SPLIT
# --------------------------------

df = ocr_df.merge(
    manifest_df,
    on="image",
    how="left"
)


# --------------------------------
# CHECK FOR MISSING SPLITS
# --------------------------------

missing_split = df["split"].isna().sum()

if missing_split > 0:

    print(
        f"WARNING: {missing_split} images "
        "do not have a split."
    )

else:

    print("All images have a split.")


# --------------------------------
# SELECT FINAL TEXT DATA
# --------------------------------

target_columns = [
    "image",
    "clean_text",
    "hard_to_close",
    "default_choice",
    "false_hierarchy",
    "nagging",
    "privacy_zuckering",
    "split"
]

df = df[target_columns]


# --------------------------------
# SAVE COMPLETE TEXT DATASET
# --------------------------------

complete_file = os.path.join(
    OUTPUT_DIR,
    "text_dataset.csv"
)

df.to_csv(
    complete_file,
    index=False,
    encoding="utf-8"
)


# --------------------------------
# SAVE INDIVIDUAL SPLITS
# --------------------------------

for split_name in [
    "train",
    "validation",
    "test"
]:

    split_df = df[
        df["split"] == split_name
    ].copy()

    output_file = os.path.join(
        OUTPUT_DIR,
        f"{split_name}.csv"
    )

    split_df.to_csv(
        output_file,
        index=False,
        encoding="utf-8"
    )

    print(
        f"{split_name}: "
        f"{len(split_df)} images"
    )


# --------------------------------
# LABEL DISTRIBUTION
# --------------------------------

label_columns = [
    "hard_to_close",
    "default_choice",
    "false_hierarchy",
    "nagging",
    "privacy_zuckering"
]

print("\nLabel distribution:")

for split_name in [
    "train",
    "validation",
    "test"
]:

    split_df = df[
        df["split"] == split_name
    ]

    print(f"\n{split_name.upper()}")

    for label in label_columns:

        print(
            f"{label}: "
            f"{split_df[label].sum()}"
        )


print("\n" + "=" * 50)
print("TEXT DATASET PREPARATION COMPLETE")
print("=" * 50)

print(
    f"\nSaved to:\n{OUTPUT_DIR}"
)