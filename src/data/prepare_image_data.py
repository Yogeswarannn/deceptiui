from pathlib import Path
import csv
import zipfile
import shutil


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LABELS_PATH = PROJECT_ROOT / "dataset" / "processed" / "labels.csv"

ORIGINAL_DIR = PROJECT_ROOT / "dataset" / "original"

OUTPUT_DIR = PROJECT_ROOT / "dataset" / "processed" / "images"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("DeceptiUI - Image Dataset Preparation")
    print("=" * 60)

    # --------------------------------------------------------
    # Check labels.csv
    # --------------------------------------------------------

    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            f"labels.csv not found:\n{LABELS_PATH}"
        )

    # --------------------------------------------------------
    # Find ZIP
    # --------------------------------------------------------

    fixed_zip = ORIGINAL_DIR / "ContextRico-DP-fixed.zip"
    original_zip = ORIGINAL_DIR / "ContextRico-DP.zip"

    if fixed_zip.exists():
        zip_path = fixed_zip
        print(f"\nUsing fixed archive:")
        print(zip_path)

    elif original_zip.exists():
        zip_path = original_zip
        print("\nFixed archive not found.")
        print("Using original archive:")
        print(original_zip)

    else:
        raise FileNotFoundError(
            "Could not find ContextRico-DP-fixed.zip "
            "or ContextRico-DP.zip"
        )

    # --------------------------------------------------------
    # Read image names from labels.csv
    # --------------------------------------------------------

    image_names = []

    with open(
        LABELS_PATH,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        if "image" not in reader.fieldnames:
            raise ValueError(
                "labels.csv does not contain an 'image' column."
            )

        for row in reader:

            image_name = row["image"].strip()

            if image_name:
                image_names.append(image_name)

    # Remove duplicates while preserving order
    image_names = list(dict.fromkeys(image_names))

    print(f"\nImages required: {len(image_names)}")

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Open ZIP
    # --------------------------------------------------------

    print("\nReading ZIP archive...")

    extracted = 0
    already_exists = 0
    missing = []

    with zipfile.ZipFile(zip_path, "r") as z:

        # Create filename -> ZIP member mapping
        archive_files = {}

        for member in z.namelist():

            if member.endswith("/"):
                continue

            filename = Path(member).name

            if filename:
                archive_files[filename] = member

        print(f"Files found in archive: {len(archive_files)}")

        # ----------------------------------------------------
        # Extract required images only
        # ----------------------------------------------------

        for i, image_name in enumerate(image_names, start=1):

            output_path = OUTPUT_DIR / image_name

            # Already extracted
            if output_path.exists():
                already_exists += 1
                continue

            # Image not found in ZIP
            if image_name not in archive_files:
                missing.append(image_name)
                continue

            member = archive_files[image_name]

            with z.open(member) as source:

                with open(output_path, "wb") as target:

                    shutil.copyfileobj(
                        source,
                        target
                    )

            extracted += 1

            # Progress
            if extracted % 100 == 0:
                print(
                    f"Extracted {extracted}/"
                    f"{len(image_names)}"
                )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)

    print(f"Required images : {len(image_names)}")
    print(f"Newly extracted : {extracted}")
    print(f"Already existed : {already_exists}")
    print(f"Missing         : {len(missing)}")

    if missing:

        print("\nMissing images:")

        for image in missing[:20]:
            print(f"  {image}")

        if len(missing) > 20:
            print(
                f"  ... and {len(missing) - 20} more"
            )

    else:

        print(
            "\nAll required images were extracted successfully."
        )

    print("\nImages location:")
    print(OUTPUT_DIR)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()