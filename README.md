# DeceptiUI

> **Explainable Multimodal Multi-Label Detection of Dark Patterns Using Visual and Textual Evidence**

DeceptiUI is a research-oriented machine learning system designed to detect dark patterns in user interface screenshots by combining visual and textual evidence to produce multi-label predictions with grounded explanations.

---

## Research Question

> *Does combining visual and textual evidence improve multi-label dark-pattern detection compared with using either modality independently?*

```text
                    DeceptiUI
                        |
           +------------+------------+
           |            |            |
           v            v            v
       Text-only    Image-only   Multimodal
           |            |            |
        OCR Text     Screenshot   Image + Text
           |            |            |
        TF-IDF       ResNet-50      Fusion
           |            |            |
           +------------+------------+
                        |
                        v
              Multi-Label Prediction
```

---

## Target Dark Patterns

Evaluated across five classes from the ContextRico-DP dataset (multi-label: a single UI can contain multiple patterns):

| Pattern               | Description                                                                                |
|---------------------  |--------------------------------------------------------------------------------------------|
| **Hard to Close**     | Popups or modal windows that are intentionally difficult to close.                         |
| **Default Choice**    | Options that are automatically selected or nudged as the default.                          |
| **False Hierarchy**   | Visual hierarchy that directs users toward a preferred choice.                             |
| **Nagging**           | Repeated interruptions or requests that divert users from their intended action.           |
| **Privacy Zuckering** | Interfaces that encourage users to share more personal information than intended.          |

---

## Dataset

Built on the **ContextRico-DP** dataset with identical splits reused across text, image, and multimodal experiments:

| Dataset Metric             | Count   | Split          | Samples |
|----------------------------|---------|----------------|---------|
| **Ground-Truth Entries**   | 1,608   | **Train**      | 974     |
| **Selected Screenshots**   | 1,390   | **Validation** | 210     |
| **Target-Label Instances** | 1,624   | **Test**       | 206     |

---

## Project Structure

```text
DeceptiUI/
├── dataset/
│   ├── original/          # Raw dataset files and annotations
│   └── processed/         # Prepared splits and processed data
├── src/
│   ├── data/              # Data preparation scripts
│   ├── ocr/               # OCR extraction and cleaning
│   ├── models/            # Text, image, and multimodal baselines
│   ├── evaluation/        # Metrics and model evaluations
│   └── explainability/    # Evidence extraction and explanations
├── app/
│   ├── backend/
│   └── frontend/
├── notebooks/
├── models/
├── results/
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Evaluation & Explainability

### Metrics
- **Primary Metrics**: Precision, Recall, F1-Score (Macro & Micro)
- **Held-Out Test Set Comparison**:
  - `Text-only (TF-IDF)` &rarr; Test F1
  - `Image-only (ResNet-50)` &rarr; Test F1
  - `Multimodal (Fusion)` &rarr; Test F1

### Explainability
- **Textual Evidence**: OCR-derived phrases and keywords.
- **Visual Evidence**: Relevant screenshot UI regions (buttons, modals, visual hierarchy, and bounding boxes).

---

## Installation

### 1. Python Environment
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Tesseract OCR (Windows)
Tesseract OCR must be installed as a system application. Verify installation:
```powershell
tesseract --version
python -c "import pytesseract; print(pytesseract.get_tesseract_version())"
```

---

## Running the Pipeline

```powershell
# 1. OCR Extraction & Cleaning
python src\ocr\extract_ocr.py
python src\ocr\clean_ocr.py

# 2. Dataset Preparation
python src\data\prepare_text_data.py
python src\data\prepare_image_data.py

# 3. Baseline Model Training
python src\models\text_baseline.py
```