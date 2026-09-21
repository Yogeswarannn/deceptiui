"""
DeceptiUI - Image-Only ResNet-50 Model

Task:
    Multi-label dark-pattern classification from UI screenshots.

Classes:
    0 - Hard to Close
    1 - Default Choice
    2 - False Hierarchy
    3 - Nagging
    4 - Privacy Zuckering
"""

import torch
import torch.nn as nn
from torchvision import models


# ============================================================
# Configuration
# ============================================================

CLASS_NAMES = [
    "Hard to Close",
    "Default Choice",
    "False Hierarchy",
    "Nagging",
    "Privacy Zuckering"
]

NUM_CLASSES = len(CLASS_NAMES)


# ============================================================
# Create ResNet-50
# ============================================================

def create_model(pretrained=True):
    """
    Create the DeceptiUI image-only ResNet-50 model.

    Args:
        pretrained (bool):
            If True, load ImageNet pretrained weights.

    Returns:
        torch.nn.Module:
            ResNet-50 with a 5-output classification head.
    """

    if pretrained:
        weights = models.ResNet50_Weights.DEFAULT
    else:
        weights = None

    model = models.resnet50(weights=weights)

    # Original ResNet-50:
    # 2048 features -> 1000 ImageNet classes
    #
    # DeceptiUI:
    # 2048 features -> 5 dark-pattern classes

    num_features = model.fc.in_features

    model.fc = nn.Linear(
        num_features,
        NUM_CLASSES
    )

    return model


# ============================================================
# Load trained model
# ============================================================

def load_model(checkpoint_path, device=None):
    """
    Load a trained DeceptiUI image-only model.

    Args:
        checkpoint_path:
            Path to the trained .pth checkpoint.

        device:
            torch.device.
            If None, automatically uses CUDA when available.

    Returns:
        model:
            Loaded ResNet-50 model.

        checkpoint:
            Original checkpoint dictionary.
    """

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    # Load checkpoint
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    # Create architecture without downloading
    # ImageNet weights again
    model = create_model(
        pretrained=False
    )

    # Load our trained weights
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)

    # Inference mode
    model.eval()

    return model, checkpoint


# ============================================================
# Calculate class weights
# ============================================================

def get_pos_weights(train_df, label_columns):
    """
    Calculate positive-class weights using
    ONLY the training dataset.

    Used to handle class imbalance.
    """

    positive_counts = (
        train_df[label_columns]
        .sum()
        .values
    )

    negative_counts = (
        len(train_df) - positive_counts
    )

    pos_weight = (
        negative_counts / positive_counts
    )

    return pos_weight


# ============================================================
# Loss function
# ============================================================

def get_loss(
    train_df,
    label_columns,
    device
):
    """
    Create the weighted BCEWithLogitsLoss
    used for the image-only model.
    """

    pos_weight = get_pos_weights(
        train_df,
        label_columns
    )

    pos_weight_tensor = torch.tensor(
        pos_weight,
        dtype=torch.float32,
        device=device
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight_tensor
    )

    return criterion


# ============================================================
# Test model creation
# ============================================================

if __name__ == "__main__":

    model = create_model(
        pretrained=False
    )

    print("DeceptiUI Image-Only Model")
    print("=" * 40)

    print("Architecture: ResNet-50")
    print("Number of classes:", NUM_CLASSES)

    print("\nClasses:")

    for i, class_name in enumerate(CLASS_NAMES):
        print(f"{i}: {class_name}")

    print("\nFinal classifier:")
    print(model.fc)