import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class MultimodalDataset(Dataset):

    def __init__(self, csv_path, image_dir, transform=None):

        self.data = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform

        self.label_columns = [
            "hard_to_close",
            "default_choice",
            "false_hierarchy",
            "nagging",
            "privacy_zuckering"
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        # -------------------------
        # Image
        # -------------------------

        image_name = row["image"]

        image_path = os.path.join(
            self.image_dir,
            image_name
        )

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # -------------------------
        # OCR text
        # -------------------------

        text = str(row["clean_text"])

        # -------------------------
        # Labels
        # -------------------------

        labels = row[self.label_columns].values.astype("float32")

        labels = torch.tensor(labels)

        return {
            "image": image,
            "text": text,
            "labels": labels,
            "image_name": image_name
        }


# -----------------------------------
# Image transformations
# -----------------------------------

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

if __name__ == "__main__":

    csv_path = "dataset/processed/text/train.csv"
    image_dir = "dataset/processed/images"

    dataset = MultimodalDataset(
        csv_path=csv_path,
        image_dir=image_dir,
        transform=train_transform
    )

    print("Dataset size:", len(dataset))

    sample = dataset[0]

    print("Image shape:", sample["image"].shape)
    print("Text:", sample["text"])
    print("Labels:", sample["labels"])
    print("Image name:", sample["image_name"])