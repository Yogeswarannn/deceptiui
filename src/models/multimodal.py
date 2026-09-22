import torch
import torch.nn as nn


class MultimodalModelV2(nn.Module):

    def __init__(self, image_encoder):
        super().__init__()

        self.image_encoder = image_encoder

        # Image projection: 2048 -> 512
        self.image_projection = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Text projection: 7102 -> 256
        self.text_projection = nn.Sequential(
            nn.Linear(7102, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Fusion: 512 + 256 -> 256 -> 5
        self.fusion = nn.Sequential(
            nn.Linear(512 + 256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 5)
        )

    def forward(self, images, text_features):

        # Image branch
        image_features = self.image_encoder(images)
        image_features = torch.flatten(image_features, 1)
        image_features = self.image_projection(image_features)

        # Text branch
        text_features = self.text_projection(text_features)

        # Combine image and text features
        combined = torch.cat(
            [image_features, text_features],
            dim=1
        )

        # Final prediction
        outputs = self.fusion(combined)

        return outputs