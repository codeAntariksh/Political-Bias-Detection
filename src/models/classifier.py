import torch.nn as nn


class BiasClassifier(nn.Module):
    """MLP head: 768 -> 512 -> 256 -> num_classes (GELU, dropout 0.30/0.20)."""

    def __init__(self, input_dim=768, num_classes=3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(256, num_classes),
        )

    def forward(self, fused):
        return self.classifier(fused)
