"""
Medical Deepfake Detector
CNN Baseline Model

Purpose:
    Binary classification of 128x128 single-channel CT ROI patches.

Classes:
    0 = Authentic
    1 = Tampered

This file defines the CNN architecture only.
Training will be implemented in a separate step.
"""

import torch
import torch.nn as nn


class CNNBaseline(nn.Module):
    """
    Small CNN baseline for binary CT ROI classification.

    Input:
        (batch_size, 1, 128, 128)

    Output:
        (batch_size,)
        Raw logits for binary classification.
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # 128x128 -> 64x64

            # Block 2
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # 64x64 -> 32x32

            # Block 3
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # 32x32 -> 16x16

            # Block 4
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # 16x16 -> 8x8
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                in_features=256 * 8 * 8,
                out_features=128
            ),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),

            nn.Linear(
                in_features=128,
                out_features=1
            )
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Tensor of shape (B, 1, 128, 128)

        Returns:
            Tensor of shape (B,)
            Raw binary classification logits.
        """

        x = self.features(x)
        x = self.classifier(x)

        return x.squeeze(1)


def count_parameters(model):
    """
    Count trainable parameters in the model.
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


if __name__ == "__main__":
    print("=" * 70)
    print("MEDICAL DEEPFAKE DETECTOR")
    print("CNN BASELINE MODEL CHECK")
    print("=" * 70)

    print()
    print("Creating model...")

    model = CNNBaseline()

    print("Model created successfully.")
    print()

    print("-" * 70)
    print("MODEL INFORMATION")
    print("-" * 70)

    print(f"Trainable parameters: {count_parameters(model):,}")

    print()
    print("Testing forward pass...")

    # Test with the exact shape produced by the existing DataLoader.
    sample_input = torch.randn(8, 1, 128, 128)

    with torch.no_grad():
        output = model(sample_input)

    print(f"Input shape : {tuple(sample_input.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Output dtype: {output.dtype}")

    print()
    print("-" * 70)

    if output.shape != (8,):
        raise RuntimeError(
            f"Unexpected output shape: {tuple(output.shape)}"
        )

    if not torch.isfinite(output).all():
        raise RuntimeError(
            "Model produced non-finite output values."
        )

    print("FORWARD PASS: PASS")
    print()
    print("CNN BASELINE MODEL CHECK COMPLETE")
    print("STATUS: PASS")
    print("=" * 70)