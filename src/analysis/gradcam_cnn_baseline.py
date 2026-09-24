"""
Medical Deepfake Detector
CNN Baseline — Grad-CAM Explainability

Purpose:
    Visualize which regions of CT ROI patches influence the CNN baseline
    prediction.

Classes:
    0 = Authentic
    1 = Tampered

This script:
    1. Loads the trained CNN baseline checkpoint.
    2. Loads EXP2 predictions.
    3. Selects representative TP, FN, FP and TN samples.
    4. Computes Grad-CAM using the final convolutional block.
    5. Saves heatmap overlays for visual inspection.

Important:
    This is an explainability analysis only.
    It does not retrain or modify the CNN.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from models.cnn_baseline import CNNBaseline


# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------

CHECKPOINT_PATH = "outputs/checkpoints/cnn_baseline_best.pt"

PREDICTION_PATH = (
    "outputs/analysis/cnn_baseline/EXP2_CNN_PREDICTIONS.csv"
)

ROI_DIR = "outputs/roi_dataset/test"

OUTPUT_DIR = (
    "outputs/analysis/cnn_baseline/gradcam"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ----------------------------------------------------------------------
# SETUP
# ----------------------------------------------------------------------

print("=" * 80)
print("MEDICAL DEEPFAKE DETECTOR")
print("CNN BASELINE — GRAD-CAM EXPLAINABILITY")
print("=" * 80)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print()
print("Device:", DEVICE)

# ----------------------------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------------------------

print()
print("-" * 80)
print("LOADING CNN BASELINE")
print("-" * 80)

model = CNNBaseline()

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.to(DEVICE)
model.eval()

print("Checkpoint loaded successfully.")


# ----------------------------------------------------------------------
# LOAD PREDICTIONS
# ----------------------------------------------------------------------

print()
print("-" * 80)
print("LOADING EXP2 PREDICTIONS")
print("-" * 80)

df = pd.read_csv(PREDICTION_PATH)

print("Samples:", len(df))

required_columns = [
    "uuid",
    "slice",
    "type",
    "binary_label",
    "predicted_label",
    "tampered_probability",
    "prediction_category"
]

missing = [
    c for c in required_columns
    if c not in df.columns
]

if missing:
    raise RuntimeError(
        f"Missing prediction columns: {missing}"
    )


# ----------------------------------------------------------------------
# GRAD-CAM HOOK
# ----------------------------------------------------------------------

activations = None
gradients = None


def forward_hook(module, input, output):
    global activations
    activations = output


def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]


# Final convolutional layer of Block 4
target_layer = model.features[12]

target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)


# ----------------------------------------------------------------------
# ROI PATH RESOLUTION
# ----------------------------------------------------------------------

def resolve_roi_path(row):

    roi_path = row.get("roi_path")

    if (
        roi_path is not None
        and not pd.isna(roi_path)
        and os.path.exists(roi_path)
    ):
        return roi_path

    filename = (
        f"EXP2_{int(row['uuid'])}_{int(row['slice'])}_"
        f"{int(row['x'])}_{int(row['y'])}.npy"
    )

    path = os.path.join(
        ROI_DIR,
        filename
    )

    if not os.path.exists(path):
        return None

    return path


# ----------------------------------------------------------------------
# GRAD-CAM FUNCTION
# ----------------------------------------------------------------------

def compute_gradcam(image):

    global activations
    global gradients

    activations = None
    gradients = None

    image = image.to(DEVICE)
    image.requires_grad_(True)

    model.zero_grad(set_to_none=True)

    output = model(image)

    # Positive-class logit.
    score = output[0]

    score.backward()

    if activations is None:
        raise RuntimeError("Activations were not captured.")

    if gradients is None:
        raise RuntimeError("Gradients were not captured.")

    # activations:
    # (1, C, H, W)

    # gradients:
    # (1, C, H, W)

    weights = gradients.mean(
        dim=(2, 3),
        keepdim=True
    )

    cam = (
        weights * activations
    ).sum(dim=1, keepdim=True)

    cam = F.relu(cam)

    cam = F.interpolate(
        cam,
        size=(128, 128),
        mode="bilinear",
        align_corners=False
    )

    cam = cam[0, 0]

    cam_min = cam.min()
    cam_max = cam.max()

    if (cam_max - cam_min) > 1e-8:
        cam = (
            (cam - cam_min)
            / (cam_max - cam_min)
        )

    return (
        cam.detach().cpu().numpy(),
        float(torch.sigmoid(output[0]).detach().cpu())
    )


# ----------------------------------------------------------------------
# SAMPLE SELECTION
# ----------------------------------------------------------------------

print()
print("-" * 80)
print("SELECTING REPRESENTATIVE SAMPLES")
print("-" * 80)

categories = [
    "TP",
    "FN",
    "FP",
    "TN"
]

selected = []

for category in categories:

    subset = df[
        df["prediction_category"] == category
    ].copy()

    if len(subset) == 0:
        print(category, ": none")
        continue

    # Select the most confident example.
    if category in ["TP", "FP"]:
        row = subset.loc[
            subset["tampered_probability"].idxmax()
        ]
    else:
        row = subset.loc[
            subset["tampered_probability"].idxmin()
        ]

    selected.append(row)

    print(
        f"{category}: "
        f"UUID={int(row['uuid'])}, "
        f"slice={int(row['slice'])}, "
        f"type={row['type']}, "
        f"p_tampered={row['tampered_probability']:.4f}"
    )


# ----------------------------------------------------------------------
# ADD UUID 6031 REPRESENTATIVE SAMPLES
# ----------------------------------------------------------------------

uuid_6031 = df[
    df["uuid"] == 6031
].copy()

if len(uuid_6031):

    print()
    print("Adding UUID 6031 samples...")

    for _, row in uuid_6031.iterrows():

        selected.append(row)

        print(
            f"UUID=6031, "
            f"slice={int(row['slice'])}, "
            f"type={row['type']}, "
            f"p_tampered="
            f"{row['tampered_probability']:.6f}"
        )


# ----------------------------------------------------------------------
# GENERATE GRAD-CAM
# ----------------------------------------------------------------------

print()
print("=" * 80)
print("GENERATING GRAD-CAM")
print("=" * 80)

results = []

for i, row in enumerate(selected):

    roi_path = resolve_roi_path(row)

    if roi_path is None:
        print(
            "WARNING: ROI not found:",
            row["uuid"],
            row["slice"]
        )
        continue

    img = np.load(
        roi_path
    ).astype(np.float32)

    if img.shape != (128, 128):
        print(
            "WARNING: unexpected ROI shape:",
            img.shape
        )
        continue

    tensor = torch.from_numpy(
        img
    ).unsqueeze(0).unsqueeze(0)

    cam, model_probability = compute_gradcam(
        tensor
    )

    # --------------------------------------------------------------
    # CREATE FIGURE
    # --------------------------------------------------------------

    fig = plt.figure(
        figsize=(12, 4)
    )

    # Original ROI
    ax1 = fig.add_subplot(1, 3, 1)

    ax1.imshow(
        img,
        cmap="gray"
    )

    ax1.set_title(
        "Original ROI"
    )

    ax1.axis("off")

    # Grad-CAM
    ax2 = fig.add_subplot(1, 3, 2)

    ax2.imshow(
        img,
        cmap="gray"
    )

    ax2.imshow(
        cam,
        cmap="jet",
        alpha=0.45
    )

    ax2.set_title(
        "Grad-CAM"
    )

    ax2.axis("off")

    # Heatmap alone
    ax3 = fig.add_subplot(1, 3, 3)

    ax3.imshow(
        cam,
        cmap="jet"
    )

    ax3.set_title(
        "Grad-CAM Heatmap"
    )

    ax3.axis("off")

    category = row["prediction_category"]

    true_label = int(
        row["binary_label"]
    )

    predicted_label = int(
        row["predicted_label"]
    )

    probability = float(
        row["tampered_probability"]
    )

    fig.suptitle(
        f"UUID {int(row['uuid'])} | "
        f"Slice {int(row['slice'])} | "
        f"Type {row['type']} | "
        f"{category} | "
        f"True={true_label} | "
        f"Pred={predicted_label} | "
        f"P(tampered)={probability:.4f}"
    )

    plt.tight_layout()

    filename = (
        f"UUID{int(row['uuid'])}_"
        f"SLICE{int(row['slice'])}_"
        f"{row['type']}_"
        f"{category}.png"
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    results.append({
        "uuid": int(row["uuid"]),
        "slice": int(row["slice"]),
        "type": row["type"],
        "category": category,
        "true_label": true_label,
        "predicted_label": predicted_label,
        "prediction_probability": probability,
        "gradcam_max": float(cam.max()),
        "gradcam_mean": float(cam.mean()),
        "output_path": output_path
    })

    print(
        f"[{i + 1}/{len(selected)}] "
        f"Saved {filename}"
    )


# ----------------------------------------------------------------------
# SAVE SUMMARY
# ----------------------------------------------------------------------

results_df = pd.DataFrame(results)

summary_path = os.path.join(
    OUTPUT_DIR,
    "gradcam_summary.csv"
)

results_df.to_csv(
    summary_path,
    index=False
)

print()
print("=" * 80)
print("GRAD-CAM ANALYSIS COMPLETE")
print("=" * 80)

print()
print("Images saved to:")
print(
    os.path.abspath(OUTPUT_DIR)
)

print()
print("Summary:")
print(
    os.path.abspath(summary_path)
)

print()
print("STATUS: PASS")
print("=" * 80)

