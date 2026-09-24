"""
Medical Deepfake Detector
CNN Augmented — Grad-CAM for Changed Predictions

Purpose:
    Generate Grad-CAM visualizations for the 10 EXP2 samples whose
    predictions changed between the baseline CNN and augmented CNN.

Important:
    Uses the SAME Grad-CAM methodology as gradcam_cnn_baseline.py:
    - Same CNN architecture
    - Same final convolutional layer: model.features[12]
    - Same positive-class logit
    - Same Grad-CAM calculation
    - Same 128x128 ROI input
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

CHECKPOINT_PATH = "outputs/checkpoints/cnn_augmented_best.pt"

COMPARISON_PATH = (
    "outputs/analysis/cnn_comparison/baseline_vs_augmented.csv"
)

TEST_METADATA_PATH = (
    "outputs/roi_dataset/test_roi_metadata.csv"
)

OUTPUT_DIR = (
    "outputs/analysis/cnn_comparison/augmented_gradcam"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ----------------------------------------------------------------------
# SETUP
# ----------------------------------------------------------------------

print("=" * 80)
print("MEDICAL DEEPFAKE DETECTOR")
print("CNN AUGMENTED — CHANGED PREDICTIONS GRAD-CAM")
print("=" * 80)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print()
print("Device:", DEVICE)


# ----------------------------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------------------------

print()
print("-" * 80)
print("LOADING AUGMENTED CNN")
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
# LOAD COMPARISON
# ----------------------------------------------------------------------

print()
print("-" * 80)
print("LOADING CHANGED PREDICTIONS")
print("-" * 80)

comparison = pd.read_csv(COMPARISON_PATH)

changed = comparison[
    comparison["baseline_predicted_label"]
    != comparison["augmented_predicted_label"]
].copy()

print("Changed predictions:", len(changed))


# ----------------------------------------------------------------------
# LOAD TEST METADATA
# ----------------------------------------------------------------------

metadata = pd.read_csv(TEST_METADATA_PATH)

print("Test metadata:", len(metadata))


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


# Same target layer as baseline Grad-CAM
target_layer = model.features[12]

target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)


# ----------------------------------------------------------------------
# ROI PATH RESOLUTION
# ----------------------------------------------------------------------

def resolve_roi_path(uuid, slice_number):

    matches = metadata[
        (metadata["uuid"].astype(str) == str(uuid))
        &
        (metadata["slice"].astype(int) == int(slice_number))
    ]

    if len(matches) == 0:
        return None

    return str(matches.iloc[0]["roi_path"])


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

    # Same positive-class logit used by baseline Grad-CAM.
    score = output[0]

    score.backward()

    if activations is None:
        raise RuntimeError(
            "Activations were not captured."
        )

    if gradients is None:
        raise RuntimeError(
            "Gradients were not captured."
        )

    weights = gradients.mean(
        dim=(2, 3),
        keepdim=True
    )

    cam = (
        weights * activations
    ).sum(
        dim=1,
        keepdim=True
    )

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

    probability = float(
        torch.sigmoid(output[0])
        .detach()
        .cpu()
    )

    return (
        cam.detach().cpu().numpy(),
        probability
    )


# ----------------------------------------------------------------------
# GENERATE GRAD-CAM
# ----------------------------------------------------------------------

print()
print("=" * 80)
print("GENERATING AUGMENTED GRAD-CAM")
print("=" * 80)

results = []


for i, (_, row) in enumerate(changed.iterrows()):

    uuid = int(row["UUID"])
    slice_number = int(row["slice"])

    roi_path = resolve_roi_path(
        uuid,
        slice_number
    )

    if roi_path is None:
        print(
            "WARNING: ROI not found:",
            uuid,
            slice_number
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
    # METADATA
    # --------------------------------------------------------------

    true_label = int(
        row["true_label"]
    )

    baseline_pred = int(
        row["baseline_predicted_label"]
    )

    augmented_pred = int(
        row["augmented_predicted_label"]
    )

    baseline_probability = float(
        row["baseline_p_tampered"]
    )

    augmented_probability = float(
        row["augmented_p_tampered"]
    )

    transition = str(
        row["transition"]
    )


    # --------------------------------------------------------------
    # FIGURE
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


    # Grad-CAM overlay
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
        "Augmented CNN Grad-CAM"
    )

    ax2.axis("off")


    # Heatmap
    ax3 = fig.add_subplot(1, 3, 3)

    ax3.imshow(
        cam,
        cmap="jet"
    )

    ax3.set_title(
        "Grad-CAM Heatmap"
    )

    ax3.axis("off")


    # Overall title
    true_name = (
        "Tampered"
        if true_label == 1
        else "Authentic"
    )

    baseline_name = (
        "Tampered"
        if baseline_pred == 1
        else "Authentic"
    )

    augmented_name = (
        "Tampered"
        if augmented_pred == 1
        else "Authentic"
    )

    fig.suptitle(
        f"UUID {uuid} | Slice {slice_number} | "
        f"True={true_name} | "
        f"Baseline={baseline_name} "
        f"(P={baseline_probability:.4f}) | "
        f"Augmented={augmented_name} "
        f"(P={augmented_probability:.4f}) | "
        f"{transition}"
    )

    plt.tight_layout()


    # --------------------------------------------------------------
    # SAVE
    # --------------------------------------------------------------

    filename = (
        f"UUID{uuid}_"
        f"SLICE{slice_number}_"
        f"{transition}.png"
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
        "uuid": uuid,
        "slice": slice_number,
        "true_label": true_label,
        "baseline_predicted_label": baseline_pred,
        "augmented_predicted_label": augmented_pred,
        "baseline_p_tampered": baseline_probability,
        "augmented_p_tampered": augmented_probability,
        "augmented_gradcam_probability": model_probability,
        "transition": transition,
        "gradcam_max": float(cam.max()),
        "gradcam_mean": float(cam.mean()),
        "output_path": output_path
    })


    print(
        f"[{i + 1}/{len(changed)}] "
        f"UUID={uuid}, "
        f"Slice={slice_number}, "
        f"{transition} -> saved"
    )


# ----------------------------------------------------------------------
# SAVE SUMMARY
# ----------------------------------------------------------------------

results_df = pd.DataFrame(results)

summary_path = os.path.join(
    OUTPUT_DIR,
    "augmented_gradcam_summary.csv"
)

results_df.to_csv(
    summary_path,
    index=False
)


# ----------------------------------------------------------------------
# COMPLETE
# ----------------------------------------------------------------------

print()
print("=" * 80)
print("AUGMENTED GRAD-CAM ANALYSIS COMPLETE")
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
print("Generated:", len(results), "Grad-CAM visualizations")

print()
print("STATUS: PASS")
print("=" * 80)