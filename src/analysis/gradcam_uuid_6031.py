import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from src.models.cnn_baseline import CNNBaseline


METADATA_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "roi_dataset"
    / "test_roi_metadata.csv"
)

BASELINE_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "cnn_baseline_best.pt"
)

AUGMENTED_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "cnn_augmented_best.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "analysis"
    / "uuid_6031"
)

BASELINE_OUTPUT = OUTPUT_DIR / "baseline_gradcam"
AUGMENTED_OUTPUT = OUTPUT_DIR / "augmented_gradcam"


def load_model(checkpoint_path, device):

    model = CNNBaseline().to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(
            checkpoint
        )

    model.eval()

    return model


def generate_gradcam(
    model,
    image,
    device
):

    activations = []
    gradients = []

    target_layer = model.features[12]

    def forward_hook(module, inp, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    forward_handle = target_layer.register_forward_hook(
        forward_hook
    )

    backward_handle = target_layer.register_full_backward_hook(
        backward_hook
    )

    tensor = torch.from_numpy(
        image.astype(np.float32)
    )

    tensor = tensor.unsqueeze(0).unsqueeze(0).to(device)

    model.zero_grad()

    output = model(tensor)

    score = output[0]

    probability = torch.sigmoid(
        output[0]
    ).item()

    score.backward()

    activation = activations[0]
    gradient = gradients[0]

    weights = gradient.mean(
        dim=(2, 3),
        keepdim=True
    )

    cam = (
        weights * activation
    ).sum(
        dim=1,
        keepdim=True
    )

    cam = torch.relu(cam)

    cam = torch.nn.functional.interpolate(
        cam,
        size=image.shape,
        mode="bilinear",
        align_corners=False
    )

    cam = cam.squeeze().detach().cpu().numpy()

    if cam.max() > cam.min():

        cam = (
            cam - cam.min()
        ) / (
            cam.max() - cam.min()
        )

    else:

        cam = np.zeros_like(cam)

    forward_handle.remove()
    backward_handle.remove()

    return probability, cam


def save_visualization(
    image,
    cam,
    probability,
    uuid,
    slice_number,
    model_name,
    output_path
):

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 4)
    )

    axes[0].imshow(
        image,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[0].set_title(
        "Original ROI"
    )

    axes[0].axis("off")

    axes[1].imshow(
        image,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[1].imshow(
        cam,
        cmap="jet",
        alpha=0.45,
        vmin=0,
        vmax=1
    )

    axes[1].set_title(
        "Grad-CAM Overlay"
    )

    axes[1].axis("off")

    axes[2].imshow(
        cam,
        cmap="jet",
        vmin=0,
        vmax=1
    )

    axes[2].set_title(
        "Grad-CAM Heatmap"
    )

    axes[2].axis("off")

    fig.suptitle(
        f"UUID 6031 | Slice {slice_number} | "
        f"{model_name} | P(Tampered)={probability:.6f}",
        fontsize=13
    )

    plt.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


def main():

    print("=" * 80)
    print("UUID 6031 GRAD-CAM GENERATION")
    print("=" * 80)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("Device:", device)

    metadata = pd.read_csv(
        METADATA_CSV
    )

    samples = metadata[
        metadata["uuid"].astype(str) == "6031"
    ].copy()

    samples = samples.sort_values(
        "slice"
    )

    print(
        "UUID 6031 samples:",
        len(samples)
    )

    baseline_model = load_model(
        BASELINE_CHECKPOINT,
        device
    )

    augmented_model = load_model(
        AUGMENTED_CHECKPOINT,
        device
    )

    summary = []

    for _, row in samples.iterrows():

        slice_number = int(
            row["slice"]
        )

        roi_path = Path(
            row["roi_path"]
        )

        image = np.load(
            roi_path
        ).astype(np.float32)

        print()
        print(
            f"Processing slice {slice_number}"
        )

        baseline_probability, baseline_cam = generate_gradcam(
            baseline_model,
            image,
            device
        )

        augmented_probability, augmented_cam = generate_gradcam(
            augmented_model,
            image,
            device
        )

        baseline_path = (
            BASELINE_OUTPUT
            / f"UUID6031_SLICE{slice_number}_baseline.png"
        )

        augmented_path = (
            AUGMENTED_OUTPUT
            / f"UUID6031_SLICE{slice_number}_augmented.png"
        )

        save_visualization(
            image,
            baseline_cam,
            baseline_probability,
            6031,
            slice_number,
            "Baseline CNN",
            baseline_path
        )

        save_visualization(
            image,
            augmented_cam,
            augmented_probability,
            6031,
            slice_number,
            "Augmented CNN",
            augmented_path
        )

        summary.append(
            {
                "uuid": 6031,
                "slice": slice_number,
                "true_label": int(row["binary_label"]),
                "baseline_p_tampered": baseline_probability,
                "augmented_p_tampered": augmented_probability,
                "baseline_gradcam": str(baseline_path),
                "augmented_gradcam": str(augmented_path),
            }
        )

    summary_df = pd.DataFrame(
        summary
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    summary_df.to_csv(
        OUTPUT_DIR
        / "uuid_6031_gradcam_summary.csv",
        index=False
    )

    print()
    print("=" * 80)
    print("GRAD-CAM GENERATION COMPLETE")
    print("=" * 80)

    print()
    print(
        "Saved to:",
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()