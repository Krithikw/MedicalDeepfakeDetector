import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.models.cnn_baseline import CNNBaseline

# ============================================================
# PATHS
# ============================================================

CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "cnn_augmented_best.pt"
)


# ============================================================
# MODEL LOADING
# ============================================================

def load_final_model(device):

    print("=" * 70)
    print("LOADING FINAL MEDICAL IMAGE AUTHENTICITY MODEL")
    print("=" * 70)

    print()
    print("Checkpoint:")
    print(CHECKPOINT)

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT}"
        )

    model = CNNBaseline().to(device)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device
    )

    if "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        epoch = checkpoint.get(
            "epoch",
            "Unknown"
        )

        print()
        print("Checkpoint epoch:", epoch)

        if "val_macro_f1" in checkpoint:

            print(
                "Validation Macro F1:",
                f"{checkpoint['val_macro_f1']:.6f}"
            )

    else:

        model.load_state_dict(
            checkpoint
        )

    model.eval()

    print()
    print("Model loaded successfully.")

    return model


# ============================================================
# ROI LOADING
# ============================================================

def load_roi(roi_path):

    roi_path = Path(roi_path)

    if not roi_path.exists():
        raise FileNotFoundError(
            f"ROI file not found:\n{roi_path}"
        )

    image = np.load(
        roi_path
    ).astype(np.float32)

    if image.shape != (128, 128):

        raise ValueError(
            f"Expected ROI shape (128, 128), "
            f"but received {image.shape}"
        )

    if not np.isfinite(image).all():

        raise ValueError(
            "ROI contains NaN or infinite values."
        )

    return image


# ============================================================
# PREDICTION
# ============================================================

def predict(model, image, device):

    tensor = torch.from_numpy(
        image
    )

    tensor = (
        tensor
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():

        output = model(
            tensor
        )

        p_tampered = torch.sigmoid(
            output[0]
        ).item()

    p_authentic = 1.0 - p_tampered

    if p_tampered >= 0.5:

        predicted_class = "TAMPERED"

    else:

        predicted_class = "AUTHENTIC"

    confidence = max(
        p_tampered,
        p_authentic
    )

    return (
        predicted_class,
        p_tampered,
        p_authentic,
        confidence
    )

# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam(model, image, device, output_path):

    model.eval()

    tensor = torch.from_numpy(
        image
    ).unsqueeze(0).unsqueeze(0).to(device)

    activations = []
    gradients = []

    target_layer = model.features[12]

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    forward_handle = target_layer.register_forward_hook(
        forward_hook
    )

    backward_handle = target_layer.register_full_backward_hook(
        backward_hook
    )

    try:

        output = model(tensor)

        model.zero_grad()

        output[0].backward()

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

        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=(128, 128),
            mode="bilinear",
            align_corners=False
        )

        cam = cam[0, 0].detach().cpu().numpy()

        cam_min = cam.min()
        cam_max = cam.max()

        if cam_max > cam_min:

            cam = (
                cam - cam_min
            ) / (
                cam_max - cam_min
            )

        else:

            cam = np.zeros_like(cam)

        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(12, 4)
        )

        axes[0].imshow(
            image,
            cmap="gray"
        )

        axes[0].set_title(
            "Original ROI"
        )

        axes[1].imshow(
            cam,
            cmap="jet"
        )

        axes[1].set_title(
            "Grad-CAM"
        )

        axes[2].imshow(
            image,
            cmap="gray"
        )

        axes[2].imshow(
            cam,
            cmap="jet",
            alpha=0.45
        )

        axes[2].set_title(
            "Grad-CAM Overlay"
        )

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()

        plt.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close(fig)

    finally:

        forward_handle.remove()
        backward_handle.remove()

    return output_path

# ============================================================
# REPORT
# ============================================================

def print_report(
    roi_path,
    predicted_class,
    p_tampered,
    p_authentic,
    confidence
):

    print()
    print("=" * 70)
    print(
        "MEDICAL IMAGE AUTHENTICITY ANALYSIS REPORT"
    )
    print("=" * 70)

    print()

    print(
        "Input ROI          :",
        roi_path
    )

    print(
        "Image Size         : 128 × 128"
    )

    print(
        "Model              : CNN with Training Augmentation"
    )

    print(
        "Model Checkpoint   : cnn_augmented_best.pt"
    )

    print()
    print("-" * 70)
    print("CLASSIFICATION RESULT")
    print("-" * 70)

    print()

    print(
        "Predicted Class    :",
        predicted_class
    )

    print(
        "Tampered Probability :",
        f"{p_tampered * 100:.2f}%"
    )

    print(
        "Authentic Probability:",
        f"{p_authentic * 100:.2f}%"
    )

    print(
        "Confidence         :",
        f"{confidence * 100:.2f}%"
    )

    print()
    print("-" * 70)
    print("INTERPRETATION")
    print("-" * 70)

    print()

    print(
        "The trained CNN classified the supplied ROI"
    )

    print(
        f"as {predicted_class} based on the learned"
    )

    print(
        "image characteristics."
    )

    print()
    print("-" * 70)
    print("LIMITATION")
    print("-" * 70)

    print()

    print(
        "This output represents an experimental"
    )

    print(
        "deep-learning classification result."
    )

    print(
        "It is not a clinical diagnosis and should"
    )

    print(
        "not be used independently for clinical"
    )

    print(
        "decision-making."
    )

    print()
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("FINAL INFERENCE PIPELINE")
    print("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("Device:", device)

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    model = load_final_model(
        device
    )

    print()
    print("=" * 70)
    print("ENTER ROI PATH")
    print("=" * 70)

    roi_path = input(
        "\nROI .npy path: "
    ).strip()

    if not roi_path:

        raise ValueError(
            "No ROI path was supplied."
        )

    image = load_roi(
        roi_path
    )

    (
        predicted_class,
        p_tampered,
        p_authentic,
        confidence
    ) = predict(
        model,
        image,
        device
    )

    # ========================================================
    # GENERATE GRAD-CAM
    # ========================================================

    gradcam_dir = (
        PROJECT_ROOT
        / "outputs"
        / "inference"
        / "gradcam"
    )

    gradcam_filename = (
        Path(roi_path).stem
        + "_gradcam.png"
    )

    gradcam_path = (
        gradcam_dir
        / gradcam_filename
    )

    generate_gradcam(
        model,
        image,
        device,
        gradcam_path
    )

    print_report(
        roi_path,
        predicted_class,
        p_tampered,
        p_authentic,
        confidence
    )

    print()
    print("-" * 70)
    print("GRAD-CAM OUTPUT")
    print("-" * 70)
    print()
    print("Saved visualization:")
    print(gradcam_path)


if __name__ == "__main__":
    main()