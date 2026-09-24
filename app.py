import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import streamlit as st
import torch
import json
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.models.cnn_baseline import CNNBaseline


# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "cnn_augmented_best.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "inference"
    / "streamlit"
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Medical Image Authenticity Analysis",
    page_icon="🩻",
    layout="wide"
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

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

    else:

        model.load_state_dict(
            checkpoint
        )

        epoch = "Unknown"

    model.eval()

    return model, device, epoch


# ============================================================
# ROI VALIDATION
# ============================================================

def validate_roi(image):

    image = np.asarray(
        image,
        dtype=np.float32
    )

    if image.shape != (128, 128):

        raise ValueError(
            f"Expected a 128 × 128 ROI, "
            f"but received {image.shape}."
        )

    if not np.isfinite(image).all():

        raise ValueError(
            "The ROI contains NaN or infinite values."
        )

    return image


# ============================================================
# PREDICTION
# ============================================================

def predict(model, image, device):

    tensor = torch.from_numpy(
        image
    ).unsqueeze(0).unsqueeze(0).to(device)

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
# CONFIDENCE INTERPRETATION
# ============================================================

def confidence_level(confidence):

    if confidence >= 0.80:

        return "Higher model confidence"

    elif confidence >= 0.60:

        return "Moderate model confidence"

    else:

        return "Low model confidence"


# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam(
    model,
    image,
    device
):

    model.eval()

    tensor = torch.from_numpy(
        image
    ).unsqueeze(0).unsqueeze(0).to(device)

    activations = []
    gradients = []

    target_layer = model.features[12]

    def forward_hook(module, input, output):

        activations.append(output)

    def backward_hook(
        module,
        grad_input,
        grad_output
    ):

        gradients.append(
            grad_output[0]
        )

    forward_handle = target_layer.register_forward_hook(
        forward_hook
    )

    backward_handle = target_layer.register_full_backward_hook(
        backward_hook
    )

    try:

        output = model(
            tensor
        )

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

        cam = F.relu(
            cam
        )

        cam = F.interpolate(
            cam,
            size=(128, 128),
            mode="bilinear",
            align_corners=False
        )

        cam = (
            cam[0, 0]
            .detach()
            .cpu()
            .numpy()
        )

        cam_min = cam.min()
        cam_max = cam.max()

        if cam_max > cam_min:

            cam = (
                cam - cam_min
            ) / (
                cam_max - cam_min
            )

        else:

            cam = np.zeros_like(
                cam
            )

    finally:

        forward_handle.remove()
        backward_handle.remove()

    return cam


# ============================================================
# VISUALIZATION
# ============================================================

def create_gradcam_figure(
    image,
    cam
):

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    axes[0].imshow(
        image,
        cmap="gray"
    )

    axes[0].set_title(
        "Original ROI",
        fontsize=13
    )

    axes[1].imshow(
        cam,
        cmap="jet"
    )

    axes[1].set_title(
        "Grad-CAM",
        fontsize=13
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
        "Grad-CAM Overlay",
        fontsize=13
    )

    for ax in axes:

        ax.axis("off")

    plt.tight_layout()

    return fig


# ============================================================
# APPLICATION HEADER
# ============================================================

st.title(
    "Medical Image Authenticity Analysis"
)

st.caption(
    "Experimental deep-learning system for medical image "
    "authenticity classification and Grad-CAM visualization."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "Model Information"
    )

    st.write(
        "**Model:** CNN with Training Augmentation"
    )

    st.write(
        "**Input:** 128 × 128 single-channel ROI"
    )

    st.write(
        "**Checkpoint:** cnn_augmented_best.pt"
    )

    st.write(
        "**Classification threshold:** 0.50"
    )

    st.divider()

    st.warning(
        "This is an experimental research system. "
        "It is not a clinical diagnostic tool."
    )


# ============================================================
# MODEL INITIALIZATION
# ============================================================

try:

    model, device, epoch = load_model()

except Exception as error:

    st.error(
        f"Model loading failed:\n{error}"
    )

    st.stop()


# ============================================================
# UPLOAD
# ============================================================

st.header(
    "Upload ROI"
)

st.write(
    "Upload a preprocessed `.npy` ROI with shape "
    "**128 × 128**."
)

uploaded_file = st.file_uploader(
    "Select ROI file",
    type=["npy"]
)


# ============================================================
# ANALYSIS
# ============================================================

if uploaded_file is not None:

    try:

        image = np.load(
            uploaded_file
        ).astype(np.float32)

        image = validate_roi(
            image
        )

    except Exception as error:

        st.error(
            f"Invalid ROI:\n{error}"
        )

        st.stop()

    st.success(
        "ROI loaded successfully."
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

    confidence_text = confidence_level(
        confidence
    )

    cam = generate_gradcam(
        model,
        image,
        device
    )

    # ========================================================
    # CLASSIFICATION RESULT
    # ========================================================

    st.header(
        "Classification Result"
    )

    col1, col2, col3 = st.columns(
        3
    )

    with col1:

        st.metric(
            "Predicted Class",
            predicted_class
        )

    with col2:

        st.metric(
            "Tampered Probability",
            f"{p_tampered * 100:.2f}%"
        )

    with col3:

        st.metric(
            "Authentic Probability",
            f"{p_authentic * 100:.2f}%"
        )

    st.progress(
        confidence
    )

    st.write(
        f"**Model confidence:** "
        f"{confidence * 100:.2f}%"
    )

    st.write(
        f"**Confidence interpretation:** "
        f"{confidence_text}"
    )

    # ========================================================
    # ORIGINAL IMAGE
    # ========================================================

    st.header(
        "Input ROI"
    )

    st.image(
        image,
        caption="Uploaded 128 × 128 ROI",
        clamp=True,
        width=400
    )

    # ========================================================
    # GRAD-CAM
    # ========================================================

    st.header(
        "Grad-CAM Analysis"
    )

    st.write(
        "Grad-CAM highlights image regions that contributed "
        "to the model's output. These regions should not be "
        "interpreted as independently verified manipulation "
        "or anatomical ground truth."
    )

    fig = create_gradcam_figure(
        image,
        cam
    )

    st.pyplot(
        fig,
        clear_figure=True
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    st.header(
        "Analysis Summary"
    )

    st.info(
        f"The trained CNN classified the supplied ROI as "
        f"**{predicted_class}** with a model confidence of "
        f"**{confidence * 100:.2f}%**. "
        f"The estimated probability of the ROI being tampered "
        f"is **{p_tampered * 100:.2f}%**, while the estimated "
        f"probability of it being authentic is "
        f"**{p_authentic * 100:.2f}%**."
    )
    
    # ========================================================
    # JSON EXPORT
    # ========================================================

    analysis_report = {
        "analysis": {
            "input": {
                "filename": uploaded_file.name,
                "image_size": "128 × 128"
            },
            "model": {
                "name": "CNN with Training Augmentation",
                "checkpoint": "cnn_augmented_best.pt",
                "classification_threshold": 0.50
            },
            "result": {
                "predicted_class": predicted_class,
                "tampered_probability": round(p_tampered, 6),
                "authentic_probability": round(p_authentic, 6),
                "confidence": round(confidence, 6),
                "confidence_interpretation": confidence_text
            },
            "explainability": {
                "gradcam_generated": True,
                "interpretation": (
                    "Grad-CAM provides model-attribution information "
                    "and does not establish the true location of "
                    "image manipulation."
                )
            },
            "limitations": {
                "clinical_use": False,
                "experimental_system": True
            }
        }
    }

    json_data = json.dumps(
        analysis_report,
        indent=4
    )

    st.header(
        "Export Analysis"
    )

    st.download_button(
        label="Download Analysis Report (JSON)",
        data=json_data,
        file_name=f"{Path(uploaded_file.name).stem}_analysis.json",
        mime="application/json"
    )

    # ========================================================
    # LIMITATION
    # ========================================================

    st.header(
        "Important Limitation"
    )

    st.warning(
        "This output represents an experimental deep-learning "
        "classification result and model explanation. It is "
        "not a clinical diagnosis and should not be used "
        "independently for clinical decision-making. "
        "Grad-CAM provides model-attribution information and "
        "does not establish the true location of image "
        "manipulation."
    )