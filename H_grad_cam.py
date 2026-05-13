import cv2
import random
import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image

import torch.nn as nn
from torchvision import transforms, models

# ============================================================
# CONFIG
# ============================================================

IMG_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_ROOT / "trained_models_checkpoints" / "resnet18_fold_1.pth"

CLASS_FOLDERS = {
    "parkinson": PROJECT_ROOT / "preprocessed_images" / "grayscale" / "parkinson",
    "healthy": PROJECT_ROOT / "preprocessed_images" / "grayscale" / "healthy",
}

OUTPUT_DIR = PROJECT_ROOT / "model_interpretability_visualizations"

OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5],
        std=[0.5]
    )
])

# ============================================================
# BUILD MODEL
# ============================================================

def build_model():
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 2)

    return model

# ============================================================
# GRAD CAM CLASS
# ============================================================

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.register_hooks()

    def register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, class_idx=None):

        output = self.model(input_tensor)

        if class_idx is None:

            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()

        target = output[:, class_idx]

        target.backward()

        gradients = self.gradients[0]

        activations = self.activations[0]

        weights = gradients.mean(dim=(1, 2))

        cam = torch.zeros(
            activations.shape[1:],
            dtype=torch.float32
        )

        for i, w in enumerate(weights):

            cam += w * activations[i]

        cam = torch.relu(cam)

        cam = cam.detach().cpu().numpy()

        cam = cv2.resize(
            cam,
            (IMG_SIZE, IMG_SIZE)
        )

        cam = (
            cam - cam.min()
        ) / (
            cam.max() - cam.min() + 1e-8
        )

        return cam

# ============================================================
# LOAD MODEL
# ============================================================

model = build_model()

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model = model.to(DEVICE)

model.eval()

# ============================================================
# TARGET LAYER
# ============================================================

target_layer = model.layer4

gradcam = GradCAM(
    model,
    target_layer
)

# ============================================================
# LOAD IMAGES
# ============================================================

def get_random_images(folder, count):

    image_paths = sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    )

    if len(image_paths) < count:

        raise ValueError(
            f"Not enough PNG files in {folder} to sample {count} images."
        )

    return random.sample(image_paths, count)


def generate_gradcam(image_path, class_name):

    original = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if original is None:

        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    original_resized = cv2.resize(
        original,
        (IMG_SIZE, IMG_SIZE)
    )

    pil_image = Image.fromarray(
        original_resized
    )

    input_tensor = transform(
        pil_image
    ).unsqueeze(0).to(DEVICE)

    cam = gradcam.generate_cam(input_tensor)

    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam),
        cv2.COLORMAP_JET
    )

    original_bgr = cv2.cvtColor(
        original_resized,
        cv2.COLOR_GRAY2BGR
    )

    overlay = cv2.addWeighted(
        original_bgr,
        0.6,
        heatmap,
        0.4,
        0
    )

    save_path = OUTPUT_DIR / f"{class_name}_{image_path.stem}_gradcam.png"

    cv2.imwrite(
        str(save_path),
        overlay
    )

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(original_resized, cmap='gray')
    plt.title(f"Original - {class_name}")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(cam, cmap='jet')
    plt.title("Grad-CAM")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Overlay")
    plt.axis("off")

    plt.tight_layout()

    preview_path = OUTPUT_DIR / f"{class_name}_{image_path.stem}_preview.png"

    plt.savefig(preview_path, bbox_inches="tight")
    plt.close()

    print("\nSaved Grad-CAM result:")
    print(save_path)
    print("Saved preview:")
    print(preview_path)


selected_images = [
    ("parkinson", image_path)
    for image_path in get_random_images(CLASS_FOLDERS["parkinson"], 3)
] + [
    ("healthy", image_path)
    for image_path in get_random_images(CLASS_FOLDERS["healthy"], 3)
]

for class_name, image_path in selected_images:

    generate_gradcam(image_path, class_name)