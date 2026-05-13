# Model Interpretability: Gradient-weighted Class Activation Mapping (Grad-CAM)

## Overview

Grad-CAM provides visual explanations for deep neural network predictions by highlighting image regions that most influenced the output. In this project, Grad-CAM is run separately for `Public_Dataset` and `BD_Dataset`.

## Runtime Labeling

Grad-CAM now prints explicit dataset context when starting:

- dataset tag
- model checkpoint path
- output directory

This makes it immediately clear whether the run is using BD or Public models/images.

## Output Structure

```
model_interpretability_visualizations/
├── Public_Dataset/
│   ├── healthy_*_gradcam.png
│   ├── healthy_*_preview.png
│   ├── parkinson_*_gradcam.png
│   └── parkinson_*_preview.png
└── BD_Dataset/
    ├── healthy_*_gradcam.png
    ├── healthy_*_preview.png
    ├── parkinson_*_gradcam.png
    └── parkinson_*_preview.png
```

Each preview image contains original image, heatmap, and overlay.

## Running the Pipeline

```bash
# Public dataset, fold 1 model
python H_grad_cam.py --dataset Public_Dataset --model-fold 1

# BD dataset, fold 1 model
python H_grad_cam.py --dataset BD_Dataset --model-fold 1
```

## Notes

- The script supports `.png`, `.jpg`, and `.jpeg` source images.
- It samples up to available images per class (healthy/parkinson), so it remains stable on smaller datasets.
