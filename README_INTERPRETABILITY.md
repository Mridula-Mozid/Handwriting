# Model Interpretability: Gradient-weighted Class Activation Mapping (Grad-CAM)

## Overview

Grad-CAM provides visual explanations for deep neural network predictions by identifying which image regions most strongly influence classification decisions. For Parkinson's disease detection from handwriting, this visualization reveals whether the model learns clinically relevant patterns or spurious image artifacts.

## Clinical Significance

Parkinson's disease affects motor control, manifesting in handwriting through tremor, micrographia (progressively smaller writing), and reduced pressure. Clinically meaningful predictions should highlight regions where these characteristics are visible—for example, tremor-induced waviness in strokes or pressure-induced faintness. If the model instead highlights irrelevant regions (corners, uniform backgrounds), this would suggest the model has not learned disease-specific features.

## Technical Approach

**Gradient Computation**: For a given input image and predicted class, Grad-CAM computes gradients of the prediction score with respect to the final convolutional layer's activation maps. These gradients indicate how much each spatial location in the feature map contributes to the final classification.

**Weighted Averaging**: Each activation channel is weighted by the global average gradient, producing a single importance score per channel. This weighting emphasizes channels whose activations most directly affect the prediction.

**Spatial Integration**: All weighted activation maps are summed across channels, creating a single 2D heatmap indicating which spatial locations are most important. This heatmap is resized to match the original image dimensions (224×224).

**Normalization**: Heatmap values are normalized to [0, 1] to ensure consistent color mapping across images.

**Visualization**: The heatmap is overlaid on the original grayscale image using a jet colormap where warm colors (red/yellow) indicate high importance and cool colors (blue) indicate low importance. This overlay is saved as a PNG for inspection.

## Output Structure

```
model_interpretability_visualizations/
├── parkinson_V01PE01_gradcam.png           (heatmap overlay)
├── parkinson_V01PE01_preview.png           (side-by-side visualization)
├── parkinson_V02PE03_gradcam.png
├── parkinson_V02PE03_preview.png
├── healthy_V10HE05_gradcam.png
├── healthy_V10HE05_preview.png
└── ...
```

Each image pair shows: (left) original handwriting, (center) Grad-CAM heatmap, (right) heatmap overlaid on original.

## Interpretation Guidelines

**Localized Regions**: Grad-CAM highlighting concentrated on specific stroke regions or pressure variations (visible as intensity changes) suggests the model learned stroke morphology.

**Scattered Highlights**: If highlights are scattered randomly across the image without coherent spatial structure, the model may not have discovered genuine disease patterns.

**Consistent Patterns**: Examining multiple Parkinson's examples should reveal consistent focus on similar pathological regions (e.g., tremor-induced waviness in most Parkinson's cases).

**Comparison Between Groups**: Healthy and Parkinson's heatmaps should differ noticeably, suggesting the model learned class-discriminative features rather than random characteristics.

## Running the Pipeline

```bash
python H_grad_cam.py
```

Prerequisites: Requires trained model checkpoints from the deep learning training stage and preprocessed images.

Expected runtime: 1-5 minutes depending on number of images analyzed (default: 3 random Parkinson's + 3 random healthy samples).

## Research Application

Grad-CAM visualizations serve multiple roles in publication: (1) validation that the model learned interpretable clinical features, (2) publication figure showing which handwriting characteristics drive disease detection, (3) diagnostic aid for understanding predictions on individual patients, and (4) debugging tool to identify when models have learned spurious patterns.

For regulatory approval or clinical deployment, these interpretability visualizations are often required to establish that the AI system has not learned problematic shortcuts or dataset artifacts.
