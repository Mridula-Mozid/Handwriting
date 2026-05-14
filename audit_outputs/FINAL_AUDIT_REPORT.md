# Handwriting Modality — Scientific Audit

This report is a reproducible, scientifically grounded audit of the handwriting unimodal pipeline in this repository. It uses only the already-generated outputs present in the workspace (preprocessing outputs, model checkpoints, result CSVs, GradCAM preview images, and computed metrics). This document frames the work as exploratory, not diagnostic.

Folders and artifacts used (examples):
- Preprocessed metadata: [preprocessed_images/Public_Dataset/metadata.csv](preprocessed_images/Public_Dataset/metadata.csv), [preprocessed_images/BD_Dataset/metadata.csv](preprocessed_images/BD_Dataset/metadata.csv)
- Deep learning final results: [deep_learning_results/Public_Dataset/final_cross_validation_results.csv](deep_learning_results/Public_Dataset/final_cross_validation_results.csv), [deep_learning_results/BD_Dataset/final_cross_validation_results.csv](deep_learning_results/BD_Dataset/final_cross_validation_results.csv)
- DL fold predictions and confusion matrices: [deep_learning_results/BD_Dataset/fold_3_predictions.csv](deep_learning_results/BD_Dataset/fold_3_predictions.csv), [deep_learning_results/BD_Dataset/fold_3_cm.png](deep_learning_results/BD_Dataset/fold_3_cm.png)
- Classical ML summary: [classical_ml_results/BD_Dataset/classical_ml_results.csv](classical_ml_results/BD_Dataset/classical_ml_results.csv)
- GradCAM previews: [model_interpretability_visualizations/BD_Dataset/parkinson_BPP019_hand_spiral_01_preview.png](model_interpretability_visualizations/BD_Dataset/parkinson_BPP019_hand_spiral_01_preview.png)
- Computed DL per-fold calibration/metrics: [handwriting_audit_outputs/BD_Dataset/dl_fold_metrics_summary.csv](handwriting_audit_outputs/BD_Dataset/dl_fold_metrics_summary.csv), [handwriting_audit_outputs/Public_Dataset/dl_fold_metrics_summary.csv](handwriting_audit_outputs/Public_Dataset/dl_fold_metrics_summary.csv)

---

**1. Executive summary**

- Pipeline executed: preprocessing → handcrafted feature extraction (existing) → deep learning training (ResNet-18 transfer learning) → classical ML baselines → GradCAM explainability. All produced outputs were inspected; no re-training was performed for this audit.
- Dataset sizes (from preprocessed metadata): Public_Dataset = 102 images (51 healthy / 51 parkinson), 28 subjects (mean ≈ 3.64 images/subject). BD_Dataset = 36 images (15 healthy / 21 parkinson), 36 subjects (1 image/subject). See metadata files above.
- Main findings:
  - Public_Dataset: DL delivered strong ranking performance (ROC-AUC ≈ 0.956 ± 0.047) and moderate calibrated decision metrics (accuracy ≈ 0.769 ± 0.134) but shows fold variability in threshold metrics.
  - BD_Dataset: DL ranking remains high (ROC-AUC ≈ 0.907 ± 0.146) but threshold metrics are unstable and modest (accuracy ≈ 0.579 ± 0.152). Classical SVM on handcrafted features reports higher accuracy on BD (≈0.807 ± 0.070), suggesting handcrafted features are highly informative in the low-data regime.
  - Calibration issues: per-fold Expected Calibration Error (ECE) is non-negligible (BD mean ECE ≈ 0.29, Public mean ECE ≈ 0.22) and Brier scores are higher on BD than Public (BD mean ≈ 0.23; Public mean ≈ 0.15). See [handwriting_audit_outputs/*/dl_fold_metrics_summary.csv](handwriting_audit_outputs/).
- Multimodal readiness: Handwriting appears a strong primary behavioral modality; facial modality (outside scope) remains complementary. A conservative late-fusion approach (probability- or low-dim embedding-level fusion) is recommended.

---

**2. Dataset analysis**

Public_Dataset (separate):
- Total images: 102
- Subjects: 28
- Healthy / Parkinson: 51 / 51 (balanced)
- Images per subject: min 3, max 15, mean ≈ 3.64 (histogram saved at [handwriting_audit_outputs/Public_Dataset_samples_per_subject_hist.png](handwriting_audit_outputs/Public_Dataset_samples_per_subject_hist.png))
- Limitations: modest overall size; some subjects contribute many images producing potential within-subject dominance if not grouped correctly (grouped CV was used in training).

BD_Dataset (separate):
- Total images: 36
- Subjects: 36
- Healthy / Parkinson: 15 / 21 (mild imbalance towards PD)
- Images per subject: all 1 (min=max=1) — no repeated measures available.
- Limitations: critical — one-sample-per-subject prevents any within-subject variability estimation. Grouped CV at subject level becomes identical to sample-level CV for BD, so subject-level generalization cannot be validated. This dataset is appropriate only for initial exploratory signals and not for robust predictive claims.

Discussion:
- BD’s tiny per-subject data is the dominant limitation of the study. Public_Dataset is better suited for grouped CV and subject-level generalization, but is still small for robust ML practice.

---

**3. Preprocessing analysis**

Observed pipeline (see `preprocessing.py`): grayscale conversion → denoising (fastNlMeans) → CLAHE contrast enhancement → adaptive Gaussian thresholding (inverted) → morphological open/close → small-component removal → foreground normalization (invert if necessary) → mask + crop to content → resize with padding to 224×224. QC preview images are saved under `preprocessed_images/*/quality_check`.

Assessment:
- Grayscale conversion and CLAHE are appropriate for handwriting where ink contrast varies. Denoising reduces sensor noise.
- Adaptive thresholding plus morphology isolates stroke pixels; small-component removal (MIN_COMPONENT_AREA) can remove speckles but risks discarding legitimate small stroke fragments (thin strokes, light dots) if parameters are not tuned per acquisition.
- Content-cropping with padding + resize-with-padding yields centered inputs for CNNs, which is desirable for transfer learning. However, if cropping behaves differently by class (for instance, PD spirals with tenuous strokes leading to larger crop extents or different centering), there is a risk of introducing class-dependent spatial cues.

Risks and evidence to check (recommended for paper):
- Verify that cropping/padding statistics (crop box sizes, foreground centroid positions) are not class dependent — compute and report class-wise distributions to rule out positional shortcuts.
- Keep and publish QC preview grids (they are in `preprocessed_images/*/quality_check`) so reviewers can visually confirm preprocessing consistency.

Short conclusion: preprocessing is standard and defensible for handwriting, but the pipeline must report and check per-class cropping/foreground statistics to exclude shortcut leakage.

---

**4. Deep learning analysis (ResNet-18 transfer learning)**

Data sources: per-fold prediction CSVs and final CV summaries under `deep_learning_results/*`.

Public_Dataset (aggregated results):
- Accuracy: 0.7690 ± 0.1338
- Precision: 0.7793 ± 0.1740
- Sensitivity (recall): 0.8133 ± 0.2334
- Specificity: 0.6889 ± 0.3393
- F1-score: 0.7670 ± 0.1437
- ROC-AUC: 0.9563 ± 0.0470

BD_Dataset (aggregated results):
- Accuracy: 0.5786 ± 0.1524
- Precision: 0.5762 ± 0.3690
- Sensitivity: 0.55 ± 0.4472
- Specificity: 0.6000 ± 0.3651
- F1-score: 0.5067 ± 0.3394
- ROC-AUC: 0.9067 ± 0.1461

Interpretation and key observations:
- ROC-AUC vs threshold metrics: In both datasets, ROC-AUC remains comparatively high while thresholded metrics (accuracy, sensitivity, specificity) are more modest and have large standard deviations across folds (particularly BD). This pattern indicates the model often ranks PD > healthy correctly (good discriminative signal) but the absolute probabilities and thresholded decisions vary by fold (poor calibration and sensitivity to operating point).
- Fold instability: BD shows extreme fold-to-fold variance in AUC and ECE (see [handwriting_audit_outputs/BD_Dataset/dl_fold_metrics_summary.csv](handwriting_audit_outputs/BD_Dataset/dl_fold_metrics_summary.csv)). Several BD folds report AUC=1.0 and others as low as 0.667 — a sign of overfitting on very small validation sets or class-sample splits that favor the model.
- Overfitting risks: early stopping and LR scheduling are present, but very small validation sets (especially BD) can lead to unstable best-epoch selection and optimistic folds. Models with near-perfect AUC on some folds but poor generalization on others suggest the signal is fragile.

Calibration specifics:
- BD mean ECE ≈ 0.29 (per-fold mean from computed summary) and mean Brier ≈ 0.23 — probabilities are not well calibrated and are unreliable for absolute risk reporting.
- Public mean ECE ≈ 0.22 and mean Brier ≈ 0.15 — better but still substantial; apply temperature scaling or Platt scaling if probabilistic outputs are to be used.

Practical takeaways:
- Report both ranking (AUC) and calibrated decision metrics; present per-fold distributions rather than only mean ± std.
- Avoid over-interpreting perfect or near-perfect fold AUCs; show confusion matrices and per-subject predictions for the most unstable folds (files under `deep_learning_results/*/fold_*_predictions.csv`).

---

**5. Classical ML analysis (handcrafted features)**

Source: `classical_ml_results/BD_Dataset/classical_ml_results.csv` (and analogous Public_Dataset results if present).

Key BD findings (selected):
- SVM: accuracy ≈ 0.8071 ± 0.0696; precision 1.0; sensitivity ≈ 0.67 ± 0.115; ROC-AUC ≈ 0.8667 ± 0.1514
- RandomForest and XGBoost show lower mean performance and higher variance.

Interpretation:
- On BD_Dataset, the SVM with handcrafted features outperforms DL on thresholded accuracy. This is a predictable and scientifically interpretable outcome in very small-data regimes: well-engineered features (HOG, LBP, Hu moments, contour/statistics) can be strong compact biomarkers and require far fewer samples to reach good classification thresholds.
- Handcrafted features often capture domain-specific signals (stroke width, curvature, regularity) that are clinically plausible. When classical ML outperforms or matches DL, it is an argument for emphasizing interpretable features and for treating DL as an augmenting modality rather than a replacement in low-resource settings.

Recommendation:
- For BD-like tiny cohorts, prioritize feature-based analyses in the main thesis text and present DL as a complementary representation-learning approach. Provide ROC overlays and per-fold metrics for both paradigms (files exist under `classical_ml_results/*` and `deep_learning_results/*`).

---

**6. GradCAM (explainability) analysis**

Artifacts inspected: preview images under `model_interpretability_visualizations/*`.

Qualitative summary:
- Many GradCAM previews show activations overlapping inked spiral strokes, especially areas of irregularity or high stroke density — this aligns with clinical expectations that tremor/irregularity localizes to the drawn spiral.
- Some examples show diffuse activation or attention extending into background/padding regions; these cases are less interpretable and may indicate that the model sometimes leverages gross textural or border cues.

Limitations and caution:
- GradCAM is qualitative: it highlights correlated regions, not causal mechanisms. GradCAM resolution (spatial coarseness of late-layer feature maps) limits precise localization of micro-features like pen-pressure or fine tremor.
- Do not claim clinical interpretability from GradCAM alone. Use GradCAM as hypothesis-generation, then test hypotheses quantitatively (occlusion analysis across stroke regions, lesioning test inputs, or controlled perturbations).

---

**7. Failure analysis**

Using fold predictions and per-fold summaries, the main failure modes are:
- False negatives with borderline probabilities (example: `deep_learning_results/BD_Dataset/fold_3_predictions.csv` contains `BPP003` predicted 0 with PD probability ≈ 0.352). Such borderline examples indicate that ranking is imperfect and threshold placement matters.
- Fold collapse/instability: BD shows some folds with AUC=1.0 while others have AUC≈0.667, indicating highly variable generalization depending on which subjects fall into train/test splits.
- Calibration-instability: models are often overconfident or underconfident in certain folds (high ECE in BD fold 4 ~0.505 suggests strong miscalibration in some splits).

Probable causes:
- Tiny sample and single-image subjects (BD) cause high variance and sensitive splits.
- Preprocessing-induced shortcuts: if cropping/padding or foreground removal correlates with label, models can learn spurious cues.
- Class imbalance (mild in BD) can skew thresholds and metrics; class weighting is implemented in training but does not fully compensate in tiny folds.

Mitigations (for reporting and future work):
- Present failure-case panels (QC image + GradCAM + predicted probability) for FPs/FNs. Use available previews and prediction CSVs to assemble such panels.
- Emphasize grouped CV, and for BD explicitly note that grouped CV is equivalent to sample-level CV given one sample per subject.

---

**8. Scientific assessment (brutal but fair)**

Strengths:
- The pipeline is complete and reproducible: preprocessing, DL training, classical ML, and GradCAM outputs are present and saved.
- Presence of Bangladeshi clinical data (BD_Dataset) increases the project's translational relevance.
- Careful inclusion of grouped cross-validation and classical-DL comparisons strengthens scientific defensibility.

Weaknesses / critical limitations:
- Data scarcity is the main limitation. BD has one image per subject, preventing subject-level generalization and rendering many evaluation claims exploratory only.
- Calibration is poor in places (ECE mean ~0.29 BD; ~0.22 Public). Absolute probabilities cannot be used clinically without calibration.
- Fold instability and occasional perfect AUC folds indicate fragile signals and risk overinterpreting results.

Publication/thesis readiness:
- This work is thesis-ready as an exploratory behavioral-biomarker study if the write-up clearly states limitations, includes per-fold result distributions, shows QC/GradCAM examples, and avoids diagnostic claims. For a journal submission, external validation or a larger cohort is recommended.

---

**9. Multimodal readiness and recommendations**

Why handwriting + face is a coherent multimodal approach:
- Handwriting targets fine motor control (tremor, micrographia, irregular stroke), which is a direct motor manifestation of Parkinsonian pathology.
- Facial analysis targets hypomimia (reduced expressivity) and subtle movement changes in the face — a complementary behavioral biomarker.

Fusion recommendations (low-parameter, low-risk):
- Start with late fusion of calibrated probabilities (per-subject aggregation) or low-dimensional embeddings (PCA on DL embeddings) fed into a small meta-learner (logistic regression or small MLP). This minimizes overfitting risk in small datasets.
- Avoid large multimodal transformers or heavy joint training on the available cohort.

Robustness notes:
- Ensure per-subject alignment and grouped CV across modalities to prevent subject overlap leakage.
- Prefer probability-level fusion when embeddings are high dimensional relative to sample count.

---

**10. Final conclusion**

- The current unimodal handwriting system provides scientifically useful exploratory baselines and demonstrates complementary strengths of handcrafted features and deep learning representations.
- Results are exploratory: high ROC-AUC values indicate discriminative signal exists, but thresholded metrics, calibration, and fold stability are insufficient to support any diagnostic claim.
- The main limitation is data quantity and subject-level sampling (BD has 1 image/subject). Improving the per-subject sampling and adding external validation is the highest priority.
- Multimodal fusion (handwriting + facial) is the logical next step: use conservative, modular, low-parameter fusion and strong grouped CV.

---

Appendix — Quick pointers to reviewed artifacts
- Dataset metadata: [preprocessed_images/Public_Dataset/metadata.csv](preprocessed_images/Public_Dataset/metadata.csv), [preprocessed_images/BD_Dataset/metadata.csv](preprocessed_images/BD_Dataset/metadata.csv)
- DL final summaries: [deep_learning_results/Public_Dataset/final_cross_validation_results.csv](deep_learning_results/Public_Dataset/final_cross_validation_results.csv), [deep_learning_results/BD_Dataset/final_cross_validation_results.csv](deep_learning_results/BD_Dataset/final_cross_validation_results.csv)
- DL fold-level metrics (calibration): [handwriting_audit_outputs/BD_Dataset/dl_fold_metrics_summary.csv](handwriting_audit_outputs/BD_Dataset/dl_fold_metrics_summary.csv)
- Classical ML BD summary: [classical_ml_results/BD_Dataset/classical_ml_results.csv](classical_ml_results/BD_Dataset/classical_ml_results.csv)
- Example GradCAM preview: [model_interpretability_visualizations/BD_Dataset/parkinson_BPP019_hand_spiral_01_preview.png](model_interpretability_visualizations/BD_Dataset/parkinson_BPP019_hand_spiral_01_preview.png)
- Example borderline prediction: [deep_learning_results/BD_Dataset/fold_3_predictions.csv](deep_learning_results/BD_Dataset/fold_3_predictions.csv)

If you want, I will now:
- assemble a small figure panel (QC images + GradCAM + predicted probability) for 6 curated cases (FP/FN/uncertain) for thesis figures, or
- run temperature scaling on saved fold logits to produce calibrated probabilities and updated calibration plots (requires re-running small calibration script on existing predictions — no retraining).
