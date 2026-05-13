# Research Report: Handwriting-Based Parkinson's Screening

## Study Purpose

This project studies whether handwriting spirals can be used as a behavioral marker for Parkinson's disease. The current work is a unimodal handwriting study run on two datasets with fully separated pipelines: `Public_Dataset` and `BD_Dataset`.

The pipeline was implemented so each dataset has its own preprocessing metadata, feature files, classical ML outputs, deep learning checkpoints/results, and Grad-CAM explanations. This avoids output mixing and allows direct cross-dataset comparison.

The main goal of the current cycle was not only to train a model, but to build a clean and reproducible research pipeline. The pipeline starts from raw spiral images, produces cleaned data, creates handcrafted features, trains a deep learning classifier, compares classical machine learning baselines, and generates explanations with Grad-CAM. That full cycle has now been completed separately for both datasets.

## Methodology

The first stage was preprocessing. The raw spiral images were renamed into a consistent patient-oriented pattern. Healthy samples now use the `HP` prefix and Parkinson samples use the `PP` prefix. The preprocessing code then reads the raw image folders, checks whether each image is valid, converts it to grayscale, removes noise, enhances contrast, applies adaptive thresholding, removes tiny connected components, crops to the handwriting region, and resizes the result to a fixed size. The output of this stage is a set of standardized images and a metadata CSV file. The metadata file keeps the patient ID, class label, and file path for each image.

The second stage was handcrafted feature extraction. From each preprocessed image, the script extracted texture, shape, and stroke features that are known to describe handwriting behavior. These include HOG, LBP, stroke density, contour statistics, skeleton statistics, Hu moments, and image entropy. This step produced a numerical feature table plus NumPy arrays for labels and patient IDs. These files are the basis for the classical machine learning baseline.

The third stage was deep learning training. The current deep learning model is ResNet-18, trained with patient-level stratified cross-validation. The model uses grayscale handwriting images as input. It includes data augmentation during training so the model sees small variations in rotation, translation, and scale. This matters because handwriting changes naturally from person to person and from sample to sample. The training process uses early stopping, class weighting, and fold-based evaluation. The output of this stage is a set of saved checkpoints and fold-wise prediction files.

The fourth stage was classical machine learning. The handcrafted feature table was used to train SVM, Random Forest, and XGBoost. PCA was applied to reduce dimensionality, and grouped cross-validation was used so that a patient's samples never leak into both training and testing. This gave you a traditional baseline that can be compared fairly with the deep learning model.

The fifth stage was interpretability. Grad-CAM was run on the trained ResNet-18 checkpoints to show where the model was paying attention when it predicted healthy or Parkinsonian handwriting. This is important for biomedical research because it shows whether the network is focusing on real stroke regions instead of irrelevant background patterns.

## Results

The preprocessing stage completed successfully for both datasets:

- `Public_Dataset`: 102 images (51 healthy, 51 Parkinson), 0 skipped
- `BD_Dataset`: 36 images (15 healthy, 21 Parkinson), 0 skipped

Each dataset produced its own cleaned grayscale images, binary masks, quality check images, and `metadata.csv` inside `preprocessed_images/<dataset>/`.

The feature extraction stage also completed successfully for both datasets. Feature matrices were:

- `Public_Dataset`: 102 x 6118
- `BD_Dataset`: 36 x 6118

Patient IDs were preserved, allowing grouped patient-level splitting in all downstream evaluation.

The deep learning stage completed five-fold cross-validation with ResNet-18 for each dataset.

`Public_Dataset` deep learning means:

- Accuracy: 0.7690
- Precision: 0.7793
- Sensitivity: 0.8133
- Specificity: 0.6889
- F1-score: 0.7670
- ROC-AUC: 0.9563

`BD_Dataset` deep learning means:

- Accuracy: 0.5786
- Precision: 0.5762
- Sensitivity: 0.5500
- Specificity: 0.6000
- F1-score: 0.5067
- ROC-AUC: 0.9067

In practical terms, deep learning remained strong on ranking/discrimination (ROC-AUC) for both datasets, with stronger balanced classification on the public dataset than on BD.

The classical machine learning baseline also completed successfully for both datasets.

Best classical model by ROC-AUC in both datasets was SVM:

- `Public_Dataset` SVM: Accuracy 0.7373, ROC-AUC 0.8899
- `BD_Dataset` SVM: Accuracy 0.8071, ROC-AUC 0.8667

Comparison summary:

- On `Public_Dataset`, deep learning outperformed classical SVM in both accuracy and ROC-AUC.
- On `BD_Dataset`, classical SVM achieved higher accuracy and specificity than deep learning, while deep learning kept slightly higher ROC-AUC.

The interpretability stage produced Grad-CAM outputs for both Parkinson and healthy examples in each dataset-specific output folder:

- `model_interpretability_visualizations/Public_Dataset/`
- `model_interpretability_visualizations/BD_Dataset/`

These visualizations support qualitative checking that the model responds to handwriting structure rather than irrelevant image background.

## Discussion

This run shows that your unimodal handwriting pipeline is already scientifically meaningful. It is not just a demo project anymore. It now has a full research workflow: cleaned inputs, patient-aware metadata, feature engineering, deep learning, classical baselines, and explainability. That is a solid computer science biomedical research structure.

In relation to your larger vision, this is a strong first modality. Handwriting is a clinically coherent signal for Parkinson's disease because it reflects motor control, tremor, rigidity, and movement irregularity. That means the modality is not arbitrary. It is biologically linked to the disease. This is important because later, when you add facial expression data, the two modalities will support the same disease process from different angles. Handwriting shows fine motor impairment, and facial expression shows reduced expressiveness and rigidity. Together, they can support a much stronger multimodal screening narrative.

In the current state, your work matches the vision in several important ways. It already uses real patient-oriented naming and patient-aware splitting. It already treats the problem as a healthcare task rather than a generic ML problem. It already includes interpretability, which is a major requirement in biomedical AI. It also already supports a future path toward a Bangladesh-specific dataset, which will make the work more locally relevant and more clinically valuable than a simple public benchmark study.

At the same time, this is still unimodal, and the BD dataset size is still relatively small. That means the current results should be treated as a strong benchmark stage rather than a final clinical system. The next step is to extend the dataset with more Bangladeshi Parkinson's and healthy subjects to improve robustness and external validity.

## Limitations

The current dataset is small. That limits how far the models can generalize. A small dataset also increases the chance that the model learns accidental patterns instead of disease patterns, even when cross-validation is done carefully.

The current study is unimodal. It only uses handwriting. That means it cannot yet capture the full multimodal healthcare story that you described. The facial expression component still needs to be added later.

The current data combines a public benchmark dataset with a BD dataset, but the BD cohort is still limited. This means the present results are useful and directionally strong, but they are not yet the final population-specific result.

The current training pipeline is strong, but it still reflects the limits of the dataset size. Very deep or very complex models would not be justified here. The present choice of ResNet-18 is reasonable, but future work should still be tested carefully to avoid overfitting.

## Future Scope

The most important next step is to bring in the Bangladeshi Parkinson's and healthy handwriting data. Once that data is available, the current pipeline can be reused with only small changes. This will make the work much more clinically grounded.

The next major step after that is multimodal fusion. Once handwriting and facial expression data are both available, you can build separate unimodal models first and then combine them. That will let you show whether fusion actually improves performance. This is the scientifically proper way to build a multimodal healthcare AI system.

A later improvement would be to collect richer metadata such as age, sex, disease duration, medication status, and severity stage. That will make the analysis more robust and will help reviewers see that the study is clinically designed rather than purely algorithmic.

Another useful future direction is external validation. If the model works on your Bangladeshi data after being benchmarked on the public Kaggle data, that will make the research much stronger.

A final future direction is lightweight deployment. Since your vision is low-resource healthcare support, the final system should remain efficient enough to run on modest hardware or mobile-friendly setups.

## Current Position in the Research Journey

At this point, you are in a strong unimodal baseline stage with an added cross-dataset validation view. You now have a complete handwriting pipeline, working code, usable outputs, dataset-tagged artifacts, and a clear methodological story.

So the honest answer is this: you are not yet at the final multimodal research goal, but you are definitely past the student-project stage. You now have the foundation for a real biomedical AI paper. The next leap is collecting more BD samples and extending this same clean, dataset-separated pipeline into a multimodal study.

## Conclusion

This full cycle produced a working and research-oriented handwriting screening pipeline for Parkinson's disease. It cleaned raw images, generated metadata, extracted handcrafted features, trained deep learning models, tested classical machine learning baselines, and created Grad-CAM explanations for both `Public_Dataset` and `BD_Dataset` separately. The system now supports clean cross-dataset performance comparison.

More importantly, the work now has a clear direction. It is no longer just a model on a single dataset. It is the first half of a clinically grounded behavioral biomarker system with dataset-aware reproducibility. The unimodal handwriting branch is now ready to be expanded with larger BD data and, later, into the multimodal handwriting-plus-face framework.
