# Research Report: Handwriting-Based Parkinson's Screening

## Study Purpose

This project studies whether handwriting spirals can be used as a behavioral marker for Parkinson's disease. The current work is a unimodal handwriting study. It uses a public Kaggle handwriting dataset as the present benchmark, while your Bangladeshi Parkinson's and healthy data can later replace or extend it. That makes the current work useful in two ways. First, it gives you a working and testable baseline. Second, it gives you a clear structure that can later support a clinically grounded local dataset and, eventually, a multimodal study with facial expression data.

The main goal of the current cycle was not only to train a model, but to build a clean and reproducible research pipeline. The pipeline had to start from raw spiral images, produce cleaned data, create handcrafted features, train a deep learning classifier, compare classical machine learning baselines, and generate explanations with Grad-CAM. That full cycle has now been completed.

## Methodology

The first stage was preprocessing. The raw spiral images were renamed into a consistent patient-oriented pattern. Healthy samples now use the `HP` prefix and Parkinson samples use the `PP` prefix. The preprocessing code then reads the raw image folders, checks whether each image is valid, converts it to grayscale, removes noise, enhances contrast, applies adaptive thresholding, removes tiny connected components, crops to the handwriting region, and resizes the result to a fixed size. The output of this stage is a set of standardized images and a metadata CSV file. The metadata file keeps the patient ID, class label, and file path for each image.

The second stage was handcrafted feature extraction. From each preprocessed image, the script extracted texture, shape, and stroke features that are known to describe handwriting behavior. These include HOG, LBP, stroke density, contour statistics, skeleton statistics, Hu moments, and image entropy. This step produced a numerical feature table plus NumPy arrays for labels and patient IDs. These files are the basis for the classical machine learning baseline.

The third stage was deep learning training. The current deep learning model is ResNet-18, trained with patient-level stratified cross-validation. The model uses grayscale handwriting images as input. It includes data augmentation during training so the model sees small variations in rotation, translation, and scale. This matters because handwriting changes naturally from person to person and from sample to sample. The training process uses early stopping, class weighting, and fold-based evaluation. The output of this stage is a set of saved checkpoints and fold-wise prediction files.

The fourth stage was classical machine learning. The handcrafted feature table was used to train SVM, Random Forest, and XGBoost. PCA was applied to reduce dimensionality, and grouped cross-validation was used so that a patient's samples never leak into both training and testing. This gave you a traditional baseline that can be compared fairly with the deep learning model.

The fifth stage was interpretability. Grad-CAM was run on the trained ResNet-18 checkpoints to show where the model was paying attention when it predicted healthy or Parkinsonian handwriting. This is important for biomedical research because it shows whether the network is focusing on real stroke regions instead of irrelevant background patterns.

## Results

The preprocessing stage completed successfully on 102 images, with 51 healthy and 51 Parkinson samples. No images were skipped. The output included cleaned grayscale images, binary masks, quality check images, and a metadata CSV file. This means the input data was successfully standardized and the rest of the pipeline had a stable base.

The feature extraction stage also completed successfully. It produced a feature matrix of shape 102 by 6118. That means each image was converted into a high-dimensional numerical description that classical machine learning could use. The presence of the patient ID column also means the dataset can still be split safely at the patient level.

The deep learning stage completed five-fold cross-validation with ResNet-18. The final average results were 0.7862 accuracy, 0.7663 precision, 0.8900 sensitivity, 0.6444 specificity, 0.8097 F1-score, and 0.9515 ROC-AUC. In simple English, the model was very good at detecting Parkinson's cases, but it was weaker when trying to avoid false alarms on healthy cases. That is visible in the lower specificity compared with the sensitivity. This kind of pattern is common in medical screening systems where the model is tuned to be sensitive to disease.

The classical machine learning baseline also completed successfully. SVM achieved 0.7373 accuracy and 0.8899 ROC-AUC. Random Forest achieved 0.7538 accuracy and 0.8288 ROC-AUC. XGBoost achieved 0.7496 accuracy and 0.7860 ROC-AUC. In practical terms, the classical models were useful baselines, but they did not clearly beat the deep learning model. The ResNet-18 model gave the strongest overall screening performance in this run, especially in ROC-AUC and sensitivity.

The interpretability stage produced Grad-CAM outputs for both Parkinson and healthy examples. These visualizations are now available as image files and can be used in a thesis figure or paper appendix. They help show that the model is responding to handwriting structure rather than random image noise.

## Discussion

This run shows that your unimodal handwriting pipeline is already scientifically meaningful. It is not just a demo project anymore. It now has a full research workflow: cleaned inputs, patient-aware metadata, feature engineering, deep learning, classical baselines, and explainability. That is a solid computer science biomedical research structure.

In relation to your larger vision, this is a strong first modality. Handwriting is a clinically coherent signal for Parkinson's disease because it reflects motor control, tremor, rigidity, and movement irregularity. That means the modality is not arbitrary. It is biologically linked to the disease. This is important because later, when you add facial expression data, the two modalities will support the same disease process from different angles. Handwriting shows fine motor impairment, and facial expression shows reduced expressiveness and rigidity. Together, they can support a much stronger multimodal screening narrative.

In the current state, your work matches the vision in several important ways. It already uses real patient-oriented naming and patient-aware splitting. It already treats the problem as a healthcare task rather than a generic ML problem. It already includes interpretability, which is a major requirement in biomedical AI. It also already supports a future path toward a Bangladesh-specific dataset, which will make the work more locally relevant and more clinically valuable than a simple public benchmark study.

At the same time, this is still unimodal, and it is still based on a public Kaggle dataset for now. That means the current results should be treated as a benchmark rather than the final clinical system. The next step is to replace or extend the current data with your Bangladeshi Parkinson's and healthy subjects. That will make the study more population-specific and more aligned with your real research vision.

## Limitations

The current dataset is small. That limits how far the models can generalize. A small dataset also increases the chance that the model learns accidental patterns instead of disease patterns, even when cross-validation is done carefully.

The current study is unimodal. It only uses handwriting. That means it cannot yet capture the full multimodal healthcare story that you described. The facial expression component still needs to be added later.

The current data also comes from a public source, not from the Bangladeshi population you ultimately want to study. That means the present results are useful as a benchmark, but they are not yet the final population-specific result.

The current training pipeline is strong, but it still reflects the limits of the dataset size. Very deep or very complex models would not be justified here. The present choice of ResNet-18 is reasonable, but future work should still be tested carefully to avoid overfitting.

## Future Scope

The most important next step is to bring in the Bangladeshi Parkinson's and healthy handwriting data. Once that data is available, the current pipeline can be reused with only small changes. This will make the work much more clinically grounded.

The next major step after that is multimodal fusion. Once handwriting and facial expression data are both available, you can build separate unimodal models first and then combine them. That will let you show whether fusion actually improves performance. This is the scientifically proper way to build a multimodal healthcare AI system.

A later improvement would be to collect richer metadata such as age, sex, disease duration, medication status, and severity stage. That will make the analysis more robust and will help reviewers see that the study is clinically designed rather than purely algorithmic.

Another useful future direction is external validation. If the model works on your Bangladeshi data after being benchmarked on the public Kaggle data, that will make the research much stronger.

A final future direction is lightweight deployment. Since your vision is low-resource healthcare support, the final system should remain efficient enough to run on modest hardware or mobile-friendly setups.

## Current Position in the Research Journey

At this point, you are in a strong unimodal baseline stage. That is not a weak place to be. It is actually the correct place to be before moving into the more complex multimodal phase. You now have a complete handwriting pipeline, working code, usable outputs, and a clear methodological story.

So the honest answer is this: you are not yet at the final multimodal research goal, but you are definitely past the student-project stage. You now have the foundation for a real biomedical AI paper. The next leap is not more random model experiments. The next leap is adding your own Bangladeshi data and then extending this same clean pipeline into a multimodal study.

## Conclusion

This full cycle produced a working and research-oriented handwriting screening pipeline for Parkinson's disease. It cleaned the raw images, generated metadata, extracted handcrafted features, trained a deep learning model, tested classical machine learning baselines, and created Grad-CAM explanations. The results show that the system can already separate healthy and Parkinsonian handwriting reasonably well, especially when measured with ROC-AUC and sensitivity.

More importantly, the work now has a clear direction. It is no longer just a model on a dataset. It is the first half of a clinically grounded behavioral biomarker system. The unimodal handwriting branch is now ready to be carried forward into your Bangladeshi data and, later, into the multimodal handwriting-plus-face framework you described.
