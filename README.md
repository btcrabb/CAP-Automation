Towards fully automated cardiac statistical modeling: a deep-learning based MRI view and frame selection tool
==============================
Author: Brendan Crabb

Email: brendan.crabb@hsc.utah.edu


Project Organization
------------

    ├── data
    │   ├── processed      <- Data sorted according to MRI view.
    │   └── raw            <- The original, immutable data (dicom files).
    |
    ├── models             <- Trained and serialized models (VGG-19, ResNet50, and Xception)
    │
    ├── notebooks          <- Jupyter notebooks for performing view classification and phase prediction
    │
    ├── reports            <- Generated analysis, saved .csv with view predictions for each series
    │
    └── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
                              generated with `pip freeze > requirements.txt`
                              
--------

Statistical atlases of cardiac shape and function have enabled the quantification of patient-specific heart characteristics against reference populations for a variety of conditions; however, the manual input and time required to create models has limited the clinical translation of this tool. Although deep learning has enabled ventricular segmentation and landmark localization, manual cardiac view and phase identification are still required, preventing end-to-end automation and processing times consistent with a clinical workflow. To address this barrier, we developed a fully-automated method for cardiac view and end-systolic (ES) phase selection. 

To perform automated view identification and selection, use the jupyter notebook "CAP View Prediction.ipynb". This notebook utilizes three trained neural networks (VGG-19, ResNet50, and Xception) to perform view classification at the frame and series level. The algorithms were trained on a multi-institutional dataset of labeled T1-weighted MR images from 1,610 series of 61 patients with tetralogy of Fallot. For additional information and access to the original data, please email the authors. 

In the intial analysis, the network VGG-19 achieved the best performance for MRI view classification with a weighted average ROC AUC of 0.998 and an F1-score of 0.97 on the test dataset. ResNet50 and Xception performed similarly with ROC AUCs of 0.996 and 0.995 and F1-scores of 0.98 and 0.95, respectively. For ES phase selection, the ResNet50-LSTM network had the best performance with an average absolute frame difference (aaFD) of 1.36 ± 1.12 frames. This score is comparable to the inter-observer variation between the two manual annotators in this study (aaFD 1.39 ± 1.35; p value = 0.89). Notably, model inference time was less than 0.36 seconds per series on average when executed on a GPU. 

### Precision, Recall, and F1 scores for each model evaluated on the original testing dataset

![f1](https://github.com/btcrabb/CAP-Automation/blob/master/reports/figures/f1_scores.png)

### Correlation matrix between predicted and ground truth view annotations on the original testing dataset

![corr](https://github.com/btcrabb/CAP-Automation/blob/master/reports/figures/correlation_matrix.png)

### Attention, saliency, and GradCAM++ maps for the VGG19 model
![att](https://github.com/btcrabb/CAP-Automation/blob/master/reports/figures/attention_maps.png)
