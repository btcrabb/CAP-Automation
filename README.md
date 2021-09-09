CAP Automated View and Phase Selection Tool
==============================
Author: Brendan Crabb
Email: brendan.crabb@hsc.utah.edu

Towards fully automated cardiac statistical modeling: a deep-learning based MRI view and frame selection tool

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
