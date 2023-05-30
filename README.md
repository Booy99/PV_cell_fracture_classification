# PV_cell_fracture_classification
Computer Vision on The Edge: Micro-Fracture Classification in Photovoltaic Cells in Constrained Environments

This repository consists of datasets and Jupyter Notebooks that accompany the MSc. Thesis in Software Design at the IT University of Copenhagen, completed by Booy Faassen on 1st of June, 2023.

**Splitting_data.ipynb:** Creates eight different datasets based on the original 2,624 images.

**Augmentation_Pytorch.ipynb:** Augments selected datasets either for feeding directly into a model for training or to save images in a file.

**Augmentation_Tensorflow.ipynb:** Augments images for feeding directly into a model for training.

**EDA.ipynb:** Exploratory data analysis for the collected data.

**Results_EDA.ipynb:** Exploratory data analysis for the results from training 72 models on the HPC.

**MobileNetV2_training.ipynb:** Building and training MobileNetV2 models.

**InceptionV3_training.ipynb:** Building and training InceptionV3 models.

**EfficientNetB0_training.ipynb:** Building and training EfficientNet-B0 models.

**Evaluate_models.ipynb:** Run inference on all models.

**Quantization-ipynb:** Quantize all models using different quantization techniques and evaluate model performance after quantization.
