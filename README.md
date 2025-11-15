##### Note:
Over the past month, the model was developed, trained, and evaluated on this [GitHub Repo](https://github.com/BelalAdelDev/DEPI-Final-Project).
***

# Land Type Classification Using Sentinel-2 Satellite Images

This project implements a land type classification system leveraging Sentinel-2 satellite imagery. The goal is to classify land into categories such as agriculture, water, urban areas, desert, roads, and trees using deep learning techniques, primarily convolutional neural networks (CNNs).

---

## Project Overview

The project uses satellite image datasets (EuroSAT RGB), preprocessing and data augmentation techniques, a CNN-based classification model, and multiple machine learning tools for training and evaluation. It focuses on building a robust system capable of accurately predicting land cover types from satellite images.

Major components include:

- Data Collection and Preprocessing  
- Data Splitting into Training, Validation, and Test Sets  
- CNN Model Architecture Design and Training  
- Performance Evaluation with Classification Metrics  
- Model Checkpointing and Hyperparameter Tuning  
- Visualization of Training Progress and Results  

---

## Dataset

The project uses the EuroSAT RGB dataset containing images across 10 land classes:

- AnnualCrop
- Forest
- HerbaceousVegetation
- Highway
- Industrial
- Pasture
- PermanentCrop
- Residential
- River
- SeaLake

Data is split roughly into 70% training, 15% validation, and 15% test sets. Images are resized to \(64 \times 64\) pixels for model input.

---

## TensorFlow GPU Setup
For GPU acceleration, TensorFlow was configured to utilize the system’s NVIDIA GPU.  
The setup followed [this tutorial](https://github.com/entbappy/Setup-NVIDIA-GPU-for-Deep-Learning.git).

### My Current GPU Environment
| Component | Details |
|------------|----------|
| **TensorFlow** | 2.10.1 |
| **Keras** | 2.10.0 |
| **NumPy** | 1.26.4 |
| **Scikit-learn** | 1.7.2 |
| **Driver Version** | 581.57 |
| **CUDA Toolkit** | 11.2.0 (cuda_11.2.0_460.89_win10) |
| **cuDNN** | 8.9.7.29 (cudnn-windows-x86_64-8.9.7.29) |

The environment was validated by running TensorFlow GPU checks to ensure CUDA and cuDNN were correctly recognized:
```python
import tensorflow as tf
print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))
```

---

## Features

- **Data Augmentation:** Rescaling, rotation, and fill modes applied during training to improve model robustness.
- **CNN Architecture:** Configurable layers with Conv2D, MaxPooling, BatchNormalization, Global Average Pooling, Dense, and Dropout layers.
- **Loss Function:** Custom categorical focal loss to address class imbalance.
- **Hyperparameter Tuning:** Random search over convolutional layers, filters, dropout rates, optimizers (Adam, RMSprop, SGD), learning rates, and batch normalization.
- **GPU Support:** TensorFlow GPU acceleration and VRAM management.
- **Training Callbacks:** Early stopping, learning rate reduction, and model checkpointing.
- **Comprehensive Metrics:** Accuracy, precision, recall, F1-score, confusion matrix, and classification reports.

---

## Installation

The project requires Python 3.8+ and the following major packages:

- TensorFlow 2.x (with GPU support recommended)
- NumPy
- OpenCV (cv2)
- Matplotlib
- Seaborn
- Scikit-learn

Install dependencies using:

```
pip install tensorflow numpy opencv-python matplotlib seaborn scikit-learn
```

---

## Usage

1. **Prepare Data:**  
   Organize the EuroSAT RGB dataset as per the folder structure. Use the provided `splitdata` function to split data into train/val/test folders.

2. **Data Augmentation and Loading:**  
   Use the TensorFlow ImageDataGenerator to load and augment images during training.

3. **Model Training:**  
   Configure model parameters (filters, layers, dropout, etc.) and start training. Use callbacks for saving best weights and early stopping.

4. **Evaluation:**  
   Evaluate the model on test set and visualize the confusion matrix and classification report.

5. **Prediction:**  
   Perform predictions on random or new images and visualize results with predicted and actual labels.

---

## Results

- Achieved test accuracy of approximately **96%** currently  
- Detailed classification report and confusion matrix visualization available  
- Training plots showing accuracy and loss over epochs  

---

## Folder Structure

```
├── Dataset/
│   ├── EuroSATRGB/             # Original RGB images dataset
│   ├── EuroSATRGBsplit/        # Split dataset for train/val/test
├── ModelFineTuning/            # Scripts and notebooks for fine-tuning
│   ├── Random_Search/          # Hyperparameter tuning using Random search
│   │    ├── Checkpoints/       # Saved model weights and checkpoints
│   │    └── RandomSearchModel.ipynb    # run file for Random search
│   └── Bayesian_Optimization/  # Hyperparameter tuning using Bayesian optimization
│        ├── Checkpoints/       # Saved model weights and checkpoints
│        └── RandomSearchModel.ipynb    # run file for Bayesian optimization
├── model.h5                    # Trained model file
├── Model.ipynb                 # Main notebook for experimentation
└── README.md                   # Project documentation
```
