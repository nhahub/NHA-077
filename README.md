# Land Type Classification Using Sentinel-2 Satellite Images

This project implements a land type classification system based on Sentinel-2 satellite imagery using deep learning models. The system classifies land into ten categories—covering agriculture, water, urban zones, forests, and industrial areas—using a range of convolutional neural network architectures with optional ensemble methods for improved accuracy.


## Features

* **Multiple Model Architectures:** VGG16, ResNet50, MobileNetV2, EfficientNet, and custom Sequential networks
* **Hyperparameter Tuning:** Bayesian Optimization and Random Search
* **Production-Ready API:** FastAPI-based REST API with an integrated web interface
* **Desktop Application:** Tkinter-based GUI for offline usage
* **Ensemble Learning:** Support for model ensemble prediction
* **Comprehensive Evaluation:** Metrics, confusion matrices, and visualization tools
* **GPU Acceleration:** Optimized for TensorFlow with CUDA support

## Dataset

This project uses the **EuroSAT RGB** dataset, containing approximately 27,000 labeled images across ten land cover categories.

| Class                | Description                         | Sample Count |
| -------------------- | ----------------------------------- | ------------ |
| AnnualCrop           | Seasonal agricultural fields        | ~3,000       |
| Forest               | Woodland and forested regions       | ~3,000       |
| HerbaceousVegetation | Grasslands and meadows              | ~3,000       |
| Highway              | Major roads and highways            | ~2,500       |
| Industrial           | Industrial zones and structures     | ~2,500       |
| Pasture              | Grazing areas                       | ~2,000       |
| PermanentCrop        | Orchards and vineyard areas         | ~2,500       |
| Residential          | Residential and urban housing areas | ~3,000       |
| River                | Rivers and smaller water bodies     | ~2,500       |
| SeaLake              | Oceans and large lakes              | ~3,000       |

**Dataset Split:** 70% training, 15% validation, 15% testing

## Project Structure

```
DEPI Final Project/
├── API/                           # FastAPI-based REST API
│   ├── Models/                    # Trained model files
│   ├── server.py                  # API server
│   ├── client.py                  # Python client
│   ├── ensemble.py                # Ensemble implementation
│   └── index.html                 # Web interface
│
├── Dataset/                       # Dataset storage
│   ├── EuroSAT_RGB/               # Original dataset
│   └── EuroSAT_RGB_split/         # Train/validation/test split
│
├── Model_Fine_Tuning/             # Hyperparameter tuning
│   ├── Bayesian_Optimization/     
│   └── Random_Search/             
│
├── Ready_Models/                  # Production-ready models and notebooks
│   ├── VGG16_Model.ipynb
│   ├── ResNet_Model.ipynb
│   ├── MobileNet_Model.ipynb
│   ├── EfficientNet_Model.ipynb
│   ├── Sequential_Model.ipynb
│   └── Visualization.ipynb
│
├── Tkinter_App/                   # Desktop application
│   └── app.py
│
├── requirements.txt               # Python dependencies
├── start.sh                       # Deployment script
└── README.md                      # Documentation
```

## Quick Start

### Prerequisites

* Python 3.8 or newer
* NVIDIA GPU with CUDA support (recommended)
* At least 8 GB of RAM

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/nhahub/NHA-077.git
   cd "DEPI Final Project"
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the API server** (downloads models automatically)

   ```bash
   python API/server.py
   ```

### Using the Web API

1. Start the server:

   ```bash
   python API/server.py
   ```
2. Navigate to:

   ```
   http://127.0.0.1:8000
   ```
3. Upload an image to obtain predictions.

### Using the Desktop Application

```bash
python Tkinter_App/app.py
```

### Using the Python Client

```bash
python API/client.py path/to/image.jpg
```

## Model Performance

| Model          | Test Accuracy |
| -------------- | ------------- |
| VGG16          | 97.46%        |
| ResNet50       | 95.28%        |
| Sequential CNN | 95.16%        |
| MobileNetV2    | 94.54%        |
| **Ensemble**   | **98.42%**    |

## Advanced Usage

### Hyperparameter Tuning

* **Random Search:** 160 trials
* **Bayesian Optimization:** 30+ trials per architecture
* **Custom Loss Function:** Categorical Focal Loss (due to using older TensorFlow)

### Training Example

```python
from Ready_Models.VGG16_Model import train_vgg16

params = {
    'dense_units': 256,
    'dropout': 0.67,
    'learning_rate': 0.000235,
    'unfreeze_from': 'block5_conv3'
}

model = train_vgg16(params)
```

### Ensemble Example

```python
from API.ensemble import EnsembleModel

ensemble = EnsembleModel([
    'API/Models/model_vgg16.keras',
    'API/Models/model_resnet50.keras',
    'API/Models/sequential_model.keras'
])

ensemble.load_models()
predictions = ensemble.predict(image_array)
```

## API Endpoints

### REST Endpoints

* `GET /` – Web interface
* `GET /api` – API information
* `GET /models` – List available models
* `GET /labels` – Classification labels
* `POST /predict` – Upload and classify image
* `GET /ensemble-info` – Ensemble configuration

### Example Request

```python
import requests

with open('satellite_image.jpg', 'rb') as f:
    response = requests.post('http://localhost:8000/predict', files={'file': f})
    result = response.json()

print(f"Predicted: {result['predicted_label']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## Technical Details

### TensorFlow GPU Setup
For GPU acceleration, TensorFlow was configured to utilize the system’s NVIDIA GPU.  
The setup followed [this tutorial](https://github.com/entbappy/Setup-NVIDIA-GPU-for-Deep-Learning.git).

### Current GPU Environment
| Component | Details |
|------------|----------|
| **TensorFlow** | 2.10.1 |
| **Keras** | 2.10.0 |
| **NumPy** | 1.26.4 |
| **Scikit-learn** | 1.7.2 |
| **Driver Version** | 581.57 |
| **CUDA Toolkit** | 11.2.0 |
| **cuDNN** | 8.9.7.29 |

The environment was validated by running TensorFlow GPU checks to ensure CUDA and cuDNN were correctly recognized:
```python
import tensorflow as tf
print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))
```

### Model Architectures

1. VGG16 
2. ResNet50
3. MobileNetV2
4. EfficientNet
5. Custom Sequential CNN

### Data Augmentation

* Rotation: ±20°
* Rescaling: 1/255
* Fill mode: nearest
* Target size: 64–224 px

## Visualization Tools

Includes notebooks for generating:

* Training and validation curves
* Confusion matrices
* Per-class analysis
* Misclassification reports
* Model architecture summaries

Run the visualization notebook:

```bash
jupyter notebook Ready_Models/Visualization.ipynb
```

## Model Interpretation

### Key Findings

1. **VGG16** provides the strongest individual performance for this dataset.
2. **Ensemble** methods yield an additional **1%** performance gain.
3. Transfer learning significantly reduces training time.
4. Focal Loss improves handling of class imbalance.


## Acknowledgments

* EuroSAT dataset by Helber et al.
* TensorFlow development team
* FastAPI framework
* Project contributors

## Support

For help or inquiries:

* Open an issue on GitHub
* Review the API documentation at `/docs`
* Explore example notebooks in `Ready_Models/`
