# EuroSAT Image Classifier API

A FastAPI-based REST API for classifying satellite images using trained deep learning models (VGG16 or Sequential CNN) on the EuroSAT RGB dataset.

## Features

- REST API for satellite image classification
- Automatic detection of all `.keras` and `.pkl` model files
- Support for multiple model formats (Keras and Pickle)
- Returns probabilities for all classes
- Includes Python client for easy integration
- Lists all available models via API endpoint

## Prerequisites

- Python 3.8+
- TensorFlow/Keras
- FastAPI
- Pillow (PIL)
- uvicorn
- requests (for client)

## Installation

```bash
pip install fastapi uvicorn tensorflow pillow requests numpy
```

## Project Structure

```
DEPI Final Project/
├── API/
│   ├── Models/
│   │   ├── model_vgg16.keras     # Keras models
│   │   ├── sequential_model.keras
│   │   └── any_model.pkl         # Pickle models
│   ├── server.py                 # FastAPI server
│   └── client.py                 # Python client
└── Dataset/
    └── EuroSAT_RGB/              # Reference class folders
```

## Running the Server

### Basic Usage

```bash
python API/server.py
```

The server will start on `http://127.0.0.1:8000` by default.

### Custom Configuration

```bash
# Custom host and port
HOST=0.0.0.0 PORT=5000 python API/server.py

# Custom model path
MODEL_PATH=/path/to/your/model.keras python API/server.py
```

### Environment Variables

- `HOST`: Server host address (default: `127.0.0.1`)
- `PORT`: Server port (default: `8000`)
- `MODEL_PATH`: Path to specific model file (optional, otherwise uses first model found in `API/Models/`)

**Note:** The API automatically scans `API/Models/` directory for `.keras` and `.pkl` files. If multiple models exist, it loads the first one alphabetically unless `MODEL_PATH` is specified.

## API Endpoints

### 1. Health Check

**GET** `/`

Check if the API is running and see which model is loaded.

**Response:**
```json
{
  "status": "ok",
  "message": "EuroSAT classifier API",
  "labels_available": 10,
  "model_loaded": "model_vgg16.keras"
}
```

### 2. List Available Models

**GET** `/models`

Get all available models in the Models directory.

**Response:**
```json
{
  "available_models": [
    "model_vgg16.keras",
    "sequential_model.keras",
    "custom_model.pkl"
  ],
  "current_model": "model_vgg16.keras",
  "models_directory": "C:/Users/%username%/Desktop/DEPI Final Project/API/Models"
}
```

### 3. Get Labels

**GET** `/labels`

Get all available classification labels.

**Response:**
```json
{
  "labels": [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake"
  ]
}
```

### 4. Predict Image

**POST** `/predict`

Classify an uploaded image.

**Request:**
- Content-Type: `multipart/form-data`
- Body: Image file (JPEG, PNG, etc.)

**Response:**
```json
{
  "predicted_index": 3,
  "predicted_label": "Highway",
  "probabilities": [
    {
      "label": "AnnualCrop",
      "probability": 0.05
    },
    {
      "label": "Forest",
      "probability": 0.02
    },
    {
      "label": "Highway",
      "probability": 0.87
    }
  ]
}
```

## Using the Python Client

### Basic Usage

```bash
python API/client.py path/to/image.jpg
```

### Custom Server URL

```bash
python API/client.py path/to/image.jpg --server http://192.168.1.100:8000
```

### Example Output

```json
{
  "predicted_index": 7,
  "predicted_label": "Residential",
  "probabilities": [
    {
      "label": "AnnualCrop",
      "probability": 0.001234
    },
    {
      "label": "Residential",
      "probability": 0.923456
    }
  ]
}
```

## Using cURL

### Predict an image

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/image.jpg"
```

### Get labels

```bash
curl -X GET "http://127.0.0.1:8000/labels"
```

### List available models

```bash
curl -X GET "http://127.0.0.1:8000/models"
```

## Using Python Requests

```python
import requests

# Check which model is loaded
response = requests.get("http://127.0.0.1:8000/")
print(f"Current model: {response.json()['model_loaded']}")

# List all available models
response = requests.get("http://127.0.0.1:8000/models")
print(f"Available models: {response.json()['available_models']}")

# Get labels
response = requests.get("http://127.0.0.1:8000/labels")
labels = response.json()
print(labels)

# Predict image
with open("satellite_image.jpg", "rb") as f:
    files = {"file": ("image.jpg", f, "image/jpeg")}
    response = requests.post("http://127.0.0.1:8000/predict", files=files)
    result = response.json()
    print(f"Predicted: {result['predicted_label']}")
    print(f"Confidence: {result['probabilities'][result['predicted_index']]['probability']:.2%}")
```

## Interactive API Documentation

FastAPI automatically generates interactive API documentation:

- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

You can test the API directly from your browser using these interfaces.

## Model Requirements

The API supports two model formats:

### Keras Models (.keras)
- Input size: 96x96 RGB images
- Output: 10 classes (EuroSAT categories)
- Preprocessing: Normalized to [0, 1] range
- Saved using `model.save('model_name.keras')`

### Pickle Models (.pkl)
- Must be a scikit-learn compatible model or any pickled model with `.predict()` method
- Should accept preprocessed image arrays (96x96x3, normalized)
- Output: 10 class probabilities or predictions
- Saved using `pickle.dump(model, file)`

**Important Notes:**
- Place all model files in `API/Models/` directory
- The API automatically detects all `.keras` and `.pkl` files
- If multiple models exist, the first one alphabetically is loaded (unless `MODEL_PATH` is set)
- Class names are hardcoded and must match training order: AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake

## Troubleshooting

### Model Not Found Error

If you see `RuntimeError: No model files found`:

1. Create the Models directory: `mkdir API/Models`
2. Place your model files in `API/Models/` directory
3. Supported formats: `.keras` or `.pkl`
4. Or set the `MODEL_PATH` environment variable to point to a specific model file

**Examples:**
```bash
# Place model in Models folder
cp /path/to/your/model.keras API/Models/

# Or use MODEL_PATH for specific model
MODEL_PATH=/path/to/specific/model.pkl python API/server.py

# Check which models are available
curl http://127.0.0.1:8000/models
```

### Wrong Model Loaded

If you have multiple models and want to use a specific one:

```bash
# Use MODEL_PATH to specify exact model
MODEL_PATH=API/Models/model_vgg16.keras python API/server.py
```

Otherwise, the API loads the first model alphabetically from the Models directory.

### Port Already in Use

If port 8000 is busy:

```bash
PORT=8001 python API/server.py
```

### Image Upload Errors

- Ensure the file is a valid image (JPEG, PNG, etc.)
- Check file size (very large images may timeout)
- Verify the image can be opened by PIL/Pillow

## Development Mode

The server runs with auto-reload enabled by default, which is useful for development:

```bash
python API/server.py  # Changes to server.py will trigger reload
```

For production, use uvicorn directly:

```bash
uvicorn API.server:app --host 0.0.0.0 --port 8000 --workers 4
```

## License

[Your License Here]

## Contributing

[Contributing Guidelines]

## Support

For issues or questions, please open an issue on the project repository.