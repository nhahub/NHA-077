import os
import pickle
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
import uvicorn
from tensorflow.keras.models import load_model #type: ignore
import gdown #type: ignore
try:
    from .ensemble import EnsembleModel
except ImportError:
    from ensemble import EnsembleModel

# Configuration
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "Models"
CLASS_NAMES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
]

# Global state
model = None
model_path = None
input_size = None


def find_models():
    """Find all .keras and .pkl files in Models directory."""
    if not MODELS_DIR.exists():
        return []
    return sorted(MODELS_DIR.glob("*.keras")) + sorted(MODELS_DIR.glob("*.pkl"))


def load_model_file(path: Path):
    """Load model from .keras or .pkl file."""
    if not path.exists():
        raise FileNotFoundError(f"Model file does not exist: {path}")
    
    if path.suffix == ".keras":
        return load_model(str(path))
    elif path.suffix == ".pkl":
        with open(path, "rb") as f:
            return pickle.load(f)
    raise ValueError(f"Unsupported format: {path.suffix}")


def get_input_size(model):
    """Extract model's expected input size."""
    try:
        if hasattr(model, 'input_shape') and model.input_shape[1]:
            return model.input_shape[1]
    except:
        pass
    return 224


def preprocess_image(img: Image.Image, size: int):
    """Preprocess image for model prediction."""
    img = img.convert("RGB").resize((size, size))
    arr = np.asarray(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


def download_model_from_gdrive(file_id, dest):
    """Download model from Google Drive"""
    
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading from: {url}")
    print(f"Destination: {dest}")
    
    try:
        output = gdown.download(url, str(dest), quiet=False, fuzzy=True)
        
        if output is None:
            raise RuntimeError(f"gdown.download returned None for {dest.name}")
        
        if not dest.exists():
            raise RuntimeError(f"File was not created at {dest}")
        
        file_size = dest.stat().st_size
        print(f"Successfully downloaded {dest.name} ({file_size} bytes)")
        
        if file_size < 1000:
            print(f"WARNING: Downloaded file is very small ({file_size} bytes)")
            print(f"Content preview: {dest.read_text()[:200]}")
        
        return True
        
    except Exception as e:
        print(f"ERROR downloading {dest.name}: {e}")
        if dest.exists():
            dest.unlink()  # Remove partial/corrupted file
        return False


MODEL_FILES = [
    (MODELS_DIR / "model_vgg16.keras", "1k6qymM6gBIBuLBLhTx3ufRXjRewXu82v"),
    (MODELS_DIR / "sequential_model.keras", "16db17DZc4dXDmot4P2rJt5ObswKacn6-"),
    (MODELS_DIR / "model_resnet50.keras", "1-9kGrd5behTpTFA9QYmREa5Pf5Phr9207"),
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, model_path, input_size

    ignore_local = True
    if ignore_local:
        print("IGNORE_LOCAL_COPY flag is set - will re-download models")

    # Check for environment-specified model path
    env_path = os.getenv("MODEL_PATH")
    if env_path and Path(env_path).exists():
        model_path = Path(env_path)
        print(f"Using model from MODEL_PATH: {model_path}")
        
        print(f"\nLoading model: {model_path}")
        try:
            model = load_model_file(model_path)
            input_size = get_input_size(model)
            print(f"[/] Successfully loaded: {model_path.name} ({input_size}x{input_size})")
        except Exception as e:
            print(f"[x] Failed to load model: {e}")
            raise

    else:
        # Download models if they don't exist or if ignore_local is set
        print(f"Models directory: {MODELS_DIR}")
        print(f"Models directory exists: {MODELS_DIR.exists()}")
        
        for local_path, file_id in MODEL_FILES:
            should_download = not local_path.exists() or ignore_local
            
            if should_download:
                if local_path.exists() and ignore_local:
                    print(f"\nModel {local_path.name} exists but IGNORE_LOCAL_COPY is set.")
                    print(f"Removing existing file and re-downloading...")
                    local_path.unlink()
                else:
                    print(f"\nModel {local_path.name} not found locally.")
                
                print("Downloading from Google Drive...")
                success = download_model_from_gdrive(file_id, local_path)
                if not success:
                    print(f"Failed to download {local_path.name}")
            else:
                print(f"Model {local_path.name} already exists ({local_path.stat().st_size} bytes)")
        
        # Find available models
        models = find_models()
        print(f"\nAvailable models: {[m.name for m in models]}")
        
        if not models:
            raise RuntimeError(
                "No models available. Download failed or models directory is empty.\n"
                f"Models directory: {MODELS_DIR}\n"
                f"Directory exists: {MODELS_DIR.exists()}\n"
                f"Directory contents: {list(MODELS_DIR.glob('*')) if MODELS_DIR.exists() else 'N/A'}"
            )
        
        # Initialize Ensemble
        print(f"\nInitializing Ensemble...")
        try:
            model = EnsembleModel(models)
            model.load_models()
            input_size = 224 # Default for ensemble
            print(f"[/] Ensemble initialized with {len(model.models)} models")
        except Exception as e:
            print(f"[x] Failed to load ensemble: {e}")
            raise

    yield
    
    # Cleanup
    model = None
    print("Application shutdown complete")


app = FastAPI(title="EuroSAT Classifier API", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def root():
    html_file = SCRIPT_DIR / "index.html"
    if html_file.exists():
        return html_file.read_text(encoding="utf-8")
    return "<h1>EuroSAT Classifier API</h1><p>Place index.html in API folder for web interface</p>"


@app.get("/api")
def api_info():
    return {
        "status": "ok",
        "model": "Ensemble" if isinstance(model, EnsembleModel) else (model_path.name if model_path else None),
        "input_size": f"{input_size}x{input_size}" if input_size else None,
        "classes": len(CLASS_NAMES)
    }


@app.get("/labels")
def get_labels():
    return {"labels": CLASS_NAMES}


@app.get("/models")
def list_models():
    return {
        "available": [m.name for m in find_models()],
        "current": "Ensemble" if isinstance(model, EnsembleModel) else (model_path.name if model_path else None)
    }

@app.get("/ensemble-info")
def ensemble_info():
    if model and hasattr(model, "get_config"):
        return model.get_config()
    return {"error": "Ensemble model not loaded or using single model"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(503, "Model not loaded")
    
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, f"Invalid content type: {file.content_type}")
    
    try:
        data = await file.read()
        img = Image.open(BytesIO(data))
    except Exception as e:
        raise HTTPException(400, f"Failed to read image: {e}")
    
    try:
        x = preprocess_image(img, input_size) #type: ignore
        probs = model.predict(x, verbose=0)[0]
        top_idx = int(np.argmax(probs))
        
        return {
            "predicted_index": top_idx,
            "predicted_label": CLASS_NAMES[top_idx],
            "confidence": float(probs[top_idx]),
            "probabilities": [
                {"label": CLASS_NAMES[i], "probability": float(probs[i])}
                for i in range(len(probs))
            ]
        }
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(SCRIPT_DIR.parent))
    
    uvicorn.run(
        "API.server:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )