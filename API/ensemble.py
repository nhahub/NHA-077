import os
import numpy as np
import pickle
from pathlib import Path
from tensorflow.keras.models import load_model #type: ignore
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess #type: ignore
from tensorflow.keras.utils import image_dataset_from_directory #type: ignore
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf

def standard_preprocess(x):
    """Standard preprocessing: scale to [0, 1]."""
    return x / 255.0

class EnsembleModel:
    def __init__(self, model_paths):
        self.model_paths = model_paths
        self.models = []
        self.preprocessors = []
        self.input_sizes = []
        self.loaded_model_names = []
        self.strategy = os.getenv("VOTING_STRATEGY", "soft").lower()

    def load_models(self):
        self.models = []
        self.preprocessors = []
        self.input_sizes = []
        self.loaded_model_names = []
        
        print(f"Initializing Ensemble with strategy: {self.strategy}")
        
        for path in self.model_paths:
            try:
                if not path.exists():
                    print(f"[Ensemble] Warning: Model file not found: {path}")
                    continue
                
                print(f"[Ensemble] Loading {path.name}...")
                if path.suffix == ".keras":
                    model = load_model(str(path))
                elif path.suffix == ".pkl":
                    with open(path, "rb") as f:
                        model = pickle.load(f)
                else:
                    print(f"[Ensemble] Warning: Unsupported file format {path.suffix} for {path.name}")
                    continue
                
                # Determine preprocessing
                if "resnet" in path.name.lower():
                    print(f"[Ensemble] Using ResNet preprocessing for {path.name}")
                    self.preprocessors.append(resnet_preprocess)
                else:
                    print(f"[Ensemble] Using Standard preprocessing (1/255) for {path.name}")
                    self.preprocessors.append(standard_preprocess)

                # Determine input size
                input_size = 224
                try:
                    if hasattr(model, 'input_shape') and model.input_shape[1]:
                        input_size = model.input_shape[1]
                except:
                    pass
                self.input_sizes.append(input_size)
                print(f"[Ensemble] Model {path.name} expects input size: {input_size}")

                self.models.append(model)
                self.loaded_model_names.append(path.name)
                print(f"[Ensemble] Successfully loaded {path.name}")
                
            except Exception as e:
                print(f"[Ensemble] Error loading {path.name}: {e}")
        
        if not self.models:
            raise RuntimeError("No models could be loaded for the ensemble.")
        
        print(f"[Ensemble] Ready with {len(self.models)} models.")

    def predict(self, image_array, verbose=0):
        if not self.models:
            raise RuntimeError("Ensemble not initialized or no models loaded.")

        predictions = []
        predictions = []
        for model, preproc, size in zip(self.models, self.preprocessors, self.input_sizes):
            x = image_array.copy()
            
            # Resize if necessary
            if x.shape[1] != size or x.shape[2] != size:
                # Use tf.image.resize if available, else assume input is compatible or fail
                # We imported tensorflow as tf
                x = tf.image.resize(x, (size, size)).numpy()
            
            x = preproc(x)
            
            pred = model.predict(x, verbose=0)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        if self.strategy == "hard":
            return self._hard_voting(predictions)
        else:
            return self._soft_voting(predictions)

    def _soft_voting(self, predictions):
        avg_pred = np.mean(predictions, axis=0)
        return avg_pred

    def _hard_voting(self, predictions):
        votes = np.argmax(predictions, axis=2)
        
        num_classes = predictions.shape[2]
        vote_counts = np.zeros((1, num_classes))
        
        for vote in votes:
            vote_counts[0, vote] += 1
            
        vote_probs = vote_counts / len(self.models)
        
        return vote_probs

    def get_config(self):
        return {
            "strategy": self.strategy,
            "model_count": len(self.models),
            "models": self.loaded_model_names
        }


if __name__ == "__main__":
    # Setup paths
    BASE_DIR = Path(__file__).parent.resolve()
    MODELS_DIR = BASE_DIR / "Models"
    DATASET_DIR = BASE_DIR.parent / "Dataset" / "EuroSAT_RGB_split" / "test"
    
    print(f"Models Directory: {MODELS_DIR}")
    print(f"Dataset Directory: {DATASET_DIR}")
    
    # Check if dataset exists
    if not DATASET_DIR.exists():
        print(f"Error: Dataset directory not found at {DATASET_DIR}")
        exit(1)

    # Initialize Ensemble
    model_files = [
        MODELS_DIR / "model_vgg16.keras",
        MODELS_DIR / "sequential_model.keras",
        MODELS_DIR / "model_resnet50.keras",
    ]
    
    ensemble = EnsembleModel(model_files)
    try:
        ensemble.load_models()
    except Exception as e:
        print(f"Failed to load models: {e}")
        exit(1)

    # Load Test Data
    BATCH_SIZE = 32
    IMG_SIZE = (224, 224)
    
    print("Loading test dataset...")

    test_ds = image_dataset_from_directory(
        DATASET_DIR,
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    class_names = test_ds.class_names
    print(f"Classes: {class_names}")

    y_true = []
    y_pred = []
    
    print("Running predictions...")
    for images, labels in test_ds:
        imgs_np = images.numpy()
        preds = ensemble.predict(imgs_np)
        pred_indices = np.argmax(preds, axis=1)
        
        y_true.extend(labels.numpy())
        y_pred.extend(pred_indices)
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)


    print("Accuracy:", accuracy_score(y_true, y_pred))

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Ensemble Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # plot_path = BASE_DIR / "ensemble_confusion_matrix.png"
    # plt.savefig(plot_path)

    plt.show()

