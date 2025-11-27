import argparse
import json
from pathlib import Path

import requests


def predict(server: str, image_path: Path):
    """Send image to API and get prediction."""
    mime_types = {
        '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
        '.png': 'image/png', '.gif': 'image/gif',
        '.bmp': 'image/bmp', '.tiff': 'image/tiff'
    }
    mime = mime_types.get(image_path.suffix.lower(), 'image/jpeg')
    
    with open(image_path, "rb") as f:
        files = {"file": (image_path.name, f, mime)}
        response = requests.post(f"{server.rstrip('/')}/predict", files=files, timeout=60)
    
    response.raise_for_status()
    return response.json()


def main():
    parser = argparse.ArgumentParser(description="EuroSAT Classifier Client")
    parser.add_argument("image", type=Path, help="Image file path")
    parser.add_argument("--server", default="http://127.0.0.1:8000", help="API URL")
    args = parser.parse_args()
    
    if not args.image.exists():
        raise SystemExit(f"Image not found: {args.image}")
    
    print(f"Image: {args.image.name}")
    print(f"Server: {args.server}\n")
    
    try:
        result = predict(args.server, args.image)
        print("[/] Prediction successful!\n")
        print(json.dumps(result, indent=2))
        print(f"\n{'='*50}")
        print(f"Class: {result['predicted_label']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"{'='*50}")
    except requests.HTTPError as e:
        print(f"[x] HTTP Error: {e}")
        if e.response:
            print(f"Response: {e.response.text}")
    except Exception as e:
        print(f"[x] Error: {e}")


if __name__ == "__main__":
    main()