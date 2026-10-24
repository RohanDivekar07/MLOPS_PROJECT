"""
TensorFlow Serving REST Inference Client.
Sends HTTP/1.1 JSON POST requests to TF Serving REST endpoint.
"""

import os
import sys
import time
import json
import argparse
import requests
import numpy as np
from PIL import Image

def preprocess_image(image_path):
    """
    Standard ResNet-50 ImageNet preprocessing:
    Resize 256x256, center crop 224x224, normalize to [0, 1] or ImageNet standardization.
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224), Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    # Add batch dimension: [1, 224, 224, 3]
    return np.expand_dims(arr, axis=0)

def load_labels(label_path=None):
    if not label_path:
        label_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "index_to_name.json"))
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            return json.load(f)
    return {str(i): f"class_{i}" for i in range(1000)}

def run_inference(image_path, host="localhost", port=8501, model_name="resnet50", timeout=10.0):
    url = f"http://{host}:{port}/v1/models/{model_name}:predict"
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    tensor = preprocess_image(image_path)
    
    # TF Serving REST payload specification
    payload = json.dumps({
        "signature_name": "serving_default",
        "instances": tensor.tolist()
    })

    headers = {"Content-Type": "application/json"}

    start_time = time.perf_counter()
    response = requests.post(url, data=payload, headers=headers, timeout=timeout)
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    if response.status_code != 200:
        raise RuntimeError(f"TF Serving REST error [{response.status_code}]: {response.text}")

    result = response.json()
    predictions = result.get("predictions", [])
    if len(predictions) > 0:
        probs = predictions[0]
        if isinstance(probs, dict) and "probabilities" in probs:
            probs = probs["probabilities"]
    else:
        probs = []

    # Map top-5
    labels = load_labels()
    top_indices = np.argsort(probs)[-5:][::-1]
    top_predictions = {labels.get(str(i), f"class_{i}"): round(float(probs[i]), 6) for i in top_indices}

    return top_predictions, latency_ms

def main():
    parser = argparse.ArgumentParser(description="TensorFlow Serving REST Client")
    parser.add_argument("--image", type=str, default=os.path.join(os.path.dirname(__file__), "sample_image.jpg"), help="Path to input image")
    parser.add_argument("--host", type=str, default="localhost", help="TF Serving host")
    parser.add_argument("--port", type=int, default=8501, help="TF Serving REST port")
    parser.add_argument("--model", type=str, default="resnet50", help="Model name")
    args = parser.parse_args()

    print(f"=== TensorFlow Serving REST Client ===")
    print(f"Endpoint: http://{args.host}:{args.port}/v1/models/{args.model}:predict")
    print(f"Input Image: {args.image}")

    try:
        top_preds, latency = run_inference(args.image, args.host, args.port, args.model)
        print(f"\n[+] Request Successful!")
        print(f"Round-trip Latency: {latency:.2f} ms")
        print("\nTop Predictions:")
        for label, prob in top_preds.items():
            print(f"  - {label:<35} : {prob * 100:.2f}%")
    except Exception as e:
        print(f"[-] Inference failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
