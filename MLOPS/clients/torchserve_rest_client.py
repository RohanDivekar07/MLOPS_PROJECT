"""
TorchServe REST Inference Client.
Sends HTTP/1.1 POST requests to TorchServe REST inference endpoint.
"""

import os
import sys
import time
import argparse
import requests
import json

def run_inference(image_path, host="localhost", port=8080, model_name="resnet50", timeout=10.0):
    url = f"http://{host}:{port}/predictions/{model_name}"
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    headers = {
        "Content-Type": "application/octet-stream"
    }

    start_time = time.perf_counter()
    response = requests.post(url, data=image_bytes, headers=headers, timeout=timeout)
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    if response.status_code != 200:
        raise RuntimeError(f"TorchServe REST error [{response.status_code}]: {response.text}")

    result = response.json()
    return result, latency_ms

def main():
    parser = argparse.ArgumentParser(description="TorchServe REST Client")
    parser.add_argument("--image", type=str, default=os.path.join(os.path.dirname(__file__), "sample_image.jpg"), help="Path to input image")
    parser.add_argument("--host", type=str, default="localhost", help="TorchServe host")
    parser.add_argument("--port", type=int, default=8080, help="TorchServe REST inference port")
    parser.add_argument("--model", type=str, default="resnet50", help="Model name")
    args = parser.parse_args()

    print(f"=== TorchServe REST Client ===")
    print(f"Endpoint: http://{args.host}:{args.port}/predictions/{args.model}")
    print(f"Input Image: {args.image}")

    try:
        predictions, latency = run_inference(args.image, args.host, args.port, args.model)
        print(f"\n[+] Request Successful!")
        print(f"Round-trip Latency: {latency:.2f} ms")
        print("\nTop Predictions:")
        if isinstance(predictions, list) and len(predictions) > 0:
            predictions = predictions[0]
        if isinstance(predictions, dict):
            for label, prob in predictions.items():
                print(f"  - {label:<35} : {prob * 100:.2f}%")
        else:
            print(json.dumps(predictions, indent=2))
    except Exception as e:
        print(f"[-] Inference failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
