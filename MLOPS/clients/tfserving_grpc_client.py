"""
TensorFlow Serving gRPC Inference Client.
Sends HTTP/2 Protobuf RPC requests to TF Serving gRPC endpoint.
"""

import os
import sys
import time
import json
import argparse
import grpc
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from protos.tfserving import tf_grpc_helper

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224), Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def load_labels(label_path=None):
    if not label_path:
        label_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "index_to_name.json"))
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            return json.load(f)
    return {str(i): f"class_{i}" for i in range(1000)}

def run_inference(image_path, host="localhost", port=8500, model_name="resnet50", signature_name="serving_default", timeout=10.0):
    channel_address = f"{host}:{port}"
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    tensor = preprocess_image(image_path)

    options = [
        ('grpc.max_send_message_length', 100 * 1024 * 1024),
        ('grpc.max_receive_message_length', 100 * 1024 * 1024)
    ]
    channel = grpc.insecure_channel(channel_address, options=options)
    stub = tf_grpc_helper.TFServingStub(channel)

    request = tf_grpc_helper.PredictRequest(
        model_name=model_name,
        signature_name=signature_name,
        inputs={"input_tensor": tensor}
    )

    start_time = time.perf_counter()
    response = stub.Predict(request, timeout=timeout)
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    # Extract probabilities
    probs = None
    for k, v in response.outputs.items():
        probs = v
        break

    if probs is None or len(probs) == 0:
        raise RuntimeError("No prediction outputs returned from TF Serving gRPC.")

    labels = load_labels()
    top_indices = np.argsort(probs)[-5:][::-1]
    top_predictions = {labels.get(str(i), f"class_{i}"): round(float(probs[i]), 6) for i in top_indices}

    return top_predictions, latency_ms

def main():
    parser = argparse.ArgumentParser(description="TensorFlow Serving gRPC Client")
    parser.add_argument("--image", type=str, default=os.path.join(os.path.dirname(__file__), "sample_image.jpg"), help="Path to input image")
    parser.add_argument("--host", type=str, default="localhost", help="TF Serving gRPC host")
    parser.add_argument("--port", type=int, default=8500, help="TF Serving gRPC port")
    parser.add_argument("--model", type=str, default="resnet50", help="Model name")
    args = parser.parse_args()

    print(f"=== TensorFlow Serving gRPC Client ===")
    print(f"Target: {args.host}:{args.port} | Model: {args.model}")
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
