"""
TorchServe gRPC Inference Client.
Sends HTTP/2 Protobuf RPC requests to TorchServe gRPC inference service.
"""

import os
import sys
import time
import json
import argparse
import grpc

# Add repository root to path for proto imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from protos.torchserve import inference_pb2
from protos.torchserve import inference_pb2_grpc

def run_inference(image_path, host="localhost", port=7070, model_name="resnet50", timeout=10.0):
    channel_address = f"{host}:{port}"
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # Create gRPC channel and client stub
    options = [
        ('grpc.max_send_message_length', 100 * 1024 * 1024),
        ('grpc.max_receive_message_length', 100 * 1024 * 1024)
    ]
    channel = grpc.insecure_channel(channel_address, options=options)
    stub = inference_pb2_grpc.InferenceAPIsServiceStub(channel)

    request = inference_pb2.PredictionsRequest(
        model_name=model_name,
        input={"data": image_bytes}
    )

    start_time = time.perf_counter()
    response = stub.Predictions(request, timeout=timeout)
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    # Parse JSON predictions from binary bytes
    prediction_raw = response.prediction
    try:
        result = json.loads(prediction_raw.decode("utf-8"))
    except Exception:
        result = str(prediction_raw)

    return result, latency_ms

def main():
    parser = argparse.ArgumentParser(description="TorchServe gRPC Client")
    parser.add_argument("--image", type=str, default=os.path.join(os.path.dirname(__file__), "sample_image.jpg"), help="Path to input image")
    parser.add_argument("--host", type=str, default="localhost", help="TorchServe gRPC host")
    parser.add_argument("--port", type=int, default=7070, help="TorchServe gRPC port")
    parser.add_argument("--model", type=str, default="resnet50", help="Model name")
    args = parser.parse_args()

    print(f"=== TorchServe gRPC Client ===")
    print(f"Target: {args.host}:{args.port} | Model: {args.model}")
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
