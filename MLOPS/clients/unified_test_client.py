"""
Unified Test Client for Enterprise Model Serving Benchmark.
Executes test inferences across all four server configurations:
1. TorchServe REST (port 8080)
2. TorchServe gRPC (port 7070)
3. TensorFlow Serving REST (port 8501)
4. TensorFlow Serving gRPC (port 8500)
"""

import os
import sys
import argparse
import time

from clients import torchserve_rest_client
from clients import torchserve_grpc_client
from clients import tfserving_rest_client
from clients import tfserving_grpc_client

def test_endpoint(name, runner_fn, *args, **kwargs):
    print(f"\n=======================================================")
    print(f" Testing Endpoint: {name}")
    print(f"=======================================================")
    try:
        preds, latency = runner_fn(*args, **kwargs)
        print(f" [PASS] Success! Round-trip Latency: {latency:.2f} ms")
        if isinstance(preds, list) and len(preds) > 0:
            preds = preds[0]
        if isinstance(preds, dict):
            print(" Top-3 Predictions:")
            for i, (k, v) in enumerate(preds.items()):
                if i >= 3:
                    break
                print(f"   * {k:<30} : {v * 100:.2f}%")
        return True, latency
    except Exception as e:
        print(f" [FAIL] Error connecting to {name}: {e}")
        return False, None

def main():
    parser = argparse.ArgumentParser(description="Unified Serving Benchmark Verification Client")
    parser.add_argument("--image", type=str, default=os.path.join(os.path.dirname(__file__), "sample_image.jpg"), help="Path to input image")
    parser.add_argument("--host", type=str, default="localhost", help="Serving host")
    parser.add_argument(
        "--target",
        type=str,
        default="all",
        choices=["all", "torchserve-rest", "torchserve-grpc", "tfserving-rest", "tfserving-grpc"],
        help="Target endpoint to test"
    )
    args = parser.parse_args()

    results = {}

    if args.target in ["all", "torchserve-rest"]:
        ok, lat = test_endpoint("TorchServe REST (Port 8080)", torchserve_rest_client.run_inference, args.image, args.host, 8080)
        results["TorchServe REST"] = (ok, lat)

    if args.target in ["all", "torchserve-grpc"]:
        ok, lat = test_endpoint("TorchServe gRPC (Port 7070)", torchserve_grpc_client.run_inference, args.image, args.host, 7070)
        results["TorchServe gRPC"] = (ok, lat)

    if args.target in ["all", "tfserving-rest"]:
        ok, lat = test_endpoint("TF Serving REST (Port 8501)", tfserving_rest_client.run_inference, args.image, args.host, 8501)
        results["TF Serving REST"] = (ok, lat)

    if args.target in ["all", "tfserving-grpc"]:
        ok, lat = test_endpoint("TF Serving gRPC (Port 8500)", tfserving_grpc_client.run_inference, args.image, args.host, 8500)
        results["TF Serving gRPC"] = (ok, lat)

    print("\n================== Verification Summary ==================")
    for target, (status, lat) in results.items():
        stat_str = "[ONLINE]" if status else "[UNREACHABLE / OFFLINE]"
        lat_str = f"{lat:.2f} ms" if lat else "N/A"
        print(f"{target:<25} : {stat_str:<22} (Latency: {lat_str})")
    print("==========================================================")

if __name__ == "__main__":
    main()
