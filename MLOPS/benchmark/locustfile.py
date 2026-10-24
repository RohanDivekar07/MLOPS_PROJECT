"""
Comprehensive Locust Load Testing Suite for Enterprise Model Serving.
Profiles both TorchServe and TensorFlow Serving across REST and gRPC endpoints.
"""

import os
import sys
import json
import numpy as np
from PIL import Image
from locust import HttpUser, task, between, constant, tag, events

# Add repository root to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from benchmark.grpc_locust_user import GrpcUser
from protos.torchserve import inference_pb2, inference_pb2_grpc
from protos.tfserving import tf_grpc_helper

# Cache payloads in-memory to prevent client-side I/O bottleneck
SAMPLE_IMG_PATH = os.path.join(REPO_ROOT, "clients", "sample_image.jpg")
if os.path.exists(SAMPLE_IMG_PATH):
    with open(SAMPLE_IMG_PATH, "rb") as f:
        RAW_IMAGE_BYTES = f.read()
    
    img = Image.open(SAMPLE_IMG_PATH).convert("RGB").resize((224, 224))
    NUMPY_TENSOR = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)
else:
    RAW_IMAGE_BYTES = b"\x00" * 1024
    NUMPY_TENSOR = np.zeros((1, 224, 224, 3), dtype=np.float32)

TF_REST_JSON_PAYLOAD = json.dumps({
    "signature_name": "serving_default",
    "instances": NUMPY_TENSOR.tolist()
})

TORCH_GRPC_REQUEST = inference_pb2.PredictionsRequest(
    model_name="resnet50",
    input={"data": RAW_IMAGE_BYTES}
)

TF_GRPC_REQUEST = tf_grpc_helper.PredictRequest(
    model_name="resnet50",
    signature_name="serving_default",
    inputs={"input_tensor": NUMPY_TENSOR}
)


class TorchServeRestUser(HttpUser):
    """Profiles TorchServe HTTP/1.1 REST inference endpoint (Port 8080)."""
    wait_time = between(0.01, 0.05)
    host = "http://localhost:8080"

    @tag("torchserve", "rest")
    @task
    def predict_rest(self):
        headers = {"Content-Type": "application/octet-stream"}
        with self.client.post(
            "/predictions/resnet50",
            data=RAW_IMAGE_BYTES,
            headers=headers,
            name="TorchServe_REST_Predict",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")


class TorchServeGrpcUser(GrpcUser):
    """Profiles TorchServe HTTP/2 gRPC inference endpoint (Port 7070)."""
    wait_time = between(0.01, 0.05)
    host = "localhost"
    port = 7070

    def on_start(self):
        self.stub = inference_pb2_grpc.InferenceAPIsServiceStub(self.client.channel)

    @tag("torchserve", "grpc")
    @task
    def predict_grpc(self):
        self.client.invoke(
            "TorchServe_gRPC_Predict",
            self.stub.Predictions,
            TORCH_GRPC_REQUEST
        )


class TFServingRestUser(HttpUser):
    """Profiles TensorFlow Serving HTTP/1.1 REST inference endpoint (Port 8501)."""
    wait_time = between(0.01, 0.05)
    host = "http://localhost:8501"

    @tag("tfserving", "rest")
    @task
    def predict_rest(self):
        headers = {"Content-Type": "application/json"}
        with self.client.post(
            "/v1/models/resnet50:predict",
            data=TF_REST_JSON_PAYLOAD,
            headers=headers,
            name="TFServing_REST_Predict",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")


class TFServingGrpcUser(GrpcUser):
    """Profiles TensorFlow Serving HTTP/2 gRPC inference endpoint (Port 8500)."""
    wait_time = between(0.01, 0.05)
    host = "localhost"
    port = 8500

    def on_start(self):
        self.stub = tf_grpc_helper.TFServingStub(self.client.channel)

    @tag("tfserving", "grpc")
    @task
    def predict_grpc(self):
        self.client.invoke(
            "TFServing_gRPC_Predict",
            self.stub.Predict,
            TF_GRPC_REQUEST
        )
