"""
Pretrained Model Exporter for Enterprise Model Serving Benchmark.
Exports pre-trained ResNet-50 weights (ImageNet) for both TorchServe (.mar)
and TensorFlow Serving (SavedModel format).
"""

import os
import sys
import json
import zipfile
import argparse
import datetime

def export_torchserve_pretrained(output_dir, model_name="resnet50"):
    print("[1/2] Exporting Pretrained PyTorch ResNet-50...")
    import torch
    import torchvision.models as models

    # Load pretrained ImageNet ResNet-50
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.eval()

    example_input = torch.rand(1, 3, 224, 224)
    traced_model = torch.jit.trace(model, example_input)

    temp_pt = os.path.join(output_dir, f"{model_name}.pt")
    traced_model.save(temp_pt)
    print(f" -> Traced model saved to {temp_pt}")

    mar_dest = os.path.join(output_dir, f"{model_name}.mar")
    handler_path = os.path.abspath(os.path.join(output_dir, "..", "custom_handler.py"))
    mapping_path = os.path.abspath(os.path.join(output_dir, "..", "..", "index_to_name.json"))

    manifest = {
        "createdOn": datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "description": "Pre-trained ResNet-50 ImageNet Archive",
        "torchserveVersion": "0.11.0",
        "schemaVersion": "1.0",
        "model": {
            "modelName": model_name,
            "serializedFile": f"{model_name}.pt",
            "handler": "custom_handler.py",
            "modelVersion": "1.0"
        }
    }

    with zipfile.ZipFile(mar_dest, "w", zipfile.ZIP_DEFLATED) as mar:
        mar.writestr("MAR-INF/MANIFEST.json", json.dumps(manifest, indent=2))
        mar.write(temp_pt, arcname=f"{model_name}.pt")
        if os.path.exists(handler_path):
            mar.write(handler_path, arcname="custom_handler.py")
        if os.path.exists(mapping_path):
            mar.write(mapping_path, arcname="index_to_name.json")

    print(f" -> TorchServe archive generated: {mar_dest}")
    return mar_dest


def export_tfserving_pretrained(output_dir, model_name="resnet50"):
    print("[2/2] Exporting Pretrained TensorFlow ResNet-50...")
    import tensorflow as tf
    from tensorflow.keras.applications.resnet50 import ResNet50

    model = ResNet50(weights="imagenet", classes=1000, input_shape=(224, 224, 3))
    version_dir = os.path.join(output_dir, "resnet50", "1")
    os.makedirs(version_dir, exist_ok=True)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 224, 224, 3], dtype=tf.float32, name="input_tensor")])
    def serve_fn(input_tensor):
        outputs = model(input_tensor, training=False)
        probabilities = tf.nn.softmax(outputs, axis=-1)
        return {"probabilities": probabilities}

    tf.saved_model.save(
        model,
        version_dir,
        signatures={"serving_default": serve_fn}
    )
    print(f" -> TF Serving SavedModel exported to: {version_dir}")
    return version_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Pretrained Models for Serving Benchmark")
    parser.add_argument("--offline", action="store_true", help="Generate models with random weights without downloading")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    ts_dir = os.path.join(base_dir, "torchserve", "model_store")
    tf_dir = os.path.join(base_dir, "tfserving")

    if args.offline:
        import create_dummy_models
        create_dummy_models.generate_torchserve_model(ts_dir)
        create_dummy_models.generate_tfserving_model(tf_dir)
    else:
        try:
            export_torchserve_pretrained(ts_dir)
            export_tfserving_pretrained(tf_dir)
        except Exception as e:
            print(f"Failed to download pretrained weights: {e}")
            print("Falling back to offline model initialization...")
            import create_dummy_models
            create_dummy_models.generate_torchserve_model(ts_dir)
            create_dummy_models.generate_tfserving_model(tf_dir)
