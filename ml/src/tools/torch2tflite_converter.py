import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
from model.PFLD import PFLDInference

def pt2onnx(model: nn.Module, sample_input: torch.Tensor, output_file: str):
    torch.onnx.export(
        model,
        sample_input,
        output_file,
        input_names=['input_1'],
        opset_version=18
    )
    print(f"Converted to onnx, saved to {output_file}")

def onnx2tflite(model_path, graph_dir: str, tflite_model_path):
    onnx_model = onnx.load(model_path)
    tf_rep = prepare(onnx_model)

    tf_rep.export_graph(graph_dir)

    converter = tf.lite.TFLiteConverter.from_saved_model(graph_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Optional: enable quantization if needed
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()

    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)

    print(f"Exported TensorFlow SavedModel to {tflite_model_path}")

if __name__ == '__main__':
    pt_model = PFLDInference().to("cuda")
    pt_model.load_state_dict(torch.load("ml/weights/PFLD/checkpoint_epoch_200.pth.tar", map_location="cuda")["plfd_backbone"])
    sample_input = torch.randn(1,3,112,112).to("cuda")
    output_file = "ml/weights/PFLD/pfld2.onnx"
    pt2onnx(pt_model, sample_input, output_file)

    model_path = "ml/weights/PFLD/pfld2.onnx"
    graph_dir = "ml/weights/PFLD/graph_dir"
    tflite_model_path = "ml/weights/PFLD/pfld.tflite"
    onnx2tflite(model_path, graph_dir, tflite_model_path)