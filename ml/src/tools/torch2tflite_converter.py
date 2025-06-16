import sys
import os
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import numpy as np
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
from model.PFLD import PFLDInference
from model.STGCN import STGCN, TwoStreamSTGCN

def pt2onnx(model: nn.Module, sample_input: torch.Tensor, output_file: str):
    torch.onnx.export(
        model,
        sample_input,
        output_file,
        input_names=['input_1'], 
        output_names=['output_landmarks'],
        opset_version=12
    )
    print(f"Converted to onnx, saved to {output_file}")

def onnx2tflite(model_path, graph_dir: str, tflite_model_path):
    onnx_model = onnx.load(model_path)
    tf_rep = prepare(onnx_model)

    tf_rep.export_graph(graph_dir)

    converter = tf.lite.TFLiteConverter.from_saved_model(graph_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  
    converter.target_spec.supported_types = [tf.float32]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    tflite_model = converter.convert()

    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)

    print(f"Exported TensorFlow SavedModel to {graph_dir}")

def main(model_name, output_file, tflite_path, saved_model_path, pt_weights_path=None, in_channels=2, num_class=2, T_frame=40, deep=False):
    if model_name == 'STGCN':
        pt_model = STGCN(in_channels,num_class,T_frame=T_frame,deep=deep,device='cuda').to("cuda") # [1,2,40,30], 3 layers
    elif model_name == 'STGCN2':
        pt_model = TwoStreamSTGCN(in_channels, num_class, T_frame=T_frame).to("cuda")
    else:
        pt_model = PFLDInference().to("cuda")
    if pt_weights_path is not None and model_name in ['STGCN', 'STGCN2']:
        pt_model.load_state_dict(torch.load(pt_weights_path, map_location='cpu')['model_state_dict'])
        sample_input = torch.randn(1,in_channels,T_frame,30).to("cuda")
    elif pt_weights_path is not None and model_name == 'PFLD':
        pt_model.load_state_dict(torch.load(pt_weights_path, map_location='cpu')['plfd_backbone'])
        sample_input = torch.randn(1, 3, 112, 112).to("cuda")  
    pt2onnx(pt_model, sample_input, output_file)

    # graph_dir = "ml/weights/STGCN/graph_dir_dynamic_60_twostream"
    onnx2tflite(output_file, saved_model_path, tflite_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-name", type=str, default="STGCN",
                        help="The name of the model to convert (STGCN or TwoStreamSTGCN)")
    parser.add_argument("-tf", "--tflite-path", type=str)
    parser.add_argument("-pt", "--torch-path", type=str)
    parser.add_argument("-onnx", "--onnx-path", type=str)
    parser.add_argument("-sm", "--saved_model", help="The path to saved model")
    parser.add_argument("-t", "--len", type=int, help="The length of the input sequence")
    parser.add_argument("--deep", help="If deep=True, number of STGCN layers is 6, otherwise it's 3")
    args = parser.parse_args()
    if args.deep is None:
        deep = 0
    else:
        deep = int(args.deep)
    main(parser.model_name, args.onnx_path, args.tflite_path, args.saved_model,
         pt_weights_path=args.torch_path, T_frame=int(args.len), deep=deep>0)
