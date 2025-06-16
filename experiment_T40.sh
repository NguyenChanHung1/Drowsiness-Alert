python3 ml/src/tools/intermediate_data_processor.py -nf 40
python3 ml/src/tools/train.py -nf 40 -d 0
python3 ml/src/tools/train.py -nf 40 -d 1
python3 ml/src/tools/train.py -nf 40 --twostream 1
python3 ml/src/tools/torch2tflite_converter.py -m STGCN2 -tf ml/weights/STGCN/best_t40_twostream.tflite -pt ml/weights/STGCN/best_t40_twostream.pth -onnx ml/weights/STGCN/best_t40_twostream.onnx -sm ml/weights/STGCN/graph_dir_40/ -t 40
python3 ml/src/tools/quantize.py --model_name STGCN --saved_model_path ml/weights/STGCN/graph_dir_40/ --output_path ml/weights/STGCN/best_t40_twostream_uint8.tflite
