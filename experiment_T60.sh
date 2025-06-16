python3 ml/src/tools/intermediate_data_processor.py -nf 60
python3 ml/src/tools/train.py -nf 60 -d 0
python3 ml/src/tools/train.py -nf 60 -d 1
python3 ml/src/tools/train.py -nf 60 --twostream 1
python3 ml/src/tools/torch2tflite_converter.py -m STGCN2 -tf ml/weights/STGCN/best_t60_twostream.tflite -pt ml/weights/STGCN/best_t60_twostream.pth -onnx ml/weights/STGCN/best_t60_twostream.onnx -sm ml/weights/STGCN/graph_dir_60/ -t 60
python3 ml/src/tools/quantize.py --model_name STGCN --saved_model_path ml/weights/STGCN/graph_dir_60/ --output_path ml/weights/STGCN/best_t60_twostream_uint8.tflite
