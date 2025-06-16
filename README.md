# Driver-Drowsiness-Android-App

## ML models experiments
Clone this repo:
https://github.com/samuelyu2002/PFLD.git

cd PFLD

Inside train.py, check the correct data path:
# --dataset
parser.add_argument(
    '--dataroot',
    default='./data/300W/train_data/list.txt',
    type=str,
    metavar='PATH')
parser.add_argument(
    '--val_dataroot',
    default='./data/300W/test_data/list.txt',
    type=str,
    metavar='PATH')

Download 300W dataset from 
https://drive.google.com/file/d/1zyahLoR9i2hgS7d_taRPcYBYJzD3Ir1b/view?usp=sharing and unzip in the data/300W/ folder
Re-train PFLD:
python train.py

After the re-training process finishes, move the last checkpoint file into the `ml/weights/` folder:
cd ../
mv PFLD/checkpoint/1/checkpoint_epoch_{last_epoch}.pth.tar ml/weights/PFLD/checkpoint_epoch_{last_epoch}.pth.tar

Convert and quantize PFLD to UINT8:
python3 ml/src/tools/torch2tfliteconverter.py -m PFLD -tf ml/weights/PFLD/pfld.tflite -pt ml/weights/PFLD/checkpoint_epoch_{last_epoch}.pth.tar -onnx ml/weights/PFLD/pfld.onnx -sm ml/weights/PFLD/graph_dir
python3 ml/src/tools/quantize.py --model_name PFLD --saved_model_path ml/weights/PFLD/graph_dir/ --output_path ml/weights/PFLD/pfld_uint8.tflite

Download the drowsiness dataset from this link: 
https://drive.google.com/file/d/1uapqLnNbSWAyqcO1CXUvQpKTwnOCrpeb/view?usp=drive_link


and unzip it to the folder ml/raw-data

Process the data to train:
python3 ml/src/tools/data_preprocessor.py -d ml/raw-data/ -nf 90 -of ml/dataset/frame -oc ml/dataset/clips

Run training and post-training quantization (to UINT8) with Two-stream ST-GCN models (40 frames and 60 frames version):
sh experiment_T40.sh
sh experiment_T60.sh

## Run mobile app
Open Android Studio and click on Run button
