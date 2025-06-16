import numpy as np
import argparse
import tensorflow as tf
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageFilter
import glob
import torch

IMG_SIZE = 112
IMG_DIR = "/data/AI/Driver-Drowsiness-Android-App/ml/src/model/PFLD/data/300W/train_data/imgs"
SEQUENCE_DIR = "/data/AI/Driver-Drowsiness-Android-App/ml/dataset/landmarks_sequence_t60/train"

def deterministic_blur(img:Image.Image):
    return img.filter(ImageFilter.GaussianBlur(radius=1))

color_jitter = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
to_tensor = transforms.ToTensor()
resize = transforms.Resize((IMG_SIZE, IMG_SIZE))

def representative_dataset_gen():
    image_paths = sorted(glob.glob(IMG_DIR+"/*")[:1000])  
    for image_path in image_paths:
        image = Image.open(image_path)
        image = resize(image)
        image = color_jitter(image)
        image = deterministic_blur(image)
        np_img = np.array(image, dtype=np.float32).swapaxes(0,2).swapaxes(1,2) / 255.0
        
        yield [np.expand_dims(np_img, axis=0)]

def representative_landmarks_set_gen():
    sequence_paths = sorted(glob.glob(SEQUENCE_DIR+"/*"))
    for seq in sequence_paths:
        with open(seq, "r") as file:
            landmarks = []
            all_frames = file.readlines()
            for frame in all_frames:
                numbers = list(map(float, frame.strip().split()))
                landmark_frame_i = torch.Tensor(numbers).reshape(-1,2)
                landmarks.append(landmark_frame_i.unsqueeze(0))
            landmarks = torch.vstack(landmarks) # [40,30,2]
            landmarks_numpy = landmarks.permute(2,0,1).unsqueeze(0).numpy().astype(np.float32)
            yield [landmarks_numpy]

def quantize(saved_model_path, output_path, representative_set_generator, input_type=tf.uint8, output_type=tf.uint8):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_set_generator
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.inference_input_type = input_type
    converter.inference_output_type =  output_type

    tflite_int8_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_int8_model)
    print(f"Quantized to TFLite INT8, saved into {output_path}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="Model name")
    parser.add_argument("--saved_model_path", type=str, required=True, help="Path to the saved model directory")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the quantized TFLite model")
    args = parser.parse_args()
    if args.model_name == "PFLD":
        quantize_set_generator = representative_dataset_gen
    else:
        quantize_set_generator = representative_landmarks_set_gen
    quantize(args.saved_model_path, args.output_path, quantize_set_generator)
