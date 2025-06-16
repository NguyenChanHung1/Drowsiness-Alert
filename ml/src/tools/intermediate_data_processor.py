import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from dataset import DrowsinessData
from glob import glob
import cv2
from ultralytics import YOLO
from model.PFLD import PFLDInference # type: ignore
import torchvision.transforms as transforms
import torchvision.io as io
from tqdm import tqdm
from pytorchvideo.transforms import UniformTemporalSubsample

DEVICE = "cuda" # default
NUM_FRAMES = 40
DETECT_SIZE = 320
PFLD_SIZE = 112                                                                                                 
LANDMARKS_MASK = [0, 2, 5, 8, 11, 14, 16, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 27, 29, 31, 33, 35, 60, 61, 63, 64, 65, 67]

class IntermediateDataProcessor:
    def __init__(self, clips_path, num_frames=NUM_FRAMES):
        self.clips_path = clips_path
        self.num_frames = int(num_frames)
        self.detector = YOLO("/data/AI/Driver-Drowsiness-Android-App/ml/weights/YOLO/yolo8n_face_320.pt").to(DEVICE)
        self.landmark_regressor = PFLDInference()
        self.landmark_regressor.load_state_dict(torch.load("ml/weights/PFLD/checkpoint_epoch_250.pth.tar", map_location=DEVICE)["plfd_backbone"])
        self.landmark_regressor.eval()
        self.video_transform = UniformTemporalSubsample(self.num_frames, 0) # Temporal sampling to ensure frame length consistency 
        self.detect_transform = transforms.Compose([transforms.Resize((DETECT_SIZE,DETECT_SIZE)),])
        self.regress_transform = transforms.Compose([transforms.Resize((PFLD_SIZE,PFLD_SIZE)),])# transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)])

    def process_video(self, video_path: str):
        video, _, info = io.read_video(video_path, pts_unit='sec') # [90,w,h,3]
        video = self.video_transform(video)
        landmark_seq_path = video_path.replace("clips", f"landmarks_sequence_t{self.num_frames}").replace("mp4", "txt")
        video = video.permute(0,3,1,2) # [40,3,W,H]
        with open(landmark_seq_path, "w") as f:
            for i in range(video.shape[0]):
                frame = video[i]
                frame = self.detect_transform(frame).unsqueeze(0).float() / 255.0
                detection = self.detector(frame, verbose=False)[0] # The first detection
                x1,y1,x2,y2,conf,cls = detection.boxes.data[0]
                x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                detected_face = frame[:,:,y1:y2,x1:x2] # Get the face cropped

                # Get positions relative to the whole frame
                relocate = torch.Tensor([x1,y1] * 68).reshape(-1,2)
                restore_wh = torch.Tensor([detected_face.shape[3], detected_face.shape[2]])

                # Transform for PFLD
                detected_face = self.regress_transform(detected_face)
                landmarks = self.landmark_regressor(detected_face).view(-1,2) # Landmarks vector [136,] # 68 keypoints of (x,y)
                
                landmarks = (landmarks * restore_wh + relocate) / torch.Tensor([DETECT_SIZE]) # 0 <= landmarks[i] <= 1
                landmarks = landmarks[LANDMARKS_MASK].view(1,-1)[0]
                for lm in landmarks:
                    f.write(str(lm.item()) + " ")
                
                f.write("\n")
        

    def export_intermediate_repr_data(self):
        print("Exporting to landmark sequence annotations...")
        clip_glob = glob(self.clips_path+"/*")
        for clip in tqdm(clip_glob):
            self.process_video(clip)

def intermediate_repr_gen(num_frames) :
    # Full frame
    train_proc = IntermediateDataProcessor("ml/dataset/clips/train", num_frames=num_frames)
    train_proc.export_intermediate_repr_data()
    processor = IntermediateDataProcessor("ml/dataset/clips/val", num_frames=num_frames)
    processor.export_intermediate_repr_data()

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-nf', '--number-frames', help="The number of frames for each chunk", default=40)
    args = parser.parse_args()
    intermediate_repr_gen(args.number_frames)