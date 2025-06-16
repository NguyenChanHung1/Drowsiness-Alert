import os
import torch 
import numpy as np
from torch.utils.data import Dataset
import pandas as pd

NUM_FRAMES = 90
NUM_LANDMARKS = 68

class DrowsinessData(Dataset):
    def __init__(self, annotation, split: str, T_frame=40, transform=None):
        self.transform = transform
        self.num_frames = T_frame
        self.annotation = pd.read_csv(annotation)
        assert split in ["train", "val", "valid"]
        if split == "valid":
            split = "val"
        self.annotation = self.annotation[self.annotation["clip_path"].str.contains(f"{split}/")].reset_index()
        del self.annotation ["index"]

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        landmarks_path = self.annotation.at[index, "clip_path"].replace("clips", f"landmarks_sequence_t{self.num_frames}").replace("mp4", "txt")
        landmarks = []
        with open(landmarks_path, "r") as file:
            all_frames = file.readlines()
            for frame in all_frames:
                numbers = list(map(float, frame.strip().split()))
                landmark_frame_i = torch.Tensor(numbers).reshape(-1,2)
                landmarks.append(landmark_frame_i.unsqueeze(0))
            landmarks = torch.vstack(landmarks) # [40,30,2]
        label = self.annotation.at[index, "label"] # [0/1]
    
        return landmarks, label 
    
# if __name__ == '__main__':
#     dataset = DrowsinessData('ml/dataset/data.csv', "val")
#     dataset[0], dataset[30]