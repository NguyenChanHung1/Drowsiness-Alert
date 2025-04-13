import os
import torch 
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms, utils, io
from glob import glob
import pandas as pd
import random
from scipy.ndimage import zoom
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from torchvision.transforms import Compose, Lambda

NUM_FRAMES = 90
IMG_SZ = 172

def default_transform(num_frames: int, img_sz: int):
    return ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(NUM_FRAMES),
                Lambda(lambda x: x/255.0),
                NormalizeVideo([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
                ShortSideScale(
                    size=IMG_SZ
                ),
                CenterCropVideo(IMG_SZ)
            ]
        ),
    )   

class DrowsinessData(Dataset):
    def __init__(self, annotation, split: str, transform=None, transform_prob=0, num_frames=90, img_sz=224):
        self.transform = transform
        self.annotation = pd.read_csv(annotation)
        assert split in ["train", "val", "valid"]
        if split == "valid":
            split = "val"
        self.annotation = self.annotation[self.annotation["clip_path"].str.contains(f"{split}/")].reset_index()
        del self.annotation ["index"]
        self.transform_prob = transform_prob
        self.num_frames = num_frames
        self.img_sz = img_sz

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        # video, _, _ = io.read_video(self.annotation.at[index, "clip_path"], pts_unit='sec')
        video = transform_videos(self.annotation.at[index, "clip_path"], default_transform(self.num_frames,self.img_sz), int(self.annotation.at[index, "num_frames"]), 
                                 int(self.annotation.at[index, "fps"]))
        label = self.annotation.at[index, "label"]
    
        return video, label 
    
def transform_videos(video_path, transform, num_frames, fps):
    clip_duration = num_frames / fps - 1e-3 # Little deviate
    
    start_sec = 0
    end_sec = start_sec + clip_duration
    
    # Initialize an EncodedVideo helper class
    video = EncodedVideo.from_path(video_path)

    # Load the desired clip
    video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
    # Apply a transform to normalize the video input
    video_data = transform(video_data)

    # Move the inputs to the desired device
    video = video_data["video"]
    video = [i[None,...] for i in video]
    return torch.vstack(video).permute(1,0,2,3)