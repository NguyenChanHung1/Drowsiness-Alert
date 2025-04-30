import torchvision.io as io
import os
import argparse
from glob import glob
from tqdm import tqdm
from moviepy import VideoFileClip
import pandas as pd

ML_DIR = "ml"


class VideoDataPreprocessor:
    def __init__(self, data_dir: str | None, num_frames_per_chunk: int = 60, frames_interval: int = 1, output_framedir: str = None, output_clipdir: str= None):
        self.data_dir = data_dir
        self.vid_glob = glob(os.path.join(self.data_dir, '*'))
        self.num_frames_per_chunk = int(num_frames_per_chunk)
        self.frames_interval = frames_interval
        self.output_framedir = output_framedir
        self.output_clipdir = output_clipdir
        self.data_summary = pd.DataFrame({
            "video_name": [],
            "clip_path": [],
            "frames_dir_path": [],
            "length(sec)": [],
            "fps": [],
            "num_frames": [],
            "label": []
        })
        if not os.path.exists(self.output_framedir):
            os.makedirs(self.output_framedir)
        if not os.path.exists(self.output_clipdir):
            os.makedirs(self.output_clipdir)

    def create_vid_folder(self, filename: str):
        if not os.path.exists(filename):
            os.makedirs(filename)

    def extract_frameset_chunks(self, video_file: str):
        video, _, _ = io.read_video(video_file, pts_unit='sec')
        chunk_dict = self.divide_into_chunks(video_file, option='frame')

        for item in chunk_dict.items():
            self.create_vid_folder(item[0])
            for i in range(item[1][0], item[1][1], self.frames_interval):
                frame_order = i - item[1][0]
                frame_file = "_frame" + str(frame_order) + ".jpg"
                frame_path = os.path.join(item[0], frame_file)
                frame = video[i]
                with open(frame_path, 'w') as fp:
                    io.write_jpeg(frame.permute(2,0,1), frame_path)

    def export_analysis(self, export_path: str):
        print("Exporting data analysis to csv...")
        clip_glob = glob(os.path.join(self.output_clipdir, '*/*'))
        for i in tqdm(range(len(clip_glob))):
            video_file = clip_glob[i]
            video, _, info = io.read_video(video_file, pts_unit='sec')
            video_name = '.'.join(clip_glob[i].split('/')[-1].split('.')[:-1])
            clip_path = clip_glob[i]
            frames_dir_path = os.path.join(self.output_framedir, video_name)
            length_in_second = round(video.shape[0] / info["video_fps"], 2)
            num_frames = video.shape[0]
            fps = round(info["video_fps"], 2)
            label = 0 if "focus" in video_name else 1
            new_row = {
                "video_name": video_name,
                "clip_path": clip_path,
                "frames_dir_path": frames_dir_path,
                "length(sec)": length_in_second,
                "fps": fps,
                "num_frames": num_frames,
                "label": int(label)
            }
            self.data_summary = pd.concat([self.data_summary, pd.DataFrame([new_row])], ignore_index=True)
        self.data_summary["label"] = self.data_summary["label"].astype("int")
        self.data_summary.to_csv(export_path+"/data.csv")
        print(f"Data analysis file is exported into {export_path}.")
        count_drowsy_file = len(self.data_summary[self.data_summary.label == 1])
        count_focus_file = len(self.data_summary[self.data_summary.label == 0])
        print(f"There is {len(self.data_summary)} clips in the dataset, with {count_drowsy_file} drowsy clips, and {count_focus_file} focus clips.")
        
    def cut_video(self, input_path: str, output_path: str, start_frame: int, end_frame: int, fps: float):
        start_time = start_frame / fps
        end_time = end_frame / fps - 0.4 if fps != 30.0 else end_frame / fps
        print("Clip video:", start_time, end_time, fps)

        clip = VideoFileClip(input_path).subclipped(start_time, end_time)
        clip.write_videofile(output_path, fps=fps, logger=None)

    def divide_into_chunks(self, video_file: str, option: str):
        chunk_dict = {}
        video, _, info = io.read_video(video_file, pts_unit='sec')
        frame_len = video.shape[0]
        chunk_id = 0
        chunk_name = ""
        cur_frame = 0

        while cur_frame < frame_len:
            if cur_frame // self.num_frames_per_chunk + 1 != chunk_id:
                chunk_id += 1
                if option == 'frame':
                    first = self.output_framedir
                elif option == 'clip':
                    first = self.output_clipdir
                else:
                    raise NotImplementedError
                chunk_name = os.path.join(first, '.'.join(video_file.split('.')[:-1]).split('/')[-1] + "_C" + str(chunk_id)) # example: hung_glass_drowsy1
                start_frame = cur_frame
                if (frame_len - cur_frame) / self.num_frames_per_chunk < 1.5:
                    end_frame = frame_len - 1
                    cur_frame += (end_frame - cur_frame + 1)
                else:
                    end_frame = cur_frame + self.num_frames_per_chunk
                    cur_frame += self.num_frames_per_chunk
                chunk_dict[chunk_name] = (start_frame, end_frame)
        return chunk_dict

    def extract_video_clips(self, video_file: str):
        video, _, info = io.read_video(video_file, pts_unit='sec')
        chunk_dict = self.divide_into_chunks(video_file, option='clip')
        for item in chunk_dict.items():
            output_path = item[0] + ".mp4"
            start_frame, end_frame = item[1][0], item[1][1]
            fps = info["video_fps"]
            self.cut_video(input_path=video_file, output_path=output_path, start_frame=start_frame, end_frame=end_frame, fps=fps)
        
    def extract_clips(self):
        print(f"Found {len(self.vid_glob)} videos in {self.data_dir}.")
        print("Extracting clips from videos...")
        for i in tqdm(range(len(self.vid_glob))):
            self.extract_video_clips(self.vid_glob[i])
        print(f"Non-overlapping video clips are extracted, located in {self.output_clipdir}")

    def extract_frames(self):
        print(f"Found {len(self.vid_glob)} videos in {self.data_dir}.")
        print("Extracting frames from video chunks...")
        for i in tqdm(range(len(self.vid_glob))):
            self.extract_frameset_chunks(self.vid_glob[i])
        print(f"Non-overlapping chunks of frame extraction completed. Frame data is saved into {self.output_framedir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-dir', help="The directory where the videos located.")
    parser.add_argument('-nf', '--number-frames', help="The number of frames for each chunk", default=60)
    parser.add_argument('-int', '--interval', help="Frame interval", default=1)
    parser.add_argument('-of', '--output-framedir', help="Destination folder of frames after extracting")
    parser.add_argument('-oc', '--output-clipdir', help="Destination folder of video clips after extracting")
    args = parser.parse_args()
    video_data_preprocessor = VideoDataPreprocessor(args.data_dir, args.number_frames, args.interval, args.output_framedir, args.output_clipdir)
    # video_data_preprocessor.extract_frames()
    # video_data_preprocessor.extract_clips()
    video_data_preprocessor.export_analysis("ml/dataset")

if __name__ == '__main__':
    main()

