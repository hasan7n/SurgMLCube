from tqdm import tqdm
import os
import yaml
import argparse
import csv

from utils import _filename_no_ext, _file_ext, _get_video_fps
from utils import LabelsParser


class DataPreparation:
    def __init__(self, vids_path, labels_path, params_file, output_path):
        with open(params_file, "r") as f:
            self.params = yaml.full_load(f)

        self.vids_path = vids_path
        self.labels_path = labels_path
        self.output_path = output_path

        self.supported_videos = []
        self.supported_labels = []

        # TODO: check what ffmpeg is not capable of handling, or what it can handle but in a different way
        self.supported_video_extensions = [".mp4"]
        self.supported_labels_extensions = [".txt", ".csv", ".json"]

    
    def get_and_check_video_files(self):

        for filename in os.listdir(self.vids_path):
            extenstion = _file_ext(filename)
            file = os.path.join(self.vids_path, filename)

            if extenstion in self.supported_video_extensions:
                self.supported_videos.append(file)
            else:
                print(f"Warning: Unrecognized video file type: {file}")
    
    def get_and_check_label_files(self):

        for filename in os.listdir(self.labels_path):
            extenstion = _file_ext(filename)
            file = os.path.join(self.labels_path, filename)

            if extenstion in self.supported_labels_extensions:
                self.supported_labels.append(file)
            else:
                print(f"Warning: Unrecognized label file type: {file}")

    def assign_labels_to_videos(self):

        # TODO: should we accept other naming pattern conventions? e.g. (video1.mp4 and video1_labels.txt)
        #       Currently, same name should be assigned to both the video and the labels file

        video_names = list(map(lambda x:_filename_no_ext(x), self.supported_videos))
        labels_names = list(map(lambda x:_filename_no_ext(x), self.supported_labels))

        # remove duplicate files if any
        unique_videos = []
        for i, vid_name in enumerate(video_names):
            if vid_name in unique_videos:
                print(f"Warning: Found multiple video files with the same name but with different file extensions: {self.supported_videos[i]} will be ignored")
                self.supported_videos[i] = None
            else:
                unique_videos.append(vid_name)
        
        unique_labels = []
        for i, label_name in enumerate(labels_names):
            if label_name in unique_labels:
                print(f"Warning: Found multiple label files with the same name but with different file extensions: {self.supported_labels[i]} will be ignored")
                self.supported_labels[i] = None
            else:
                unique_labels.append(label_name)
        
        self.supported_videos = [item for item in self.supported_videos if item]
        self.supported_labels = [item for item in self.supported_labels if item]

        # associate video-label pairs
        self.videos_labels_pairs = {}
        assigned_labels = []
        for i, video in enumerate(unique_videos):
            vid_path = self.supported_videos[i]
            try:
                label_index = unique_labels.index(video)
                label_file = self.supported_labels[label_index]
                self.videos_labels_pairs[vid_path] = {"labels": label_file, "fps": _get_video_fps(vid_path)}
                assigned_labels.append(label_file)
            except ValueError:
                print(f"Warning: {self.supported_videos[i]} has no associated labels. It will be ignored")
        
        if len(assigned_labels) != len(unique_labels):
            for i, label in enumerate(unique_labels):
                if label not in assigned_labels:
                    print(f"Warning: {self.supported_labels[i]} has no associated video. It will be ignored")


    def process_videos(self):

        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

        frames_path = os.path.join(self.output_path, "frames")
        if not os.path.exists(frames_path):
            os.mkdir(frames_path)

        scale = self.params["scale"]
        fps = self.params["fps"]

        print(f"Extracting videos:\n\tSampling: {fps} frames per second\n\toutput frame scale: {scale[0]}-by-{scale[1]}\n")
        for vid_path in tqdm(self.videos_labels_pairs.keys()):
            file_name = _filename_no_ext(vid_path)
            out_folder = os.path.join(frames_path, file_name)

            if not os.path.exists(out_folder):
                os.mkdir(out_folder)

            imgs_prefix_name = os.path.join(out_folder, file_name)

            os.system(
                f'ffmpeg -loglevel quiet -i {vid_path} -vf "scale={scale[0]}:{scale[1]},fps={fps}" {imgs_prefix_name}_%06d.png'
            )
            print(f"Done extracting: {vid_path}")

    def process_labels(self):

        out_path = os.path.join(self.output_path, "data_csv")
        if not os.path.exists(out_path):
            os.mkdir(out_path)

        for vid in self.videos_labels_pairs.keys():
            
            out_file = os.path.join(out_path, _filename_no_ext(vid)+".csv")

            frames_folder = os.path.join(self.output_path, "frames", _filename_no_ext(vid))

            frames = os.listdir(frames_folder)
            frames.sort()


            labels_file = self.videos_labels_pairs[vid]["labels"]
            video_fps = self.videos_labels_pairs[vid]["fps"]

            labels_file_type = _file_ext(labels_file)
            if labels_file_type in [".csv", ".txt"]:
                labels_data = LabelsParser.parse_csv_txt_labels(labels_file, video_fps, self.params["labels"])
            elif labels_file_type == ".json":
                labels_data = LabelsParser.parse_json_labels(labels_file, video_fps, self.params["labels"])

            # apply the effect of frame sampling
            labels_data = labels_data[::round(video_fps/self.params["fps"])]

            dropped_frames = 0
            dropped_labels = 0

            if len(frames) > len(labels_data):
                # drop video frames from end if they were not included in the labels file
                dropped_frames += len(frames) - len(labels_data)
                frames = frames[:len(labels_data)]
            
            elif len(frames) < len(labels_data):
                # drop labels from end if there was no corresponding frame
                dropped_labels += len(labels_data) - len(frames)
                labels_data = labels_data[:len(frames)]
            
            # if there is any other missing label, remove the corresponding frames
            frames = [frame for i, frame in enumerate(frames) if labels_data[i] != None]
            dropped_frames += len(labels_data) - len(frames)
            labels_data = [label for label in labels_data if label != None]

            frames = list(map(lambda x: os.path.join(frames_folder, x), frames))
            frames = list(map(lambda x: os.path.relpath(x, self.output_path), frames))

            # write the data
            with open(out_file, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["frame_path", "label"])
                for frame_path, label in zip(frames, labels_data):
                    writer.writerow([frame_path, label])
            
            if dropped_frames:
                print(f"Warning: {dropped_frames} frames of the video {vid} have no corresponding labels.")
            
            if dropped_labels:
                print(f"Warning: {dropped_labels} extra labels for the video {vid} has been neglected.")
            


    def run(self):

        self.get_and_check_video_files()
        self.get_and_check_label_files()

        self.assign_labels_to_videos()

        self.process_videos()
        self.process_labels()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--vids_path",
        "--vids-path",
        type=str,
        required=True,
        help="Location of videos",
    )

    parser.add_argument(
        "--labels_path",
        "--labels-path",
        type=str,
        required=True,
        help="Location of labels",
    )

    parser.add_argument(
        "--params_file",
        "--params-file",
        type=str,
        required=True,
        help="Configuration file for the data-preparation step",
    )

    parser.add_argument(
        "--output_path",
        "--output-path",
        type=str,
        required=True,
        help="Location to store the prepared data",
    )

    args = parser.parse_args()
    preprocessor = DataPreparation(args.vids_path,
                                   args.labels_path,
                                args.params_file,
                                args.output_path
                                )
    preprocessor.run()

