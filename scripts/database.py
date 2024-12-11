import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import json

class FallDetectionVideoDataset(Dataset):
    def __init__(self, data_dir, img_size=(224, 224), transform=None):
        """
        Dataset for fall detection with RGB, mask, center-of-mass (CoM), and skeleton angle data.

        Args:
            data_dir (str): Path to the directory containing train or test data.
            img_size (tuple): Desired image size (height, width) for resizing.
            transform (torchvision.transforms): Transformations to apply to the frames.
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.video_paths = []
        self.mask_paths = []
        self.coord_files = []
        self.angle_files = []
        self.labels = []
        
        for label, category in enumerate(["not_fall", "fall"]):
            category_path = os.path.join(data_dir, category)
            raw_videos_path = os.path.join(category_path, "raw_videos")
            mask_videos_path = os.path.join(category_path, "mask_videos")
            coord_files_path = os.path.join(category_path, "points")
            angle_files_path = os.path.join(category_path, "angles")
            
            if not os.path.exists(angle_files_path):
                raise ValueError(f"Missing directory for angles: {angle_files_path}")
            
            for video in os.listdir(raw_videos_path):
                if video.endswith(".mp4"):
                    video_name = os.path.splitext(video)[0]
                    raw_video_path = os.path.join(raw_videos_path, video)
                    mask_video_path = os.path.join(mask_videos_path, video)
                    coord_file_path = os.path.join(coord_files_path, f"{video_name}.txt")
                    angle_file_path = os.path.join(angle_files_path, f"{video_name}.json")

                    if not all(os.path.exists(p) for p in [mask_video_path, coord_file_path, angle_file_path]):
                        raise ValueError(f"Missing mask, coordinate, or angle file for {raw_video_path}")

                    self.video_paths.append(raw_video_path)
                    self.mask_paths.append(mask_video_path)
                    self.coord_files.append(coord_file_path)
                    self.angle_files.append(angle_file_path)
                    self.labels.append(label)
        
        self.transform = transform or transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
        ])
        self.rgb_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.mask_normalize = transforms.Normalize(mean=[0.5], std=[0.5])

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        """
        Retrieve a single sample containing RGB, mask, center-of-mass, skeleton angles, and label.
        """
        raw_video_path = self.video_paths[idx]
        mask_video_path = self.mask_paths[idx]
        coord_file_path = self.coord_files[idx]
        angle_file_path = self.angle_files[idx]

        rgb_frames = self._load_video(raw_video_path)
        mask_frames = self._load_video(mask_video_path, is_mask=True)
        coord_frames = self._load_coordinates(coord_file_path)
        angle_frames = self._load_angles(angle_file_path)

        # Sample frames
        total_frames = len(rgb_frames)
        num_frames = self._determine_num_frames(total_frames)
        rgb_frames = self._sample_frames(rgb_frames, num_frames)
        mask_frames = self._sample_frames(mask_frames, num_frames)
        coord_frames = self._sample_coords(coord_frames, num_frames)
        angle_frames = self._sample_angles(angle_frames, num_frames)

        # Processing
        processed_rgb_frames = torch.stack([self.rgb_normalize(self.transform(frame)) for frame in rgb_frames])
        processed_mask_frames = torch.stack([self.mask_normalize(self.transform(frame)) for frame in mask_frames])

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return processed_rgb_frames, processed_mask_frames, coord_frames, angle_frames, label

    def _determine_num_frames(self, total_frames):
        """
        Dynamically determine the number of frames to sample based on video length.
        """
        if total_frames <= 150:
            return 30
        elif total_frames <= 300:
            return 60
        elif total_frames <= 1000:
            return 120
        elif total_frames <= 1500:
            return 240
        else:
            return 300

    def _load_video(self, video_path, is_mask=False):
        """
        Load video frames as a list of PIL images.
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if is_mask:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = Image.fromarray(frame)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
            frames.append(frame)
        cap.release()
        return frames

    def _load_coordinates(self, coord_file):
        """
        Load center-of-mass coordinates from a .txt file.
        """
        coords = []
        with open(coord_file, 'r') as f:
            for line in f:
                _, x, y = line.strip().split()
                coords.append(torch.tensor([float(x), float(y)], dtype=torch.float32))
        return coords

    def _load_angles(self, angle_file):
        """
        Load skeleton angle data from a JSON file.
        """
        with open(angle_file, 'r') as f:
            data = json.load(f)
            return [torch.tensor(frame, dtype=torch.float32) for frame in data['frames']]

    def _sample_frames(self, frames, num_frames):
        """
        Sample or pad frames to a fixed number.
        """
        total_frames = len(frames)
        if total_frames < num_frames:
            indices = list(range(total_frames)) + [total_frames - 1] * (num_frames - total_frames)
        else:
            indices = torch.linspace(0, total_frames - 1, num_frames).long()
        return [frames[i] for i in indices]

    def _sample_coords(self, coords, num_frames):
        """
        Sample or pad center-of-mass coordinates to a fixed number.
        """
        total_coords = len(coords)
        if total_coords < num_frames:
            indices = list(range(total_coords)) + [total_coords - 1] * (num_frames - total_coords)
        else:
            indices = torch.linspace(0, total_coords - 1, num_frames).long()
        return torch.stack([coords[i] for i in indices])

    def _sample_angles(self, angles, num_frames):
        """
        Sample or pad skeleton angles to a fixed number.
        """
        total_angles = len(angles)
        if total_angles < num_frames:
            indices = list(range(total_angles)) + [total_angles - 1] * (num_frames - total_angles)
        else:
            indices = torch.linspace(0, total_angles - 1, num_frames).long()
        return torch.stack([angles[i] for i in indices])
