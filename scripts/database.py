import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2

class FallDetectionVideoDataset(Dataset):
    def __init__(self, data_dir, img_size=(224, 224), num_frames=16, transform=None):
        """
        Args:
            data_dir (str): Path to the directory containing train or test data.
            img_size (tuple): Desired image size (height, width) for resizing.
            num_frames (int): Number of frames to sample from each video.
            transform (torchvision.transforms): Transformations to apply to the frames.
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.num_frames = num_frames
        self.video_paths = []
        self.mask_paths = []
        self.labels = []
        
        for label, category in enumerate(["not_fall", "fall"]):  # 0: not_fall, 1: fall
            category_path = os.path.join(data_dir, category)
            raw_videos_path = os.path.join(category_path, "raw_videos")
            mask_videos_path = os.path.join(category_path, "mask_videos")
            
            if not os.path.exists(raw_videos_path) or not os.path.exists(mask_videos_path):
                raise ValueError(f"Missing directories for {category}: {raw_videos_path} or {mask_videos_path}")
            
            for video in os.listdir(raw_videos_path):
                if video.endswith(".mp4"):  # Assuming video files are .mp4
                    raw_video_path = os.path.join(raw_videos_path, video)
                    mask_video_path = os.path.join(mask_videos_path, video)  # Matching mask video by name
                    if not os.path.exists(mask_video_path):
                        raise ValueError(f"Missing corresponding mask video for {raw_video_path}")
                    
                    self.video_paths.append(raw_video_path)
                    self.mask_paths.append(mask_video_path)
                    self.labels.append(label)
        
        # Default transform if none provided
        self.transform = transform or transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
        ])
        # Normalize RGB differently than binary masks
        self.rgb_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.mask_normalize = transforms.Normalize(mean=[0.5], std=[0.5])

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        # Load RGB video and mask video
        raw_video_path = self.video_paths[idx]
        mask_video_path = self.mask_paths[idx]
        
        rgb_frames = self._load_video(raw_video_path)
        mask_frames = self._load_video(mask_video_path, is_mask=True)
        
        # Sample frames
        rgb_frames = self._sample_frames(rgb_frames, self.num_frames)
        mask_frames = self._sample_frames(mask_frames, self.num_frames)
        
        # Apply transformations
        processed_rgb_frames = torch.stack([self.rgb_normalize(self.transform(frame)) for frame in rgb_frames])
        processed_mask_frames = torch.stack([self.mask_normalize(self.transform(frame)) for frame in mask_frames])
        
        # Get label
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return processed_rgb_frames, processed_mask_frames, label

    def _load_video(self, video_path, is_mask=False):
        """
        Load frames from a video file.
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if is_mask:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert mask to grayscale
                frame = Image.fromarray(frame)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                frame = Image.fromarray(frame)
            frames.append(frame)
        cap.release()
        return frames

    def _sample_frames(self, frames, num_frames):
        """
        Uniformly sample a fixed number of frames from the video.
        """
        total_frames = len(frames)
        if total_frames < num_frames:
            indices = list(range(total_frames)) + [total_frames - 1] * (num_frames - total_frames)
        else:
            indices = torch.linspace(0, total_frames - 1, num_frames).long()
        return [frames[i] for i in indices]

'''
from torch.utils.data import DataLoader

# Dataset paths
train_dir = "../data/train"
test_dir = "../data/test"

# Create datasets
train_dataset = FallDetectionVideoDataset(train_dir, img_size=(224, 224), num_frames=16)
test_dataset = FallDetectionVideoDataset(test_dir, img_size=(224, 224), num_frames=16)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Iterate through the dataloader
for videos, labels in train_loader:
    print(videos.shape)  # Example: torch.Size([4, 16, 3, 224, 224]) for batch size 4
    print(labels)        # Example: tensor([0, 1, 0, 1]) (labels for the batch)
'''