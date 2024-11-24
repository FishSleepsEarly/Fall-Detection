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
        
        # List all video paths and labels
        self.video_paths = []
        self.labels = []
        
        for label, category in enumerate(["not_fall", "fall"]):  # 0: not_fall, 1: fall
            category_path = os.path.join(data_dir, category, "raw_videos")
            if not os.path.exists(category_path):
                raise ValueError(f"Path {category_path} does not exist.")
            
            for video in os.listdir(category_path):
                if video.endswith(".mp4"):  # Only process .mp4 files
                    self.video_paths.append(os.path.join(category_path, video))
                    self.labels.append(label)
        
        # Default transform if none provided
        self.transform = transform or transforms.Compose([
            transforms.Resize(self.img_size),  # Resize to (224, 224)
            transforms.ToTensor(),            # Convert to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
        ])

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        # Load video
        video_path = self.video_paths[idx]
        frames = self._load_video(video_path)
        
        # Sample frames
        frames = self._sample_frames(frames, self.num_frames)
        
        # Apply transformations to each frame
        processed_frames = torch.stack([self.transform(frame) for frame in frames])
        
        # Get label
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return processed_frames, label

    def _load_video(self, video_path):
        """
        Load frames from a video file.
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            frame = Image.fromarray(frame)  # Convert to PIL image
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