from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from database import FallDetectionVideoDataset
from net import FallDetectionModel
from torch.nn.utils.rnn import pad_sequence
import torch


def custom_collate_fn(batch):
    """
    Custom collate function to pad videos and return sequence lengths.
    Args:
        batch: List of tuples (rgb_frames, mask_frames, coord_frames, label).
    Returns:
        Padded tensors for RGB, masks, coordinates, labels, and sequence lengths.
    """
    rgb_frames, mask_frames, coord_frames, labels = zip(*batch)

    # Compute lengths of sequences before padding
    lengths = torch.tensor([x.shape[0] for x in rgb_frames])

    # Pad sequences to the length of the longest sequence in the batch
    rgb_frames = pad_sequence(rgb_frames, batch_first=True, padding_value=0)
    mask_frames = pad_sequence(mask_frames, batch_first=True, padding_value=0)
    coord_frames = pad_sequence(coord_frames, batch_first=True, padding_value=0)
    labels = torch.tensor(labels)

    return rgb_frames, mask_frames, coord_frames, labels, lengths

# Paths to dataset
train_dir = "../data/train"
val_dir = "../data/val"
test_dir = "../data/test"

# Training Dataset and DataLoader
train_dataset = FallDetectionVideoDataset(data_dir=train_dir, img_size=(224, 224))
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=custom_collate_fn
)

# Validation Dataset and DataLoader
val_dataset = FallDetectionVideoDataset(data_dir=val_dir, img_size=(224, 224))
val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=custom_collate_fn
)

# Initialize the model
model = FallDetectionModel(num_classes=2)

# Trainer with GPU acceleration
trainer = Trainer(
    max_epochs=10,
    accelerator="gpu",
    devices=1,
    log_every_n_steps=20
)

# Train the model with validation
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

# Testing Dataset and DataLoader
test_dataset = FallDetectionVideoDataset(data_dir=test_dir, img_size=(224, 224))
test_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=custom_collate_fn
)

# Test the model
trainer.test(model, test_loader)
