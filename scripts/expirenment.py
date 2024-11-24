from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from database import FallDetectionVideoDataset
from torch.utils.data import DataLoader
from net import FallDetectionModel

# Paths to the dataset
train_dir = "../data/train"
test_dir = "../data/test"

# Dataset and DataLoader
train_dataset = FallDetectionVideoDataset(train_dir, img_size=(224, 224), num_frames=16)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Initialize Model
model = FallDetectionModel(num_classes=2)

# Trainer without validation
trainer = Trainer(
    max_epochs=10,
    accelerator="gpu",  # Use GPU or set to "cpu" for CPU
    devices=1,          # Number of GPUs or CPUs
    log_every_n_steps=20  # Replace deprecated progress_bar_refresh_rate
)

# Train the model
trainer.fit(model, train_loader)


# Test Dataset and DataLoader
test_dataset = FallDetectionVideoDataset(test_dir, img_size=(224, 224), num_frames=16)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Test the model
trainer.test(model, test_loader)