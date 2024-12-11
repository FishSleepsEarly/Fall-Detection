import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from database import FallDetectionVideoDataset
from net import FallDetectionModel
from torch.nn.utils.rnn import pad_sequence


def custom_collate_fn(batch):
    """
    Custom collate function to pad videos, masks, coordinates, angles,
    and return sequence lengths.
    """
    rgb_frames, mask_frames, coord_frames, angle_frames, labels = zip(*batch)

    # Compute lengths of sequences before padding
    lengths = torch.tensor([x.shape[0] for x in rgb_frames])

    # Pad sequences to the length of the longest sequence in the batch
    rgb_frames = pad_sequence(rgb_frames, batch_first=True, padding_value=0)
    mask_frames = pad_sequence(mask_frames, batch_first=True, padding_value=0)
    coord_frames = pad_sequence(coord_frames, batch_first=True, padding_value=0)
    angle_frames = pad_sequence(angle_frames, batch_first=True, padding_value=0)
    labels = torch.tensor(labels)

    return rgb_frames, mask_frames, coord_frames, angle_frames, labels, lengths


def main():
    # Path to the test data and the checkpoint
    test_dir = "../data/test"
    checkpoint_path = "./lightning_logs/version_f/checkpoints/epoch=9-step=170.ckpt"  # Replace with the actual path

    # Load the test dataset and DataLoader
    test_dataset = FallDetectionVideoDataset(data_dir=test_dir, img_size=(224, 224))
    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=custom_collate_fn
    )

    # Load the trained model from checkpoint
    model = FallDetectionModel.load_from_checkpoint(checkpoint_path, num_classes=2)
    print("Checkpoint loaded successfully!")

    # Initialize the Trainer for testing
    trainer = Trainer(accelerator="gpu", devices=1)

    # Perform testing
    print("Starting testing...")
    trainer.test(model, dataloaders=test_loader)


if __name__ == "__main__":
    main()
