import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.models import resnet18
from torch.optim import Adam
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class FallDetectionModel(pl.LightningModule):
    def __init__(self, num_classes=2, rgb_cnn_output_dim=512, mask_cnn_output_dim=256, coord_output_dim=16, lstm_hidden_dim=128, num_lstm_layers=1):
        super(FallDetectionModel, self).__init__()
        
        # RGB branch
        self.rgb_cnn = resnet18(pretrained=True)
        self.rgb_cnn.fc = nn.Identity()  # Remove classification head
        
        # Binary mask branch
        self.mask_cnn = resnet18(pretrained=True)
        self.mask_cnn.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Adjust for single channel
        self.mask_cnn.fc = nn.Identity()

        # Projection layers for mask and coordinates
        self.mask_projection = nn.Linear(512, mask_cnn_output_dim)

        # Coordinate branch
        self.coord_fc = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, coord_output_dim)
        )

        # LSTMs for temporal modeling
        self.rgb_lstm = nn.LSTM(input_size=rgb_cnn_output_dim, hidden_size=lstm_hidden_dim, num_layers=num_lstm_layers, batch_first=True)
        self.mask_lstm = nn.LSTM(input_size=mask_cnn_output_dim, hidden_size=lstm_hidden_dim, num_layers=num_lstm_layers, batch_first=True)
        self.coord_lstm = nn.LSTM(input_size=coord_output_dim, hidden_size=lstm_hidden_dim, num_layers=num_lstm_layers, batch_first=True)

        # Fusion and classification layers
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 3, 128),  # Updated to account for 3 branches
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, rgb, mask, coords, lengths):
        """
        Args:
            rgb: Padded RGB frames [batch_size, max_frames, 3, height, width]
            mask: Padded binary mask frames [batch_size, max_frames, 1, height, width]
            coords: Padded coordinates [batch_size, max_frames, 2]
            lengths: Original sequence lengths [batch_size]
        Returns:
            logits: Output logits [batch_size, num_classes]
        """
        # Ensure lengths is a 1D CPU tensor of type int64
        lengths = lengths.cpu().to(torch.int64)

        batch_size, max_frames, _, height, width = rgb.shape

        # Process RGB frames
        rgb = rgb.view(batch_size * max_frames, 3, height, width)
        rgb_features = self.rgb_cnn(rgb)
        rgb_features = rgb_features.view(batch_size, max_frames, -1)

        # Process mask frames
        mask = mask.view(batch_size * max_frames, 1, height, width)
        mask_features = self.mask_cnn(mask)
        mask_features = self.mask_projection(mask_features)
        mask_features = mask_features.view(batch_size, max_frames, -1)

        # Process coordinates
        coord_features = self.coord_fc(coords)

        # Pack sequences to ignore padding
        packed_rgb = pack_padded_sequence(rgb_features, lengths, batch_first=True, enforce_sorted=False)
        packed_mask = pack_padded_sequence(mask_features, lengths, batch_first=True, enforce_sorted=False)
        packed_coords = pack_padded_sequence(coord_features, lengths, batch_first=True, enforce_sorted=False)

        # LSTM processing
        packed_rgb_out, _ = self.rgb_lstm(packed_rgb)
        packed_mask_out, _ = self.mask_lstm(packed_mask)
        packed_coords_out, _ = self.coord_lstm(packed_coords)

        # Unpack sequences
        rgb_lstm_out, _ = pad_packed_sequence(packed_rgb_out, batch_first=True)
        mask_lstm_out, _ = pad_packed_sequence(packed_mask_out, batch_first=True)
        coord_lstm_out, _ = pad_packed_sequence(packed_coords_out, batch_first=True)

        # Use the last valid hidden state for each sequence
        rgb_final = torch.stack([rgb_lstm_out[i, length - 1, :] for i, length in enumerate(lengths)])
        mask_final = torch.stack([mask_lstm_out[i, length - 1, :] for i, length in enumerate(lengths)])
        coord_final = torch.stack([coord_lstm_out[i, length - 1, :] for i, length in enumerate(lengths)])

        # Concatenate features and classify
        fused_features = torch.cat((rgb_final, mask_final, coord_final), dim=1)
        logits = self.fc(fused_features)
        return logits

    def training_step(self, batch, batch_idx):
        rgb_frames, mask_frames, coord_frames, labels, lengths = batch
        logits = self(rgb_frames, mask_frames, coord_frames, lengths)
        loss = self.criterion(logits, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        rgb, mask, coords, labels = batch
        logits = self(rgb, mask, coords)
        loss = self.criterion(logits, labels)
        self.log("val_loss", loss, prog_bar=True)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        """
        Test step to evaluate the model on the test dataset.
        Args:
            batch: A batch of test data including lengths.
            batch_idx: Index of the batch.
        """
        rgb, mask, coords, labels, lengths = batch  # Unpack lengths
        logits = self(rgb, mask, coords, lengths)  # Forward pass
        loss = self.criterion(logits, labels)

        # Predictions and accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

        return {"test_loss": loss, "test_acc": acc}
        
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)
