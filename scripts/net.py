import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.models import resnet18
from torch.optim import Adam
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchmetrics import Accuracy, Precision, Recall, F1Score

class FallDetectionModel(pl.LightningModule):
    def __init__(self, num_classes=2, rgb_cnn_output_dim=512, mask_cnn_output_dim=256, coord_output_dim=16, lstm_hidden_dim=128, num_lstm_layers=1):
        super(FallDetectionModel, self).__init__()
        
        # RGB branch
        self.rgb_cnn = resnet18(weights='IMAGENET1K_V1')
        self.rgb_cnn.fc = nn.Identity()  # Remove classification head
        
        # Binary mask branch
        self.mask_cnn = resnet18(weights='IMAGENET1K_V1')
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

        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_precision = Precision(task="multiclass", num_classes=num_classes)
        self.val_recall = Recall(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_precision = Precision(task="multiclass", num_classes=num_classes)
        self.test_recall = Recall(task="multiclass", num_classes=num_classes)
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes)

    def forward(self, rgb, mask, coords, lengths):
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

        # Pack sequences
        packed_rgb = pack_padded_sequence(rgb_features, lengths, batch_first=True, enforce_sorted=False)
        packed_mask = pack_padded_sequence(mask_features, lengths, batch_first=True, enforce_sorted=False)
        packed_coords = pack_padded_sequence(coord_features, lengths, batch_first=True, enforce_sorted=False)

        # LSTM processing
        _, (rgb_out, _) = self.rgb_lstm(packed_rgb)
        _, (mask_out, _) = self.mask_lstm(packed_mask)
        _, (coord_out, _) = self.coord_lstm(packed_coords)

        # Concatenate LSTM outputs
        fused_features = torch.cat((rgb_out[-1], mask_out[-1], coord_out[-1]), dim=1)
        logits = self.fc(fused_features)
        return logits

    def training_step(self, batch, batch_idx):
        rgb, mask, coords, labels, lengths = batch
        logits = self(rgb, mask, coords, lengths)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, labels)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        rgb, mask, coords, labels, lengths = batch
        logits = self(rgb, mask, coords, lengths)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)

        self.val_acc(preds, labels)
        self.val_precision(preds, labels)
        self.val_recall(preds, labels)
        self.val_f1(preds, labels)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)
        self.log("val_precision", self.val_precision, prog_bar=True)
        self.log("val_recall", self.val_recall, prog_bar=True)
        self.log("val_f1", self.val_f1, prog_bar=True)

    def test_step(self, batch, batch_idx):
        rgb, mask, coords, labels, lengths = batch
        logits = self(rgb, mask, coords, lengths)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)

        self.test_acc(preds, labels)
        self.test_precision(preds, labels)
        self.test_recall(preds, labels)
        self.test_f1(preds, labels)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_acc, prog_bar=True)
        self.log("test_precision", self.test_precision, prog_bar=True)
        self.log("test_recall", self.test_recall, prog_bar=True)
        self.log("test_f1", self.test_f1, prog_bar=True)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)
