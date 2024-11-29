import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.models import resnet18
from torch.optim import Adam

class FallDetectionModel(pl.LightningModule):
    def __init__(self, num_classes=2, rgb_cnn_output_dim=512, mask_cnn_output_dim=256, lstm_hidden_dim=128, num_lstm_layers=1):
        super(FallDetectionModel, self).__init__()
        
        # RGB branch
        self.rgb_cnn = resnet18(pretrained=True)
        self.rgb_cnn.fc = nn.Identity()  # Remove classification head
        
        # Binary mask branch
        self.mask_cnn = resnet18(pretrained=True)
        self.mask_cnn.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Adjust for single channel
        self.mask_cnn.fc = nn.Identity()

        # Projection layer for mask branch to align features with LSTM input
        self.mask_projection = nn.Linear(512, mask_cnn_output_dim)

        # LSTM for temporal modeling
        self.rgb_lstm = nn.LSTM(input_size=rgb_cnn_output_dim, hidden_size=lstm_hidden_dim, num_layers=num_lstm_layers, batch_first=True)
        self.mask_lstm = nn.LSTM(input_size=mask_cnn_output_dim, hidden_size=lstm_hidden_dim, num_layers=num_lstm_layers, batch_first=True)

        # Fusion and classification layers
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, 128),  # Concatenating the features from both branches
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, rgb, mask):
        """
        Args:
            rgb: RGB frames of shape [batch_size, num_frames, 3, height, width]
            mask: Binary mask frames of shape [batch_size, num_frames, 1, height, width]
        Returns:
            logits: Output logits [batch_size, num_classes]
        """
        batch_size, num_frames, _, height, width = rgb.shape

        # Flatten frames for CNN processing
        rgb = rgb.view(batch_size * num_frames, 3, height, width)
        mask = mask.view(batch_size * num_frames, 1, height, width)

        # Extract features
        rgb_features = self.rgb_cnn(rgb)  # [batch_size * num_frames, rgb_cnn_output_dim]
        mask_features = self.mask_cnn(mask)  # [batch_size * num_frames, 512]

        # Apply projection to mask features
        mask_features = self.mask_projection(mask_features)  # [batch_size * num_frames, mask_cnn_output_dim]

        # Reshape back for LSTM processing
        rgb_features = rgb_features.view(batch_size, num_frames, -1)  # [batch_size, num_frames, rgb_cnn_output_dim]
        mask_features = mask_features.view(batch_size, num_frames, -1)  # [batch_size, num_frames, mask_cnn_output_dim]

        # LSTM processing
        rgb_lstm_out, _ = self.rgb_lstm(rgb_features)  # [batch_size, num_frames, lstm_hidden_dim]
        mask_lstm_out, _ = self.mask_lstm(mask_features)  # [batch_size, num_frames, lstm_hidden_dim]

        # Use the last hidden state from each LSTM
        rgb_final = rgb_lstm_out[:, -1, :]  # [batch_size, lstm_hidden_dim]
        mask_final = mask_lstm_out[:, -1, :]  # [batch_size, lstm_hidden_dim]

        # Concatenate features and classify
        fused_features = torch.cat((rgb_final, mask_final), dim=1)  # [batch_size, lstm_hidden_dim * 2]
        logits = self.fc(fused_features)  # [batch_size, num_classes]

        return logits

    def training_step(self, batch, batch_idx):
        rgb, mask, labels = batch  # Adjust the dataset to return RGB and mask pairs
        logits = self(rgb, mask)
        loss = self.criterion(logits, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        rgb, mask, labels = batch
        logits = self(rgb, mask)
        loss = self.criterion(logits, labels)
        self.log("val_loss", loss, prog_bar=True)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        rgb, mask, labels = batch
        logits = self(rgb, mask)
        loss = self.criterion(logits, labels)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return {"test_loss": loss, "test_acc": acc}
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)
