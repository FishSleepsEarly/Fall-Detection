import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.models import resnet18
from torch.optim import Adam

class FallDetectionModel(pl.LightningModule):
    def __init__(self, num_classes=2, cnn_output_dim=512, lstm_hidden_dim=128, num_lstm_layers=1):
        super(FallDetectionModel, self).__init__()
        # Pretrained ResNet for feature extraction
        self.cnn = resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()  # Remove classification head, we only need features
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=cnn_output_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True
        )
        
        # Fully connected layer for classification
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, num_frames, channels, height, width]
        Returns:
            logits: Tensor of shape [batch_size, num_classes]
        """
        batch_size, num_frames, channels, height, width = x.shape
        
        # Flatten frames into a single batch for CNN processing
        x = x.view(batch_size * num_frames, channels, height, width)  # [batch_size * num_frames, channels, height, width]
        
        # Extract spatial features using CNN
        cnn_features = self.cnn(x)  # [batch_size * num_frames, cnn_output_dim]
        
        # Reshape back to [batch_size, num_frames, cnn_output_dim]
        cnn_features = cnn_features.view(batch_size, num_frames, -1)
        
        # Pass through LSTM
        lstm_out, (hn, cn) = self.lstm(cnn_features)  # [batch_size, num_frames, lstm_hidden_dim]
        
        # Use the last hidden state from LSTM
        final_features = lstm_out[:, -1, :]  # [batch_size, lstm_hidden_dim]
        
        # Pass through the fully connected layers for classification
        logits = self.fc(final_features)  # [batch_size, num_classes]
        return logits

    def training_step(self, batch, batch_idx):
        videos, labels = batch  # videos: [batch_size, num_frames, channels, height, width]
        logits = self(videos)
        loss = self.criterion(logits, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        videos, labels = batch
        logits = self(videos)
        loss = self.criterion(logits, labels)
        self.log("val_loss", loss, prog_bar=True)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        """
        Defines how to process a single batch during testing.
        """
        videos, labels = batch
        logits = self(videos)
        loss = self.criterion(logits, labels)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        
        # Log metrics
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        
        return {"test_loss": loss, "test_acc": acc}
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)
