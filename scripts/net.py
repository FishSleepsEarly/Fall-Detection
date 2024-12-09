# net.py
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class FallDetectionModel(pl.LightningModule):
    def __init__(self, num_classes=2, angle_input_dim=3, lstm_hidden_dim=128, num_lstm_layers=1):
        """
        摔倒检测模型：仅用骨骼角度分支。
        
        Args:
            num_classes (int): 分类数量（摔倒与未摔倒）。
            angle_input_dim (int): 骨骼角度输入维度（4 表示左右腿角度和上半身角度等）。
            lstm_hidden_dim (int): LSTM 的隐藏层维度。
            num_lstm_layers (int): LSTM 的层数。
        """
        super(FallDetectionModel, self).__init__()

        # Skeleton angle feature extraction
        self.skeleton_angle_fc = nn.Sequential(
            nn.Linear(angle_input_dim, 16),  # 输入：4个角度值
            nn.ReLU(),
            nn.Linear(16, lstm_hidden_dim)
        )

        # LSTM for temporal modeling
        self.angle_lstm = nn.LSTM(
            input_size=lstm_hidden_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True
        )

        # Classification layer
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, skeleton_angles, lengths):
        """
        前向传播。
        
        Args:
            skeleton_angles: [batch_size, max_frames, angle_input_dim]，骨骼角度序列。
            lengths: [batch_size]，每个样本的有效序列长度。
        
        Returns:
            logits: 分类结果，形状为 [batch_size, num_classes]。
        """
        lengths = lengths.cpu().to(torch.int64)

        # Process skeleton angles
        skeleton_features = self.skeleton_angle_fc(skeleton_angles)

        # Pack sequences to ignore padding
        packed_angles = pack_padded_sequence(skeleton_features, lengths, batch_first=True, enforce_sorted=False)

        # LSTM processing
        packed_angles_out, _ = self.angle_lstm(packed_angles)
        angle_lstm_out, _ = pad_packed_sequence(packed_angles_out, batch_first=True)

        # Extract final outputs
        angle_final = torch.stack([
            angle_lstm_out[i, lengths[i] - 1, :] for i in range(len(lengths))
        ])

        # Classification
        logits = self.fc(angle_final)
        return logits

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        skeleton_angles, labels, lengths = batch
        logits = self(skeleton_angles, lengths)
        loss = nn.CrossEntropyLoss()(logits, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        skeleton_angles, labels, lengths = batch
        logits = self(skeleton_angles, lengths)
        loss = nn.CrossEntropyLoss()(logits, labels)
        self.log("val_loss", loss, prog_bar=True)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        skeleton_angles, labels, lengths = batch
        logits = self(skeleton_angles, lengths)
        loss = nn.CrossEntropyLoss()(logits, labels)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

        return {"test_loss": loss, "test_acc": acc}
        
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-4)
