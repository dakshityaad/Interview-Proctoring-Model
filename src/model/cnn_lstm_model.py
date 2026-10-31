"""CNN+LSTM classifier for temporal face-feature windows."""

import torch
from torch import nn


class CNNLSTMClassifier(nn.Module):
    """Apply local temporal convolutions, then model sequence context with an LSTM."""

    def __init__(
        self,
        input_features=3,
        num_classes=3,
        cnn_channels=32,
        lstm_hidden_size=64,
        lstm_layers=1,
        dropout=0.2,
    ):
        if input_features <= 0 or num_classes <= 0:
            raise ValueError("input_features and num_classes must be positive")
        if lstm_layers <= 0 or lstm_hidden_size <= 0:
            raise ValueError("LSTM dimensions must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in the range [0, 1)")

        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_features, cnn_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        lstm_dropout = dropout if lstm_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_size, num_classes),
        )

    def forward(self, inputs):
        if inputs.ndim != 3:
            raise ValueError("CNN+LSTM inputs must have shape (batch, frames, features)")
        # Conv1d expects (batch, channels, frames), while LSTM expects
        # (batch, frames, channels).
        cnn_features = self.cnn(inputs.transpose(1, 2)).transpose(1, 2)
        _, (hidden_state, _) = self.lstm(cnn_features)
        logits = self.classifier(hidden_state[-1])
        return torch.softmax(logits, dim=1)


def build_model(input_features=3, num_classes=3, dropout=0.2):
    """Build the default three-class CNN+LSTM classifier."""
    return CNNLSTMClassifier(
        input_features=input_features,
        num_classes=num_classes,
        dropout=dropout,
    )
