"""DNN+LSTM classifier for temporal face-feature windows."""

import torch
from torch import nn


class DNNLSTMClassifier(nn.Module):
    """Project each frame with dense layers, then model sequence context with an LSTM."""

    def __init__(
        self,
        input_features=3,
        frames=38,
        num_classes=3,
        frame_hidden_size=64,
        lstm_hidden_size=64,
        dropout=0.2,
    ):
        if input_features <= 0 or frames <= 0 or num_classes <= 0:
            raise ValueError("input_features, frames, and num_classes must be positive")
        if frame_hidden_size <= 0 or lstm_hidden_size <= 0:
            raise ValueError("DNN and LSTM dimensions must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in the range [0, 1)")

        super().__init__()
        self.frames = frames
        self.input_features = input_features
        self.frame_projection = nn.Sequential(
            nn.Linear(input_features, frame_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(frame_hidden_size, frame_hidden_size),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=frame_hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, inputs):
        if inputs.ndim != 3:
            raise ValueError("DNN+LSTM inputs must have shape (batch, frames, features)")
        if tuple(inputs.shape[1:]) != (self.frames, self.input_features):
            raise ValueError(
                f"Expected window shape {(self.frames, self.input_features)}, "
                f"received {tuple(inputs.shape[1:])}"
            )
        projected = self.frame_projection(inputs)
        _, (hidden_state, _) = self.lstm(projected)
        logits = self.classifier(hidden_state[-1])
        return torch.softmax(logits, dim=1)


def build_model(input_features=3, frames=38, num_classes=3, dropout=0.2):
    """Build the default three-class DNN+LSTM classifier."""
    return DNNLSTMClassifier(
        input_features=input_features,
        frames=frames,
        num_classes=num_classes,
        dropout=dropout,
    )
