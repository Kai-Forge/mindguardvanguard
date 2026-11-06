"""Dual-channel 2D-CNN + BiLSTM strict-valence classifier."""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .config import CNN_DROPOUT


class ChannelLayerNorm2d(nn.Module):
    """Layer normalization over channels at each time-frequency position."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class SERModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 3,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        input_channels: int = 2,
        normalization: str = "batch",
    ):
        super().__init__()
        if normalization not in {"batch", "group", "channel_layer"}:
            raise ValueError("normalization must be 'batch', 'group', or 'channel_layer'")
        if normalization == "batch":
            norm = lambda channels: nn.BatchNorm2d(channels)
        elif normalization == "group":
            norm = lambda channels: nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        else:
            norm = lambda channels: ChannelLayerNorm2d(channels)
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1), norm(32), nn.ReLU(inplace=True),
            nn.Dropout2d(CNN_DROPOUT), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), norm(64), nn.ReLU(inplace=True),
            nn.Dropout2d(CNN_DROPOUT), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), norm(128), nn.ReLU(inplace=True),
            nn.Dropout2d(CNN_DROPOUT), nn.MaxPool2d((2, 1)),
            nn.Conv2d(128, 256, 3, padding=1), norm(256), nn.ReLU(inplace=True),
            nn.Dropout2d(CNN_DROPOUT),
        )
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.5,
        )
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

    @staticmethod
    def _pool_time_lengths(lengths: torch.Tensor, layer: nn.MaxPool2d) -> torch.Tensor:
        """Map valid frame counts through one pooling layer's time dimension."""
        kernel = layer.kernel_size[1] if isinstance(layer.kernel_size, tuple) else layer.kernel_size
        stride_value = layer.stride if layer.stride is not None else layer.kernel_size
        stride = stride_value[1] if isinstance(stride_value, tuple) else stride_value
        padding = layer.padding[1] if isinstance(layer.padding, tuple) else layer.padding
        dilation = layer.dilation[1] if isinstance(layer.dilation, tuple) else layer.dilation
        return torch.div(
            lengths + 2 * padding - dilation * (kernel - 1) - 1,
            stride,
            rounding_mode="floor",
        ) + 1

    def cnn_time_lengths(self, frame_lengths: torch.Tensor) -> list[torch.Tensor]:
        """Return valid time lengths after every temporal/spatial pooling layer."""
        lengths = frame_lengths.clone().to(torch.long)
        outputs = []
        for layer in self.cnn:
            if isinstance(layer, nn.MaxPool2d):
                lengths = self._pool_time_lengths(lengths, layer).clamp(min=1)
                outputs.append(lengths.clone())
        return outputs

    def forward(
        self,
        x: torch.Tensor,
        frame_lengths: torch.Tensor | None = None,
        mask_cnn_padding: bool = False,
    ):
        lengths = frame_lengths.clone().to(torch.long) if frame_lengths is not None else None
        if mask_cnn_padding and lengths is not None:
            for layer in self.cnn:
                x = layer(x)
                if isinstance(layer, nn.MaxPool2d):
                    lengths = self._pool_time_lengths(lengths, layer)
                    lengths = lengths.clamp(min=1, max=x.shape[-1])
                steps = torch.arange(x.shape[-1], device=x.device).view(1, 1, 1, -1)
                valid = steps < lengths.to(x.device).view(-1, 1, 1, 1)
                x = x.masked_fill(~valid, 0.0)
        else:
            x = self.cnn(x)
        x = x.mean(dim=2).transpose(1, 2)
        if frame_lengths is None:
            x, _ = self.lstm(x)
            last = x[:, -1]
            maximum = x.max(dim=1).values
        else:
            if not mask_cnn_padding:
                lengths = torch.div(frame_lengths, 4, rounding_mode="floor")
            lengths = lengths.clamp(min=1, max=x.shape[1]).to(torch.long)
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_x, _ = self.lstm(packed)
            x, _ = pad_packed_sequence(packed_x, batch_first=True, total_length=x.shape[1])
            device_lengths = lengths.to(x.device)
            rows = torch.arange(x.shape[0], device=x.device)
            last = x[rows, device_lengths - 1]
            steps = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
            valid = steps < device_lengths.unsqueeze(1)
            maximum = x.masked_fill(~valid.unsqueeze(-1), float("-inf")).max(dim=1).values
        return self.classifier(last + maximum)
