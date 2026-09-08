from __future__ import annotations

import torch
import torch.nn as nn

class ResidualTCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        filters: int,
        kernel_size: int,
        dilation_rate: int,
        dropout: float,
    ):
        super().__init__()
        padding = ((kernel_size - 1) * dilation_rate) // 2

        self.conv1 = nn.Conv1d(
            in_channels,
            filters,
            kernel_size,
            dilation=dilation_rate,
            padding=padding,
        )
        self.conv2 = nn.Conv1d(
            filters,
            filters,
            kernel_size,
            dilation=dilation_rate,
            padding=padding,
        )
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.shortcut = (
            nn.Conv1d(in_channels, filters, kernel_size=1)
            if in_channels != filters
            else nn.Identity()
        )

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.dropout(self.relu(self.conv1(x)))
        x = self.dropout(self.relu(self.conv2(x)))
        return self.relu(x + shortcut)

class PointwiseTCN(nn.Module):
    def __init__(
        self,
        n_features: int,
        filters: int,
        kernel_size: int,
        dilations: list[int],
        dense_units: int,
        dropout: float,
    ):
        super().__init__()
        blocks: list[nn.Module] = []
        in_channels = n_features

        for dilation_rate in dilations:
            blocks.append(
                ResidualTCNBlock(
                    in_channels=in_channels,
                    filters=filters,
                    kernel_size=kernel_size,
                    dilation_rate=dilation_rate,
                    dropout=dropout,
                )
            )
            in_channels = filters

        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Conv1d(filters, dense_units, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(dense_units, 1, kernel_size=1),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.tcn(x)
        return self.head(x).squeeze(1)

def count_trainable_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
