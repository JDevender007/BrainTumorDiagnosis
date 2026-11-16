from __future__ import annotations

import torch
import torch.nn as nn


class ChannelAttention3D(nn.Module):
    def __init__(
        self,
        channels: int,
        reduction: int = 8,
    ) -> None:
        super().__init__()

        hidden = max(channels // reduction, 1)

        self.pool = nn.AdaptiveAvgPool3d(1)

        self.fc = nn.Sequential(
            nn.Conv3d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.fc(self.pool(x))
        return x * weights


class SpatialAttention3D(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv = nn.Conv3d(
            2,
            1,
            kernel_size=7,
            padding=3,
            bias=False,
        )

        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        maximum = torch.max(x, dim=1, keepdim=True).values

        combined = torch.cat(
            [avg, maximum],
            dim=1,
        )

        attention = self.activation(self.conv(combined))

        return x * attention


class CBAM3D(nn.Module):
    def __init__(
        self,
        channels: int,
        reduction: int = 8,
    ) -> None:
        super().__init__()

        self.channel = ChannelAttention3D(
            channels,
            reduction,
        )

        self.spatial = SpatialAttention3D()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel(x)
        return self.spatial(x)


if __name__ == "__main__":
    model = CBAM3D(8)

    x = torch.randn(
        1,
        8,
        16,
        16,
        16,
    )

    y = model(x)

    print("Input shape:", tuple(x.shape))
    print("Output shape:", tuple(y.shape))

    assert x.shape == y.shape

    print("Attention module test passed.")
