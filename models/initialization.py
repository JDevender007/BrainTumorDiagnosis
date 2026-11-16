from __future__ import annotations

import torch.nn as nn


def initialize_weights(
    model: nn.Module,
) -> nn.Module:

    for module in model.modules():

        if isinstance(
            module,
            (
                nn.Conv3d,
                nn.Conv2d,
                nn.ConvTranspose3d,
            ),
        ):
            nn.init.kaiming_normal_(
                module.weight,
                mode="fan_out",
                nonlinearity="relu",
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(
            module,
            (
                nn.BatchNorm3d,
                nn.InstanceNorm3d,
            ),
        ):
            if module.weight is not None:
                nn.init.ones_(module.weight)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(
            module,
            nn.Linear,
        ):
            nn.init.trunc_normal_(
                module.weight,
                std=0.02,
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

    return model


if __name__ == "__main__":
    model = nn.Sequential(
        nn.Conv3d(
            4,
            8,
            3,
            padding=1,
        ),
        nn.ReLU(),
        nn.Linear(8, 4),
    )

    initialize_weights(model)

    print("Weight initialization test passed.")
