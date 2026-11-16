from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn
from torchvision import models as _torchvision_models


ModelType = Literal[
    "segmentation",
    "classification",
    "hybrid",
]


@dataclass
class ModelConfig:
    model_type: ModelType = "classification"

    in_channels: int = 4
    num_classes: int = 4

    segmentation_classes: int = 2

    base_channels: int = 16
    dropout: float = 0.2

    classification_input_size: tuple[int, int] = (224, 224)

    classification_backbone: str = "cnn2d"

    pretrained: bool = False


class ConvBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SegmentationModel3D(nn.Module):
    """
    Lightweight 3D segmentation network for MRI tumor localization.

    Input:
        [B, 4, D, H, W]

    Output:
        [B, segmentation_classes, D, H, W]
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 2,
        base_channels: int = 16,
    ) -> None:
        super().__init__()

        self.encoder1 = ConvBlock3D(
            in_channels,
            base_channels,
        )

        self.pool1 = nn.MaxPool3d(2)

        self.encoder2 = ConvBlock3D(
            base_channels,
            base_channels * 2,
        )

        self.pool2 = nn.MaxPool3d(2)

        self.bottleneck = ConvBlock3D(
            base_channels * 2,
            base_channels * 4,
        )

        self.decoder2 = nn.ConvTranspose3d(
            base_channels * 4,
            base_channels * 2,
            kernel_size=2,
            stride=2,
        )

        self.decode_block2 = ConvBlock3D(
            base_channels * 4,
            base_channels * 2,
        )

        self.decoder1 = nn.ConvTranspose3d(
            base_channels * 2,
            base_channels,
            kernel_size=2,
            stride=2,
        )

        self.decode_block1 = ConvBlock3D(
            base_channels * 2,
            base_channels,
        )

        self.output = nn.Conv3d(
            base_channels,
            num_classes,
            kernel_size=1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        e1 = self.encoder1(x)

        e2 = self.encoder2(self.pool1(e1))

        b = self.bottleneck(self.pool2(e2))

        d2 = self.decoder2(b)

        if d2.shape[2:] != e2.shape[2:]:
            d2 = nn.functional.interpolate(
                d2,
                size=e2.shape[2:],
                mode="trilinear",
                align_corners=False,
            )

        d2 = torch.cat(
            [d2, e2],
            dim=1,
        )

        d2 = self.decode_block2(d2)

        d1 = self.decoder1(d2)

        if d1.shape[2:] != e1.shape[2:]:
            d1 = nn.functional.interpolate(
                d1,
                size=e1.shape[2:],
                mode="trilinear",
                align_corners=False,
            )

        d1 = torch.cat(
            [d1, e1],
            dim=1,
        )

        d1 = self.decode_block1(d1)

        return self.output(d1)


class ResidualBlock2D(nn.Module):
    """Residual block with pre-activation BatchNorm."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class ClassificationModel2D(nn.Module):
    """
    Improved 2D CNN for brain tumor type classification.
    Uses residual connections, batch norm, and increased capacity.

    Input:
        [B, C, H, W]

    Output:
        [B, num_classes]
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 4,
        base_channels: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(base_channels, base_channels * 2, num_blocks=3, stride=1)
        self.layer2 = self._make_layer(base_channels * 2, base_channels * 4, num_blocks=4, stride=2)
        self.layer3 = self._make_layer(base_channels * 4, base_channels * 8, num_blocks=6, stride=2)
        self.layer4 = self._make_layer(base_channels * 8, base_channels * 16, num_blocks=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(base_channels * 16, base_channels * 8),
            nn.BatchNorm1d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(base_channels * 8, num_classes),
        )

        self._init_weights()

    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        layers = []
        layers.append(ResidualBlock2D(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock2D(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


class HybridModel(nn.Module):
    """
    Combined model for the project pipeline.

    Segmentation:
        3D MRI -> tumor localization

    Classification:
        2D MRI slice -> tumor type prediction
    """

    def __init__(
        self,
        config: ModelConfig,
    ) -> None:
        super().__init__()

        self.segmentation_model = SegmentationModel3D(
            in_channels=config.in_channels,
            num_classes=config.segmentation_classes,
            base_channels=config.base_channels,
        )

        if config.classification_backbone == "cnn2d":
            self.classification_model = ClassificationModel2D(
                in_channels=config.in_channels,
                num_classes=config.num_classes,
                base_channels=config.base_channels,
                dropout=config.dropout,
            )
        else:
            self.classification_model = create_backbone_classifier(
                backbone=config.classification_backbone,
                num_classes=config.num_classes,
                pretrained=config.pretrained,
                dropout=config.dropout,
            )

    def forward(
        self,
        volume: torch.Tensor,
        image: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:

        segmentation_logits = self.segmentation_model(volume)

        classification_logits = None

        if image is not None:
            classification_logits = self.classification_model(image)

        return {
            "segmentation": segmentation_logits,
            "classification": classification_logits,
        }


class ModelFactory:
    """
    Factory responsible for creating project models.
    """

    @staticmethod
    def create(
        config: ModelConfig,
    ) -> nn.Module:

        if config.model_type == "segmentation":
            return create_segmentation_model(config)

        if config.model_type == "classification":
            return create_classification_model(config)

        if config.model_type == "hybrid":
            return create_hybrid_model(config)

        raise ValueError(f"Unsupported model type: {config.model_type}")


def create_segmentation_model(
    config: ModelConfig | None = None,
) -> nn.Module:

    if config is None:
        config = ModelConfig(model_type="segmentation")

    return SegmentationModel3D(
        in_channels=config.in_channels,
        num_classes=config.segmentation_classes,
        base_channels=config.base_channels,
    )


def create_classification_model(
    config: ModelConfig | None = None,
) -> nn.Module:

    if config is None:
        config = ModelConfig(model_type="classification")

    return ClassificationModel2D(
        in_channels=config.in_channels,
        num_classes=config.num_classes,
        base_channels=config.base_channels,
        dropout=config.dropout,
    )


BACKBONE_CLASSIFIERS: dict[str, dict[str, Any]] = {
    "densenet121": {
        "builder": lambda pretrained: (
            _torchvision_models.densenet121(
                weights=(_torchvision_models.DenseNet121_Weights.DEFAULT if pretrained else None),
            )
        ),
        "head_attr": "classifier",
        "head_out_features": 1024,
    },
    "resnet50": {
        "builder": lambda pretrained: (
            _torchvision_models.resnet50(
                weights=(_torchvision_models.ResNet50_Weights.DEFAULT if pretrained else None),
            )
        ),
        "head_attr": "fc",
        "head_out_features": 2048,
    },
    "efficientnet_v2_s": {
        "builder": lambda pretrained: (
            _torchvision_models.efficientnet_v2_s(
                weights=(
                    _torchvision_models.EfficientNet_V2_S_Weights.DEFAULT
                    if pretrained
                    else None
                ),
            )
        ),
        "head_attr": "classifier",
        "head_out_features": 1280,
    },
    "convnext_tiny": {
        "builder": lambda pretrained: (
            _torchvision_models.convnext_tiny(
                weights=(
                    _torchvision_models.ConvNeXt_Tiny_Weights.DEFAULT
                    if pretrained
                    else None
                ),
            )
        ),
        "head_attr": "classifier",
        "head_out_features": 768,
    },
    "swin_t": {
        "builder": lambda pretrained: (
            _torchvision_models.swin_t(
                weights=(
                    _torchvision_models.Swin_T_Weights.DEFAULT
                    if pretrained
                    else None
                ),
            )
        ),
        "head_attr": "head",
        "head_out_features": 768,
    },
    "vit_b_16": {
        "builder": lambda pretrained: (
            _torchvision_models.vit_b_16(
                weights=(
                    _torchvision_models.ViT_B_16_Weights.DEFAULT
                    if pretrained
                    else None
                ),
            )
        ),
        "head_attr": "heads",
        "head_out_features": 768,
    },
}


def create_backbone_classifier(
    backbone: str,
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.30,
) -> nn.Module:
    """
    Create a 2D classifier from a torchvision backbone.

    Supported backbones:

        densenet121, resnet50, efficientnet_v2_s,
        convnext_tiny, swin_t, vit_b_16

    The final Linear layer of the classification head is replaced
    with ``Sequential(Dropout, Linear)`` so the architecture matches
    the checkpoints produced by the unified trainer. Pooling /
    normalization layers inside the head are preserved.
    """

    if backbone not in BACKBONE_CLASSIFIERS:
        raise ValueError(
            f"Unsupported backbone: {backbone}. "
            f"Supported: {sorted(BACKBONE_CLASSIFIERS)}"
        )

    specification = BACKBONE_CLASSIFIERS[backbone]

    model = specification["builder"](pretrained)

    head = getattr(
        model,
        specification["head_attr"],
    )

    new_head = _replace_final_linear(
        head,
        num_classes,
        dropout,
    )

    setattr(
        model,
        specification["head_attr"],
        new_head,
    )

    return model


def _replace_final_linear(
    head: nn.Module,
    num_classes: int,
    dropout: float,
) -> nn.Module:
    """
    Replace the final Linear layer of a classification head with
    ``Sequential(Dropout, Linear)``.
    """

    if isinstance(head, nn.Sequential):
        linear_index = None
        in_features = None

        for index, module in enumerate(head):
            if isinstance(module, nn.Linear):
                linear_index = index
                in_features = module.in_features

        if linear_index is None:
            raise ValueError("Classification head contains no Linear layer.")

        replacement: list[nn.Module] = []

        for module in head[:linear_index]:
            replacement.append(module)

        replacement.append(
            nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(
                    in_features,
                    num_classes,
                ),
            )
        )

        return nn.Sequential(*replacement)

    return nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(
            head.in_features,
            num_classes,
        ),
    )


def create_hybrid_model(
    config: ModelConfig | None = None,
) -> nn.Module:

    if config is None:
        config = ModelConfig(model_type="hybrid")

    return HybridModel(config)


def create_model(
    config: ModelConfig | None = None,
) -> nn.Module:

    if config is None:
        config = ModelConfig()

    return ModelFactory.create(config)


def count_parameters(
    model: nn.Module,
) -> int:

    return sum(parameter.numel() for parameter in model.parameters())


def count_trainable_parameters(
    model: nn.Module,
) -> int:

    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def freeze_model(
    model: nn.Module,
) -> None:

    for parameter in model.parameters():
        parameter.requires_grad = False


def unfreeze_model(
    model: nn.Module,
) -> None:

    for parameter in model.parameters():
        parameter.requires_grad = True


def model_summary(
    model: nn.Module,
) -> dict[str, int]:

    return {
        "total_parameters": count_parameters(model),
        "trainable_parameters": count_trainable_parameters(model),
        "non_trainable_parameters": (
            count_parameters(model) - count_trainable_parameters(model)
        ),
    }


if __name__ == "__main__":

    print("=" * 78)
    print("BrainTumorDiagnosis Model Factory Test")
    print("=" * 78)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    segmentation_config = ModelConfig(
        model_type="segmentation",
        in_channels=4,
        segmentation_classes=2,
        base_channels=8,
    )

    classification_config = ModelConfig(
        model_type="classification",
        in_channels=4,
        num_classes=4,
        base_channels=16,
    )

    hybrid_config = ModelConfig(
        model_type="hybrid",
        in_channels=4,
        num_classes=4,
        segmentation_classes=2,
        base_channels=8,
    )

    segmentation_model = create_segmentation_model(segmentation_config).to(device)

    classification_model = create_classification_model(classification_config).to(device)

    hybrid_model = create_hybrid_model(hybrid_config).to(device)

    print(
        "Segmentation parameters:",
        count_parameters(segmentation_model),
    )

    print(
        "Classification parameters:",
        count_parameters(classification_model),
    )

    print(
        "Hybrid parameters:",
        count_parameters(hybrid_model),
    )

    with torch.no_grad():

        volume = torch.randn(
            1,
            4,
            32,
            32,
            32,
            device=device,
        )

        image = torch.randn(
            1,
            4,
            224,
            224,
            device=device,
        )

        segmentation_output = segmentation_model(volume)

        classification_output = classification_model(image)

        hybrid_output = hybrid_model(
            volume,
            image,
        )

    print(
        "Segmentation output:",
        tuple(segmentation_output.shape),
    )

    print(
        "Classification output:",
        tuple(classification_output.shape),
    )

    print(
        "Hybrid segmentation output:",
        tuple(hybrid_output["segmentation"].shape),
    )

    print(
        "Hybrid classification output:",
        tuple(hybrid_output["classification"].shape),
    )

    print("=" * 78)
    print("Model Factory Test Passed")
    print("=" * 78)
