| Model / configuration | Seeds | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---|---|
| Leaky-split control (reproduces the historical split) | 1 | 0.9815 | 0.9826 | 0.9815 |
| ConvNeXt-Tiny baseline (single seed) | 1 | 0.9498 | 0.9493 | 0.9501 |
| EfficientNetV2-S baseline | 3 | 0.9478 ± 0.0023 | 0.9475 ± 0.0021 | 0.9477 ± 0.0023 |
| Swin-T baseline | 3 | 0.9450 ± 0.0118 | 0.9442 ± 0.0116 | 0.9449 ± 0.0118 |
| ResNet-50 baseline (single seed) | 1 | 0.9438 | 0.9433 | 0.9441 |
| TAMNet + channel attention (MC) | 4 | 0.9420 ± 0.0059 | 0.9416 ± 0.0059 | 0.9423 ± 0.0060 |
| TAMNet with concat fusion (no attention fusion) | 4 | 0.9402 ± 0.0085 | 0.9398 ± 0.0080 | 0.9404 ± 0.0084 |
| TAMNet 50-epoch budget control | 1 | 0.9390 | 0.9397 | 0.9394 |
| TAMNet baseline (g+t fusion), clean protocol | 5 | 0.9360 ± 0.0110 | 0.9355 ± 0.0109 | 0.9361 ± 0.0110 |
| DenseNet-121 classification-only baseline | 4 | 0.9360 ± 0.0125 | 0.9369 ± 0.0113 | 0.9365 ± 0.0122 |
| TAMNet t-only fusion | 4 | 0.9329 ± 0.0095 | 0.9326 ± 0.0092 | 0.9329 ± 0.0094 |
| ViT-B/16 baseline (single seed) | 1 | 0.9282 | 0.9277 | 0.9283 |
| BRISC λ_seg = 0 (no segmentation loss) | 1 | 0.9043 | 0.9027 | 0.9043 |
| Shared-encoder U-Net multi-task baseline (BRISC trained) | 4 | 0.9007 ± 0.0136 | 0.9004 ± 0.0140 | 0.9014 ± 0.0137 |
| BRISC λ_seg = 0.25 | 1 | 0.9001 | 0.8998 | 0.9005 |
| BRISC λ_seg = 0.1 | 1 | 0.8894 | 0.8882 | 0.8903 |
| BRISC λ_seg = 1.0 | 1 | 0.8780 | 0.8783 | 0.8791 |

Values are mean over seeds; ± is the sample standard deviation (ddof = 1) and is omitted when only one seed exists. Internal TEST split, n = 1,672, evaluated once per final configuration.

Source: `results/phase3/agg/combined_summary.json`
