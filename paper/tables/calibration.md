| Configuration | Seeds | Accuracy | ECE (15 bins) | MCE | NLL | Brier | MI (epistemic) | Error AUROC |
|---|---|---|---|---|---|---|---|---|
| ConvNeXt-Tiny baseline (single seed) | 1 | 0.9498 | 0.0410 | 0.7095 | 0.2316 | 0.0903 | 0.0005 | 0.6686 |
| EfficientNetV2-S baseline | 3 | 0.9480 ± 0.0021 | 0.0399 ± 0.0030 | 0.6042 ± 0.0879 | 0.2354 ± 0.0103 | 0.0940 ± 0.0051 | 0.0008 ± 0.0001 | 0.6922 ± 0.0203 |
| Swin-T baseline | 3 | 0.9450 ± 0.0124 | 0.0449 ± 0.0094 | 0.5171 ± 0.0722 | 0.2398 ± 0.0242 | 0.0973 ± 0.0159 | 0.0007 ± 0.0003 | 0.6662 ± 0.0584 |
| ResNet-50 baseline (single seed) | 1 | 0.9438 | 0.0466 | 0.2163 | 0.2441 | 0.1000 | 0.0010 | 0.7201 |
| TAMNet + channel attention (MC) | 4 | 0.9412 ± 0.0047 | 0.0708 ± 0.0033 | 0.5158 ± 0.2155 | 0.2676 ± 0.0196 | 0.1059 ± 0.0092 | 0.0428 ± 0.0023 | 0.7132 ± 0.0269 |
| DenseNet-121 classification-only baseline | 4 | 0.9363 ± 0.0122 | 0.0545 ± 0.0011 | 0.3448 ± 0.1405 | 0.2509 ± 0.0256 | 0.1075 ± 0.0163 | 0.0049 ± 0.0013 | 0.7524 ± 0.0444 |
| TAMNet baseline (g+t fusion), clean protocol | 5 | 0.9359 ± 0.0110 | 0.0760 ± 0.0102 | 0.4028 ± 0.1641 | 0.2804 ± 0.0212 | 0.1130 ± 0.0130 | 0.0494 ± 0.0117 | 0.7214 ± 0.0746 |
| ViT-B/16 baseline (single seed) | 1 | 0.9282 | 0.0343 | 0.5055 | 0.2952 | 0.1257 | 0.0006 | 0.7025 |
| Shared-encoder U-Net multi-task baseline (BRISC trained) | 4 | 0.8989 ± 0.0138 | 0.0508 ± 0.0107 | 0.3379 ± 0.2380 | 0.3724 ± 0.0266 | 0.1697 ± 0.0188 | 0.0050 ± 0.0020 | 0.7596 ± 0.0583 |

MC-Dropout T = 10 with BatchNorm frozen and only Dropout layers active (Phase 2 / D5 fix). Error AUROC = how well mutual information separates misclassified from correctly classified slices (0.5 = no signal).

Source: `results/phase3/uncertainty_summary.json`
