| Configuration | Seeds | Dice (tumour slices) | IoU | Pixel accuracy | Empty→empty |
|---|---|---|---|---|---|
| tamnet_brisc_lambda0 | 1 | 0.7271 | 0.6054 | 0.9914 | 0.821 |
| tamnet_brisc_lambda0.1 | 1 | 0.6657 | 0.5352 | 0.9891 | 0.950 |
| tamnet_brisc_lambda0.25 | 1 | 0.7021 | 0.5765 | 0.9905 | 0.979 |
| tamnet_brisc_lambda1 | 1 | 0.6875 | 0.5584 | 0.9885 | 0.857 |
| shared_unet_brisc | 4 | 0.8034 ± 0.0357 | 0.7135 | 0.9942 | 0.559 |

Binary tumour-vs-background Dice of the **auxiliary segmentation head** on the BRISC 2025 test split (2-D 224×224 slices). These are not the paper's WT/TC/ET numbers, which came from a standalone 3-D U-Net on BraTS volumes.

Source: `results/phase3/seg_metrics_summary.json`
