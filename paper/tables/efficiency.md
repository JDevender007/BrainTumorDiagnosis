| Model | Parameters | Weights (MiB) | Single pass (ms) | MC 10 passes (ms) | FPS (single) |
|---|---|---|---|---|---|
| densenet121 | 6,957,956 | 26.54 | 65.66 | 636.72 | 15.2 |
| resnet50 | 23,516,228 | 89.71 | 41.70 | 415.23 | 24.0 |
| efficientnet_v2_s | 20,182,612 | 76.99 | 65.50 | 636.56 | 15.3 |
| convnext_tiny | 27,823,204 | 106.14 | 44.12 | 435.42 | 22.7 |
| swin_t | 27,522,430 | 104.99 | 63.16 | 627.26 | 15.8 |
| vit_b_16 | 85,801,732 | 327.31 | 83.33 | 1238.87 | 12.0 |
| shared_unet | 11,937,158 | 45.54 | 75.33 | 748.25 | 13.3 |
| tamnet_full | 9,034,416 | 34.46 | 50.76 | 530.27 | 19.7 |
| tamnet_tonly | 8,573,870 | 32.71 | 49.90 | 551.93 | 20.0 |
| tamnet_concat | 8,840,110 | 33.72 | 56.20 | 533.67 | 17.8 |
| tamnet_mc | 9,165,488 | 34.96 | 60.31 | 544.72 | 16.6 |

Protocol: batch 1, 20 warm-up, 200 timed iterations, CUDA synchronised around each iteration, device = NVIDIA GeForce RTX 3050 Laptop GPU.

Source: `results/phase3/efficiency.json`
