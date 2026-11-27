| Run | Accuracy | 95% CI (bootstrap) |
|---|---|---|
| convnext_tiny_clean_s42 | 0.9498 | [0.9390, 0.9599] |
| densenet121_clean_s42 | 0.9252 | [0.9139, 0.9372] |
| densenet121_clean_s43 | 0.9522 | [0.9426, 0.9623] |
| densenet121_clean_s44 | 0.9270 | [0.9151, 0.9390] |
| densenet121_clean_s45 | 0.9396 | [0.9282, 0.9510] |
| efficientnet_v2_s_clean_s42 | 0.9504 | [0.9402, 0.9605] |
| efficientnet_v2_s_clean_s43 | 0.9462 | [0.9360, 0.9569] |
| efficientnet_v2_s_clean_s44 | 0.9468 | [0.9366, 0.9581] |
| resnet50_clean_s42 | 0.9438 | [0.9330, 0.9546] |
| shared_unet_brisc_s42+ls0.25+sc2 | 0.8923 | [0.8768, 0.9061] |
| shared_unet_brisc_s43+ls0.25+sc2 | 0.9151 | [0.9019, 0.9276] |
| shared_unet_brisc_s44+ls0.25+sc2 | 0.9091 | [0.8959, 0.9234] |
| shared_unet_brisc_s45+ls0.25+sc2 | 0.8864 | [0.8720, 0.9007] |
| swin_t_clean_s42 | 0.9468 | [0.9366, 0.9569] |
| swin_t_clean_s43 | 0.9557 | [0.9456, 0.9653] |
| swin_t_clean_s44 | 0.9324 | [0.9222, 0.9450] |
| tamnet_brisc_s42+ls0+sc2 | 0.8888 | [0.8738, 0.9037] |
| tamnet_brisc_s42+ls0.1+sc2 | 0.8894 | [0.8750, 0.9031] |
| tamnet_brisc_s42+ls0.25+sc2 | 0.9001 | [0.8864, 0.9139] |
| tamnet_brisc_s42+ls1+sc2 | 0.8780 | [0.8624, 0.8935] |
| tamnet_brisc_s42+sc2 | 0.9043 | [0.8900, 0.9175] |
| tamnet_clean_s42 | 0.9270 | [0.9145, 0.9402] |
| tamnet_clean_s42+concat | 0.9426 | [0.9324, 0.9533] |
| tamnet_clean_s42+mc | 0.9420 | [0.9312, 0.9533] |
| tamnet_clean_s42-tonly | 0.9384 | [0.9264, 0.9492] |
| tamnet_clean_s42_ep50budget | 0.9390 | [0.9276, 0.9504] |
| tamnet_clean_s43 | 0.9300 | [0.9186, 0.9426] |
| tamnet_clean_s43+concat | 0.9276 | [0.9151, 0.9390] |
| tamnet_clean_s43+mc | 0.9372 | [0.9258, 0.9486] |
| tamnet_clean_s43-tonly | 0.9336 | [0.9217, 0.9456] |
| tamnet_clean_s44 | 0.9486 | [0.9378, 0.9593] |
| tamnet_clean_s44+concat | 0.9462 | [0.9354, 0.9569] |
| tamnet_clean_s44+mc | 0.9504 | [0.9402, 0.9605] |
| tamnet_clean_s44-tonly | 0.9193 | [0.9073, 0.9324] |
| tamnet_clean_s45 | 0.9474 | [0.9366, 0.9575] |
| tamnet_clean_s45+concat | 0.9444 | [0.9336, 0.9545] |
| tamnet_clean_s45+mc | 0.9384 | [0.9276, 0.9504] |
| tamnet_clean_s45-tonly | 0.9402 | [0.9276, 0.9516] |
| tamnet_clean_s46 | 0.9270 | [0.9151, 0.9390] |
| tamnet_leaky_s42 | 0.9815 | [0.9743, 0.9880] |
| vit_b_16_clean_s42 | 0.9282 | [0.9163, 0.9408] |

| Comparison (same test indices) | Δ accuracy | 95% CI | Conclusion |
|---|---|---|---|
| tamnet_clean_s42 − densenet121_clean_s42 | +0.0018 | [-0.0090, +0.0132] | not distinguishable |
| tamnet_clean_s42 − resnet50_clean_s42 | -0.0167 | [-0.0287, -0.0054] | different |
| tamnet_clean_s42 − efficientnet_v2_s_clean_s42 | -0.0233 | [-0.0341, -0.0114] | different |
| tamnet_clean_s42 − convnext_tiny_clean_s42 | -0.0227 | [-0.0341, -0.0108] | different |
| tamnet_clean_s42 − swin_t_clean_s42 | -0.0197 | [-0.0305, -0.0090] | different |
| tamnet_clean_s42 − vit_b_16_clean_s42 | -0.0012 | [-0.0156, +0.0132] | not distinguishable |
| tamnet_clean_s42 − shared_unet_brisc_s42+ls0.25+sc2 | +0.0347 | [+0.0203, +0.0508] | different |

Percentile bootstrap, B = 1000, resampling test images (splits are image-level, so this is not a patient-level interval).

Source: `results/phase3/bootstrap_ci.json`
