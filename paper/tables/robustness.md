| Configuration | Seeds | Clean accuracy | MRRS (mean ± SD) | Best-seed MRRS |
|---|---|---|---|---|
| tamnet_clean | 5 | 0.9360 | 0.8508 ± 0.0035 | 0.8541 |
| densenet121_clean | 4 | 0.9360 | 0.8142 ± 0.0260 | 0.8452 |

MRRS = mean over the 27 corruptions (9 types × 3 severities, listed in the source file) of accuracy(corruption) / accuracy(clean). Headline value is the mean over seeds, never the best seed.

Source: `results/phase3/robustness.json`
