| Evaluation set | Images | Accuracy (mean ± SD) | Macro-F1 |
|---|---|---|---|
| Kaggle Testing — all images (leaky) | 1311 | 0.9788 ± 0.0050 | 0.9780 ± 0.0053 |
| Kaggle Testing — de-duplicated | 422 | 0.9431 ± 0.0073 | 0.9424 ± 0.0077 |
| BRISC test — all images (leaky) | 1000 | 0.9634 ± 0.0118 | 0.9586 ± 0.0142 |
| BRISC test — de-duplicated | 431 | 0.9234 ± 0.0218 | 0.9206 ± 0.0224 |

'De-duplicated' drops every external image whose perceptual hash near-matches a clean-training image (pHash ≤ 6 or dHash ≤ 4 union). The all-images row reproduces the leakage the legacy numbers reported; the de-duplicated row is the only number that may be called external generalisation.

Source: `results/phase3/external_eval_summary.json`
