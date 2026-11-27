# Phase 5 — claim-by-claim rewrite basis

Status: working document, updated as Phase 3 evaluations land.
Rule: a number stays in the paper only if it traces to a file listed here. Otherwise it is
replaced by a Phase 3 number or marked `TODO-RERUN`.

---

## 0. The one finding that changes everything

| | Legacy pipeline (paper) | Phase 3 clean pipeline (audit) |
|---|---|---|
| internal test size | **1857** | **1672** |
| TAMNet internal accuracy | 96.45 % | **93.60 % ± 1.10 %** (5 seeds) |
| TAMNet macro F1 | 0.9847 | **0.9355 ± 0.0109** |
| TAMNet weighted F1 | 0.9847 | **0.9361 ± 0.0110** |
| TAMNet val F1w | 0.9751 ± 0.0035 (3 seeds) | **0.9473 ± 0.0029** (5 seeds) |

Phase 1 established that the legacy split shares images between train and test (near-duplicate
slices of the same patient/scan), which inflates every legacy number. The legacy values are
**traceable, not invented** — but they belong to a leaky protocol and cannot be reported as
clean-split performance.

Evidence:
- clean: `results/phase3/stage1_test_summary.json` (n = 1672, 5 seeds, `headline_rule` = mean over seeds)
- legacy: `outputs/research_experiments/calibration/calibration_tumor_aware.json` (`mean_accuracy` = 0.9644588, `num_samples` = 1857)
- split provenance: `results/phase3/split_manifest_clean.json` (sha256 recorded in every `run_config.json`)

---

## 1. Table 1 — baseline comparison

Paper reports "Internal Accuracy" / "F1 Score" / "External Accuracy". Every "Internal Accuracy"
value matches a legacy artifact exactly, all computed on the **1857-image leaky test set**:

| Row | Paper acc | Artifact | Artifact acc | Verdict |
|---|---|---|---|---|
| CNN-2D | 95.26 % | `outputs/cnn2d_evaluation/evaluation_metrics.json` | **0.6042** | **CONTRADICTED** — paper value has no matching artifact |
| DenseNet121 | 97.95 % | `outputs/densenet121_evaluation/evaluation_metrics.json` | 0.9795 | traceable (leaky split) |
| ResNet50 | 97.04 % | `outputs/resnet50_evaluation/evaluation_metrics.json` | 0.9704 | traceable (leaky split) |
| EfficientNetV2 | 98.06 % | `outputs/efficientnet_v2_s_evaluation/evaluation_metrics.json` | 0.9806 | traceable (leaky split) |
| ConvNeXt-Tiny | 97.42 % | `outputs/convnext_tiny_evaluation/evaluation_metrics.json` | 0.9742 | traceable (leaky split) |
| Swin-T | 97.09 % | `outputs/swin_t_evaluation/evaluation_metrics.json` | 0.9709 | traceable (leaky split) |
| ViT-B/16 | 97.52 % | `outputs/vit_b_16_evaluation/evaluation_metrics.json` | 0.9752 | traceable (leaky split) |
| TAMNet | 96.45 % | `outputs/research_experiments/calibration/calibration_tumor_aware.json` | 0.9645 | traceable (leaky split) |

**Action.** Replace the whole table with Phase 3 clean-protocol numbers. All 41 runs are now
trained and TEST-evaluated (`results/phase3/agg/combined_summary.json`), and every comparison
below has a paired bootstrap test (`results/phase3/bootstrap_ci.json`):

| Model | params | TEST accuracy (mean ± SD) | TEST macro-F1 | seeds | vs TAMNet (paired Δ) |
|---|---|---|---|---|---|
| ConvNeXt-T | 27.82 M | 0.9498 (n=1) | 0.9493 | 1 | −2.27 pp, p<0.05 |
| EfficientNetV2-S | 20.18 M | **0.9478 ± 0.0023** | 0.9475 ± 0.0021 | 3 | **−2.33 pp, p<0.05** |
| Swin-T | 27.52 M | 0.9450 ± 0.0118 | 0.9442 ± 0.0116 | 3 | **−1.97 pp, p<0.05** |
| ResNet-50 | 23.52 M | 0.9438 (n=1) | 0.9433 | 1 | **−1.67 pp, p<0.05** |
| TAMNet +MC | 9.17 M | 0.9420 ± 0.0059 | 0.9416 ± 0.0059 | 4 | — |
| TAMNet +concat | 8.84 M | 0.9402 ± 0.0085 | 0.9398 ± 0.0080 | 4 | — |
| **TAMNet full** | 9.034 M | **0.9360 ± 0.0110** | 0.9355 ± 0.0109 | **5** | reference |
| DenseNet-121 | 6.96 M | 0.9360 ± 0.0125 | 0.9369 ± 0.0113 | 4 | −0.18 pp, n.s. |
| TAMNet t-only | 8.57 M | 0.9329 ± 0.0095 | 0.9326 ± 0.0092 | 4 | — |
| ViT-B/16 | 85.80 M | 0.9282 (n=1) | 0.9277 | 1 | −0.12 pp, n.s. |
| shared-encoder U-Net | 11.94 M | 0.9007 ± 0.0136 | 0.9004 ± 0.0140 | 4 | +3.47 pp, p<0.05 (TAMNet better) |

Read honestly: TAMNet is **significantly worse** than EfficientNetV2-S, Swin-T, ResNet-50 and
ConvNeXt-T, statistically indistinguishable from DenseNet-121 and ViT-B/16, and clearly better
than the shared-encoder U-Net baseline. It is *not* the best classifier, and it is dominated on
the accuracy–size plane by DenseNet-121 (same 0.9360 accuracy with 2.07 M fewer parameters), so
the "efficiency" argument cannot be made either (`fig5`).

The per-run 95 % bootstrap CIs are ≈ ±1.1 pp (`results/phase3/bootstrap_ci.json`), so the
differences in the 0.1–0.5 pp range that the paper's Table 1 relies on are inside the noise.
The "External Accuracy" column has no artifact for the backbones and must be re-measured or
dropped.

---

## 2. Table 2 — Kaggle external evaluation

Paper: 99.16 % / F1 0.9916 / MCC 0.9888 / κ 0.9887, confusion matrix rows 300 / 306 / 300 / 405.

Artifact: `outputs/kaggle_evaluation/results.json` — `accuracy` 0.9916, `f1_weighted` 0.9916,
`mcc` 0.9888, `kappa` 0.9887, `total_images` 1311, `num_mc_passes` 10. Confusion matrix sums to
1311 and reproduces 1300/1311 = 0.9916.

**Verdict: VERIFIED as a legacy (leaky, old-checkpoint) number — and now superseded.**

Phase 3 re-ran external validation properly, five seeds, on both the full sets and the
de-duplicated keep-lists (`results/phase3/external_eval_summary.json`):

| Evaluation set | Images | Accuracy (mean ± SD) | Macro-F1 |
|---|---|---|---|
| Kaggle Testing — all images (leaky) | 1311 | 0.9788 ± 0.0050 | — |
| Kaggle Testing — de-duplicated | 422 | **0.9431 ± 0.0073** | — |
| BRISC test — all images (leaky) | 1000 | 0.9634 ± 0.0118 | — |
| BRISC test — de-duplicated | 431 | **0.9234 ± 0.0218** | — |

So even the *leaky* re-run does not reach 99.16 % / 97.70 % — those came from a different
checkpoint and a different (historical) training protocol. And the defensible external numbers
are 94.31 % (Kaggle) and 92.34 % (BRISC), i.e. **3.6–4.0 pp below their leaky counterparts**;
that gap is the leakage the old table was measuring. The paper must report the de-duplicated
rows and state the drop counts (67.8 % Kaggle, 56.9 % BRISC).

---

## 3. Table 3 — BraTS segmentation Dice

Paper: WT 0.9005, TC 0.5803, ET 0.6010.

Artifact: `outputs/metrics/segmentation_evaluation_test.json` — `num_cases` 45,
WT 0.900512 (std 0.047967), TC 0.580264 (std 0.252600), ET 0.600961 (std 0.242206),
plus RC 0.647364 and mean(WT,TC,ET) 0.693912.

**Verdict: VERIFIED**, and the std values should be added — TC/ET have SD ≈ 0.24–0.25, i.e. the
mean hides huge variance. Reporting only the means overstates the segmentation quality.
The paper also omits RC entirely.

---

## 4. Table 4 — uncertainty / calibration

Paper: mean conf 0.8821, correct 0.8843, incorrect 0.8232, entropy 0.4778, MI 0.0415,
ECE 0.0855, Brier 0.0736, NLL 0.2172.

Artifacts:
- `outputs/research_experiments/uncertainty/uncertainty_metrics.json` → 0.882076, 0.884278,
  0.823241, 0.477756, 0.041491, ece 0.085316, brier 0.076405, n = 1857
- `outputs/research_experiments/calibration/calibration_tumor_aware.json` → ece 0.085528,
  brier 0.073577, nll 0.217218

**Verdict: VERIFIED but internally inconsistent.** ECE and Brier are taken from *different files*
(0.0855 matches calibration; 0.0736 matches calibration Brier, while the uncertainty file says
0.0764). The paper mixes the two sources. Pick one file per metric and cite it.
`num_samples` = 1857 → leaky split again; re-run needed for a clean claim.

---

## 5. Table 5 — robustness

Paper: clean 96.45 %, rotation 5° 95.96 %, contrast 0.5 22.40 %, brightness 54.17 %,
Gaussian noise 56.81 %, MRRS 0.7512.

Artifacts: `outputs/research_experiments/robustness/robustness_tumor_aware.json` and the three
per-seed files under `outputs/research_experiments/multi_seed/seed_*/robustness/`.
Aggregate from `outputs/research_experiments/multi_seed/statistical_results.json`:
`proposed_robustness_mrrs` 0.70884 ± 0.03092 (n=3), `proposed_robustness_num_corruptions` 27.

**Verdict: VERIFIED** (single-seed values shown in the table, 3-seed mean reported elsewhere).
The paper says "27 image corruption conditions" — confirmed by the artifact.
Same leaky-split caveat applies.

---

## 6. Table 6 — stability and efficiency (3 seeds)

| Paper claim | Artifact (`outputs/research_experiments/multi_seed/statistical_results.json`) | Verdict |
|---|---|---|
| Val F1 0.9751 ± 0.0035 [0.9665, 0.9837] | 0.97512 ± 0.00347 [0.96649, 0.98375] | exact |
| Val Acc 0.9530 ± 0.0057 [0.9388, 0.9671] | 0.95297 ± 0.00571 [0.93879, 0.96715] | exact |
| ECE 0.0838 ± 0.0056 [0.0698, 0.0977] | 0.08375 ± 0.00563 [0.06976, 0.09774] | exact |
| MRRS 0.7088 ± 0.0309 [0.6320, 0.7856] | 0.70884 ± 0.03092 [0.63204, 0.78565] | exact |
| Latency 46.27 ± 2.48 ms [40.11, 52.42] | 46.26561 ± 2.4776 [40.1109, 52.42032] | exact |
| params 9,034,416 | 9034416 (n=3, std 0) | exact |
| model size 34.46 MB | 34.46356 | exact |
| **20.6 FPS** | 21.65558 ± 1.15593 | **CONTRADICTED** (20.6 has no source) |
| **External acc 97.70 %** | 0.973 (cross_dataset_external_accuracy) | **CONTRADICTED** (paper says 97.70) |

**Action.** Keep the five exact rows, but re-label them as the legacy 3-seed leaky-split run, or
replace with the Phase 3 5-seed numbers:
val F1 **0.9473 ± 0.0029**, TEST accuracy **0.9360 ± 0.0110**, TEST macro F1 **0.9355 ± 0.0109**.
Fix FPS to 21.7 (or drop) and external accuracy to 97.30 % (or re-measure).

---

## 7. Claims that are safe as written

- **9,034,416 trainable parameters** — reproduced independently (Phase 2), and consistent with
  `proposed_efficiency_params`.
- **34.46 MB checkpoint size** — matches `proposed_efficiency_size_mb`.
- **27 corruption conditions** — matches `num_corruptions`.
- **10 MC-dropout passes** — matches `num_mc_passes` in the Kaggle artifact.
- **45 BraTS cases** for segmentation — matches `num_cases`.

---

## 7b. External validation is not independent (new, blocking)

`results/phase3/external_dedup_manifest.json` (built with the same phash≤6 / dhash≤4 union rule as
Phase 1) near-duplicate-matches every external image against the **clean train** split:

| External source | images | near-duplicates of clean train | kept | % dropped |
|---|---|---|---|---|
| Kaggle training | 3881 | 2599 | 1282 | **66.97 %** |
| Kaggle testing (the 1311 used for 99.16 %) | 1311 | 889 | 422 | **67.81 %** |
| BRISC train | 5000 | 2676 | 2324 | **53.52 %** |
| BRISC test (the 1000 used for 97.70 %) | 1000 | 569 | 431 | **56.90 %** |

Two thirds of the "external" Kaggle test images are near-duplicates of images the model trained on.
So 99.16 % (Kaggle) and 97.70 % (BRISC) are **not** external-validation results in any meaningful
sense — they are mostly in-distribution.

**Consequence for the paper.** Either
(a) re-evaluate on the de-duplicated keep-lists (`keep_paths` in the manifest: 422 Kaggle test,
431 BRISC test) and report those numbers with the drop counts stated, or
(b) drop the "two independent external datasets" claim entirely.

Option (a) is the defensible one and needs only inference, no retraining. It changes the expected
accuracy substantially downward and must be run before the paper can state any external number.

Also note the dataset sizes do not line up with the canonical Kaggle release:
- internal pool (`data/raw/classification`): Train 9291 + Test 1857 = **11148**
- external Kaggle (`data/external/kaggle_brain_tumor`): Training 3881 + Testing 1311 = **5192**
- the widely cited Kaggle "Brain Tumor MRI Dataset" is 5712 + 1311 = 7023

So the internal set is not that release either. Exact dataset URLs are still required (FILL #2).

---

## 8. Open items blocking the rewrite

| # | Item | Blocks | Status |
|---|---|---|---|
| 1 | Internal dataset provenance and label origin | Methods section wording | **PARTIALLY ANSWERED** — see §9 |
| 2 | Exact Kaggle dataset name/link used for `outputs/kaggle_evaluation` | External validation wording | **PARTIALLY ANSWERED** — see §9 |
| 3 | BraTS role (classification training vs segmentation only) and 3D→2D conversion | Methods + Table 3 caption | **ANSWERED** — see §9 |
| 4 | Kaggle dataset availability for a clean-protocol re-run | Whether 99.16 % can stay | **ANSWERED** — the zip is on disk, 7023 entries, and the de-duplicated re-run is done (`external_eval_summary.json`) |
| 5 | Whether the 5 corruption/robustness numbers are to be re-run on the clean split | Table 5 | **ANSWERED** — re-run on the clean split, 27 corruptions, `results/phase3/robustness.json` |

Items 1–2 are answered as far as the repository can prove them; the remaining unknowns stay
marked `[VERIFY]` / `[CITATION NEEDED]` in the paper rather than being filled with a guess.
Evidence: `results/fill_evidence.json`, produced by `scripts/phase5_fill_evidence.py`.

---

## 9. FILL resolutions (filesystem evidence, 2026-11-27)

### FILL #1 — internal dataset provenance / label origin

Established:

- `data/raw/classification` holds **11,148** 2-D JPEG/PNG images (Train 9,291 / Test 1,857) in
  four class folders, **10,223 unique basenames**. Counts: Train 3018 / 2183 / 2145 / 1945,
  Test 603 / 436 / 429 / 389 (glioma / meningioma / pituitary / no-tumor).
- **4,463 of 10,223 (43.7 %) basenames are also in the canonical Kaggle zip**, so the internal
  pool is substantially a re-use of that release.
- The remainder comes from at least three other naming families — `X_###_XX_.jpg` (3,439,
  institution-coded), plain numeric (1,713), `gg (N)` (363), `image(N)` (123), `N no` (34).
- 490 basenames occur in more than one class folder, but **0 are byte-identical** — they are
  benign filename collisions, **not** label conflicts.
- No patient/subject identifier exists in any filename, so the split cannot be made
  patient-level from this folder. BraTS provenance is excluded: BraTS has no pituitary and no
  no-tumour class.

Not established (stays `[CITATION NEEDED]`):

- Who produced the four-class labels, and from which primary source the non-Kaggle portion was
  collated. No download script, manifest, README or citation for this folder exists anywhere in
  the repository.

**Paper wording:** state the class counts and the image-level split; mark the label origin
`[CITATION NEEDED]`; do not claim BraTS provenance.

### FILL #2 — exact Kaggle dataset name/link

Established:

- The archive on disk is literally `data/external/kaggle_brain_tumor/brain-tumor-mri-dataset.zip`
  with **7,023 entries**: Testing `glioma 300 / meningioma 306 / notumor 405 / pituitary 300` =
  **1,311**, Training `1321 / 1339 / 1595 / 1457` = **5,712**.
- Those counts equal the canonical Kaggle **"Brain Tumor MRI Dataset"** release
  (5,712 + 1,311 = 7,023), consistent with the `Te-*` / `Tr-*` filenames.
- The *extracted* folders the pipeline reads are an **incomplete** extraction: `Training/` has
  only glioma 1321 / meningioma 1339 / notumor 1221 and **no pituitary folder** (3,881 of 5,712);
  `Testing/` matches the zip exactly (1,311).

Not established (stays `[VERIFY]`):

- The Kaggle owner slug / URL is not written anywhere in the repo — only the archive file name
  is. Confirm against the Kaggle page before camera-ready.

**Paper wording:** cite the dataset by its archive name and canonical size; `[VERIFY]` the
author/URL before camera-ready.

### FILL #3 — BraTS role and 3-D → 2-D conversion

Established:

- Classification is trained and evaluated **only** on 2-D JPEG/PNG slices
  (`data/raw/classification`); BraTS contributes **no** classification images.
- The local BraTS copy holds **1,000 NIfTI volumes** (t1c 200 / t1n 200 / t2f 200 / t2w 200) and
  **200 segmentation targets**, 200 case dirs, 39 patients; `split_manifest.json` is
  `split_level=patient`, seed 42, ratios 0.7/0.15/0.15 — the only patient-level split in the repo.
- BraTS 2024 carries **no pituitary and no healthy/no-tumour label**, so it cannot supply the
  four-class labels used by the classifier.
- **No 3-D → 2-D slice conversion exists anywhere in the classification path**; the four-class
  task was never run on converted BraTS volumes.

Not established (stays `[VERIFY]`):

- Which BraTS 2024 task/subset the segmentation checkpoint was trained on, and the exact
  slice-selection rule.

**Paper wording:** BraTS = segmentation evaluation only; WT/TC/ET Dice belong to the standalone
3-D U-Net (D3), not to TAMNet's classification path.
