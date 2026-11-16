# AUDIT_REPORT.md — Phase 0 (Recon, no changes)

**Repo:** `D:\BrainTumorDiagnosis` · **Branch:** `fix/paper-audit` · **Baseline commit:** `a2666f1` (all originals preserved) · **Cleanup commit:** `4fd1b1d`
**Paper under audit:** `TAMNet_Paper.docx` (root, 46,533 B, modified 2026-09-23 — the **current** draft; 381 paragraphs, 33 references)
**Status: STOPPED for user confirmation before Phase 1.**

---

## 0. TL;DR — the five findings that matter most

1. **The validation set used for every paper headline is 100% one class.** The paper's experiments split `data/raw/classification/Train` (9,291 images) with a naive 85/15 **tail slice** (`scripts/run_research_experiments.py:70-75`) over a sample list that is **class-block ordered** (`datasets/classification.py:146-186`). Verified by reconstruction: the 1,394-image validation tail is **`{'No Tumor': 1394}` — zero tumor images**. The training log confirms this run produced the paper numbers (`Train 7897 / Val 1394`, `outputs/full_experiment_log.txt`).
2. **This resolves the "F1 > accuracy" anomaly arithmetically.** On a single-class validation set, weighted-F1 of the No-Tumor class = `2·acc/(2·acc + (1−acc))`. With val acc 0.9699 → F1 = **0.984705** — exactly the paper's 0.9847 (`outputs/research_experiments/proposed_model/training_history.json`, epoch 26). The paper pairs a **test** accuracy (96.45%) with a **validation** F1 (0.9847) from a degenerate split. Same mechanism explains multi-seed val F1 0.9751 > val acc 0.9530 (2·0.9530/1.9530 = 0.9759 ≈ 0.9751 mean).
3. **Newer results contradict the paper.** `paper/results/*.json` (Sep 4–5, 2026) report internal **0.9295**, BRISC **0.9620**, Kaggle **0.9748** vs the paper's 96.45 / 97.70 / 99.16 (Aug 28–30 runs). Per audit rule 2, either the paper numbers get re-derived from a clean rerun (Phase 1–3) or the claims change — not the numbers.
4. **The channel-attention module M_c described in the paper is never instantiated.** `ChannelAttention` is defined (`models/tumor_aware.py:59-78`) but has **no call site anywhere in the repo**; the wired model only applies spatial attention (`TumorRegionPool`) + learned fusion. The paper's Method §III.C describes a component that does not exist in the executed model.
5. **MC dropout is implemented incorrectly in most paths.** `predict_with_uncertainty` calls `self.train()` (`models/tumor_aware.py:370`) → BatchNorm updates running stats during the 10 MC passes. Only `evaluation/uncertainty.py:46-116` does it correctly (eval mode + Dropout-only re-activation). Uncertainty numbers from different paths are not comparable.

---

## 1. Repo map (what exists, what feeds the paper)

| Area | Path | Role |
|---|---|---|
| Paper (current) | `TAMNet_Paper.docx` | **The audit target** |
| Paper (superseded) | `paper/manuscript/BrainTumorDiagnosis_IEEE_Research_Paper.docx` (Aug 30), `paper/generate_paper.py` (its generator), `scripts/generate_paper.py` (LaTeX, dangling `\bibliography{references}` — no `.bib` exists) | Historical only |
| Data sheets (not drafts) | `paper/manuscript/paper.txt` (1,451 lines), `paper/manuscript/Braintumor.txt` (1,471 lines) | Numeric source sheets used to write the docx |
| Model | `models/tumor_aware.py` (`TumorAwareMultiTaskNet` = the paper's TAMNet), `models/novel/turma_net.py` (superseded TURMA-Net) | |
| Losses | `losses/classification.py`, `losses/segmentation.py`, `training/multitask_trainer.py:75-98` | |
| Trainers | `training/classification_trainer.py` (backbones), `training/multitask_trainer.py` (TAMNet) | |
| Entry points | `python -m scripts.train_classification` (`scripts/train_classification.py:14-17`), `scripts/run_research_experiments.py` (**produced the paper numbers**), `scripts/run_multiseed.py` / `run_multi_seed.py` | |
| Results (paper source) | `outputs/research_experiments/all_experiment_results.json`, `outputs/research_experiments/multi_seed/statistical_results.json`, `outputs/kaggle_evaluation/results.json`, `outputs/comprehensive_metrics/*`, `outputs/metrics/segmentation_evaluation_test.json` | **gitignored — kept deliberately** |
| Results (newer, contradicting) | `paper/results/{ablation_all,multi_seed_all,external_validation,all_results}.json` + per-run `.pt` | Sep 4–5 reruns |
| Leakage evidence | `research_final_validation/leakage_report.json` | Kept (Kaggle 839/1311 proof) |
| Logs | `outputs/*.log`, `outputs/**/training_history.json` (31 files) | All gitignored |
| Notebooks | none | |

**Cleanup performed on branch** (commit `4fd1b1d`, recoverable from baseline `a2666f1`): deleted `research_implementation_audit/`, `research_publication_upgrade/`, `research_verification/`, `ui/`, `.pytest_cache/`, all `__pycache__/`, and the aborted `scripts/evaluation_pipeline.py`. **Kept despite being clutter:** `outputs/` and `research_final_validation/leakage_report.json` because they are the *only* sources for the paper's numbers (audit rule 1) and are gitignored/unrecoverable if deleted.

---

## 2. How each dataset is loaded

### 2.1 Internal classification set (the paper's "internal dataset")
- **On disk:** `data/raw/classification/{Train,Test}` — JPEG/PNG per class folder.
  - Train: 9,291 = Glioma 3,018 · Meningioma 2,183 · Pituitary 2,145 · No Tumor 1,945
  - Test: 1,857 = Glioma 603 · Meningioma 436 · Pituitary 429 · No Tumor 389
- **Loader:** `BrainTumorClassificationDataset` (`datasets/classification.py:244-293`); discovery `:141-186`; samples appended **class-by-class in `CLASS_NAMES = (Glioma, Meningioma, Pituitary, No Tumor)` order, files sorted** (`:97-101, :146-186`) — i.e. the sample list is class-block ordered and never shuffled at construction.
- **Class matching:** tolerant name normalization `:58-95` (`"no tumor"`/`"notumor"`/`"no_tumor"` → No Tumor).
- **⚠ FILL #1 (label provenance):** filenames are Kaggle-style (`gg (325).jpg`, `1 no.jpg`) and per-class counts do **not** match any BraTS 2024 task (BraTS-GLI has no pituitary/healthy class). **This set is not BraTS-derived.** Where its 4 labels came from is unknown → see §9.

### 2.2 BraTS 2024 (paper says: "used to evaluate the tumor segmentation branch")
- **On disk:** `data/raw/BraTS2024/` — 200 case dirs, each with `seg/t1c/t1n/t2f/t2w` NIfTI (5 files/case), **39 unique patient IDs** (case suffix `-100…` per patient).
- **Split:** patient-level, seed 42, 70/15/15 → 129/26/45 cases across 27/5/7 patients (`data/splits/brats/split_manifest.json`, `split_level: "patient"`; code `datasets/brats.py:342-442`, `datasets/splitting.py:36-39`). **This is the only patient-level split in the repo.**
- **⚠ FILL #3 (3D→2D):** how volumes become 2D classification inputs, which modality/channel, and which slices are kept — **not found in the loaders used for the paper runs**; segmentation eval used 45 test cases (`outputs/metrics/segmentation_evaluation_test.json`, `n=45` matches manifest test split).

### 2.3 BRISC 2025 (external)
- `data/external/brisc2025/brisc2025/classification_task/{train 5,000 / test 1,000}`, 4 class folders + `segmentation_task/{train,test}` with masks. Loaded flat in `scripts/run_research_experiments.py:89-121`.

### 2.4 Kaggle brain tumor (external)
- `data/external/kaggle_brain_tumor/Testing/` = **1,311** (glioma 300 · meningioma 306 · notumor 405 · pituitary 300) — **matches your FILL exactly** ✔. `Training/` = 3,881 with **no pituitary folder** (1,321/1,339/1,221) — incomplete copy.
- **⚠ FILL #2:** exact dataset name/link not recorded anywhere in the repo. Layout (Training/Testing, `notumor`) matches the common Kaggle "Brain Tumor MRI Dataset" — **[VERIFY]**, needs your confirmation.

### 2.5 Leakage (from `research_final_validation/leakage_report.json`)
- internal_train 9,286 vs internal_test: **218 filename overlaps** (earlier content check: different images, filename collision)
- internal_train vs Kaggle test: **839/1,311 overlap → Kaggle set is contaminated, not independent**
- internal_test vs Kaggle test: 0 · BRISC: no overlap found
- **⚠ Note:** report says `internal_test 1,155` but the Test folder has 1,857 files — count basis differs (needs reconciliation in Phase 1).

---

## 3. How splits are made — four inconsistent schemes (classification)

| # | Scheme | Location | Method |
|---|---|---|---|
| a | Trainer official | `training/classification_trainer.py:353-358` | stratified `train_test_split(test_size=0.2, random_state=validation_seed=42)`, **image-level** |
| b | Dataset helper | `datasets/classification.py:200-241` | per-class stratified, seed 42, image-level |
| c | **Paper runs** | `scripts/run_research_experiments.py:70-75` | `train[:int(0.85n)]` / `train[int(0.85n):]` — **naive tail slice, no shuffle, no stratification** |
| d | All-experiments | `scripts/run_all_experiments.py:127-138` | unstratified shuffle, image-level |

- **Patient/subject-level splitting for classification: does not exist.** Repo-wide grep for `GroupShuffleSplit`/`GroupKFold` = 0 hits. Only BraTS 3D is patient-level.
- **Scheme (c) + class-block ordering ⇒ the paper's validation set is 100% No Tumor** (verified by reconstruction, §0.1). Model selection (best val F1, early stopping) therefore selected on a degenerate single-class set.
- **Test-set usage:** the internal test (1,857) is evaluated in `run_research_experiments.py` cross-dataset phase; no evidence of test-based tuning found, but val-based selection is broken per above.

---

## 4. Preprocessing / normalization

| Data | Transform | Where |
|---|---|---|
| 2D classification (all sets) | Resize 224×224 → ToTensor → **ImageNet mean/std** (0.485/0.456/0.406, 0.229/0.224/0.225) | `training/classification_trainer.py:282,301-307`; `datasets/classification.py:262-265` |
| 3D volumes | per-volume z-score; percentile-clip branch exists but `clip_percentiles` is **always None** in every pipeline | `datasets/preprocessing.py:40-112` |
| BraTS dataset | z-score, `mask_background=False` | `datasets/brats.py:200-221` |
| Segmentation dataset | z-score, `mask_background=True` (**inconsistent with BraTS**) | `datasets/segmentation.py:54-73` |
| Augmentation (train) | HFlip 0.5, Rotate 10°, Affine ±5%/0.95–1.05, ColorJitter brightness 0.10 contrast 0.10 | `training/classification_trainer.py:284-299` |

- **No 1–99 percentile clipping anywhere** (Phase 1 requirement #5 does not exist yet).
- External sets: no evidence of different handling → same ImageNet normalization applied to JPEG T1 sets and (if used) BraTS-derived images.
- **Contrast corruption bug** (paper's 22.40%): pipeline variant applies `img * (1 + s*0.5*(rand−0.5))` — random per call, not a fixed factor (`scripts/evaluation_pipeline.py:147`, since deleted; original in baseline commit). The paper's `contrast_factors=[0.5,0.7,1.3,1.5]` lives in `evaluation/advanced_robustness.py:42`.

---

## 5. Model & MC dropout (code vs paper §III)

**Executed TAMNet = `TumorAwareMultiTaskNet`** (`models/tumor_aware.py:169-227`), DenseNet121 encoder via torchvision with classifier→Identity, feature captured by forward hook (`:258-285`), 3-channel input probe `zeros(1,3,224,224)` (`:287-293`) → **C=3** (paper leaves C undefined).

| Paper claim | Code reality | Verdict |
|---|---|---|
| Spatial attention M_s | `SpatialAttention`: 7×7 conv over [avg,max] channel-concat → sigmoid → elementwise multiply (`:42-56`) | ✅ exists |
| Channel attention M_c (§III.C) | `ChannelAttention` defined (`:59-78`) but **never instantiated** — no call site in repo | ❌ **does not exist in executed model** |
| `t = GAP(F_t)` after `F_s·M_c` | `TumorRegionPool` = SpatialAttention(7) → optional mask → GAP → MLP+Dropout(0.2) (`:86-103`) | ⚠ different from paper |
| Fusion `z = Φ(g, t)` | `AttentionFusion`: 2-way softmax attention `fused = w0·g + w1·t` + LayerNorm (`:111-132`) — **not concatenation** as paper's `g⊥t` states | ❌ paper mismatch |
| Seg branch `s = S(F)` | ✅ `seg_out = self.seg_head(feature_map)` — fed **raw encoder F** (`:353-357`); decoder = 2 conv blocks (in→128→64) + ×4/×2 bilinear upsample, **no skips** (`:138-150`); final upsample to 224 | ✅ input matches paper; resolution/skips under-specified in paper |
| 4-class head | Linear→BN→ReLU→Dropout(0.3) → Linear(256→128)→ReLU→Dropout(0.15) → Linear(128→4) (`:206-215`) | ✅ |
| MC dropout T=10 | `predict_with_uncertainty(num_passes=10)` (`:364-395`); dropout sites: tumor-pool 0.2, classifier 0.3/0.15, uncertainty head 0.3 | ⚠ **`self.train()` at `:370` puts BatchNorm in train mode too** — bug |
| MC done right | only `evaluation/uncertainty.py:46-116` (eval + Dropout-only re-activation, `:66-69`) | ✅ one correct path |
| Input channels C | 3 (RGB probe) | paper must state C=3 |

---

## 6. Losses (paper §III.F)

- **L_cls:** multi-task trainer = `CrossEntropyLoss(label_smoothing=0.1, class_weights)` (`training/multitask_trainer.py:75-98`; class weights inverse-frequency `classification_trainer.py:402-422`). Standalone classification runs use `label_smoothing=0.0` (`configs/classification.yaml:18`). **Paper states neither smoothing nor weights** → must be documented; smoothing 0.1 is a candidate contributor to underconfidence (conf 0.8821 < acc 0.9645).
- **L_seg:** `SegmentationLoss = 0.5·Dice + 0.5·CE` (`losses/segmentation.py:130-168`).
- **Total:** `1.0·L_cls + 0.5·L_seg` (`multitask_trainer.py:93-98`) — **matches paper `λ_cls=1.0, λ_seg=0.5`** ✅.
- (Superseded TURMA-Net losses in `models/novel/losses.py` — focal + uncertainty weighting — are not the paper's model.)

---

## 7. How F1 / ECE / MRRS are computed

### F1 — mixed averaging, and the headline is a validation metric
- **Averaging is inconsistent:** macro F1 at `scripts/run_all_experiments.py:410`, `evaluation/comprehensive_metrics.py:187`, `scripts/test_new_dataset.py:187`; **weighted** F1 at `evaluation/classification_evaluator.py:284-335`, `training/multitask_trainer.py:207` (also the **model-selection metric**), `evaluation/ablation_runner.py:124-213`. Same script uses weighted for selection and macro for reporting (`run_all_experiments.py:302` vs `:410`).
- **The paper's 0.9847 is `best_val_f1` (validation, epoch 26)** from `outputs/research_experiments/proposed_model/training_history.json`, **not** a test-set macro-F1. No test macro-F1 for TAMNet exists in the Aug results; the newer `paper/results/ablation_all.json` gives test macro-F1 **0.9315** (d_tumor_aware).
- Arithmetic proof of F1>acc on the degenerate val set: `2·0.9699/(2·0.9699+0.0301) = 0.98472` ✔.
- **Resolution for the paper (Phase 3C):** recompute all F1 from saved test predictions with one declared averaging (macro), on a fixed non-degenerate split.

### ECE — 15 bins, two non-interchangeable implementations
- 15 equal-width confidence bins: `evaluation/uncertainty.py:114-138`, `evaluation/calibration.py:44-62` (MCE `:65-83`, Brier `:85`, NLL `:92`, temperature scaling `:228-274`).
- **Inconsistency:** `uncertainty.py` computes ECE from **MC-mean** probabilities (`:72-74`); `calibration.py:306` from **single-pass** probabilities. Paper's ECE 0.0855 (single-run) / 0.0838±0.0056 (3-seed) come from `all_experiment_results.json` / `statistical_results.json` — source path must be pinned in Phase 3F.

### MRRS — defined in code, undefined in paper
- **Formula:** `MRRS = exp(mean(log(corrupted_acc / clean_acc)))` over all corruption variants (`evaluation/advanced_robustness.py:269-290`; docstring `:273-280`).
- **Corruption inventory — three disjoint suites:**
  1. `advanced_robustness.py:172-200`: **8 families** (gaussian_noise, salt_pepper, gaussian_blur, motion_blur, brightness, contrast, rotation, scale) with per-family variant counts summing to **27**; `jpeg_qualities=[10,30,50]` configured (`:43`) but **no JPEG transform implemented**.
  2. `mri_robustness.py:240-265`: 7 MRI-artifact families (bias_field, gibbs, motion, ghosting, intensity_nu, resolution_deg, slice_dropout) — no MRRS computed.
  3. (deleted) `evaluation_pipeline.py`: 6×3 grid with random contrast — superseded, in baseline commit.
- **Paper claims "27 image corruption conditions"** (`generate_paper.py:903` says "9 families × 3 severities") — the executed suite for MRRS 0.7512 is suite #1 (8 families, 27 variants) per `all_experiment_results.json.robustness.num_corruptions=27`. **Paper must list all 27 (type × severity) and state the MRRS formula** (Phase 3G).

---

## 8. Paper-number → source mapping

**Legend:** ✅ traced to a script output · ⚠ traced but flawed/ambiguous · ❌ conflicts or unsourced → `TODO-RERUN`

| # | Paper value (docx ¶) | Source file | Status |
|---|---|---|---|
| 1 | Internal test acc **96.45%** (¶2,¶10,¶134) | `outputs/research_experiments/all_experiment_results.json` → `cross_dataset.internal=0.9645` (n=1,857) | ⚠ traced; from degenerate-val run |
| 2 | **F1 0.9847** "on internal test" (¶134,¶180) | `…/proposed_model/training_history.json` epoch 26 `val_f1=0.9847050` | ❌ **it's a validation F1 on a 100%-No-Tumor val set** |
| 3 | BRISC **97.70%** (¶184,¶192) | `all_experiment_results.json` → `cross_dataset.external=0.977` | ⚠ traced; newer rerun says 0.9620 |
| 4 | Kaggle **99.16%** + F1 .9916/MCC .9888/κ .9887 (¶184,¶197-201) | `outputs/kaggle_evaluation/results.json` (n=1,311) | ❌ **contaminated: 839/1,311 filenames overlap internal train** |
| 5 | Kaggle confusion matrix (¶203-229) | `outputs/kaggle_evaluation/results.json.confusion_matrix` | ⚠ traced, but on contaminated set |
| 6 | Dice **WT .9005 / TC .5803 / ET .6010** (+HD95 4.69/34.20/16.33) (¶232-241) | `outputs/metrics/segmentation_evaluation_test.json` (n=45 test cases) | ✅ traced; no baseline, mask source unclear |
| 7 | Conf .8821/.8843/.8232, entropy .4778, MI .0415 (¶244-259) | `all_experiment_results.json.uncertainty` | ⚠ traced; MC path uses `self.train()` (BN bug) |
| 8 | ECE .0855, Brier .0736, NLL .2172, MCE .5603 (¶260-265) | `all_experiment_results.json.calibration` | ⚠ traced; ECE implementation path unpinned |
| 9 | MRRS **0.7512**, clean 96.45%, 27 corruptions, contrast 0.5→22.40%, brightness 54.17%, noise 56.81%, rot5° 95.96% (¶268-285) | `all_experiment_results.json.robustness` | ⚠ traced; corruption list missing from paper; **best-seed vs 3-seed mean 0.7088±0.0309 mismatch** |
| 10 | Multi-seed val F1 .9751±.0035, val acc .9530±.0057, ECE .0838±.0056, MRRS .7088±.0309, latency 46.27±2.48 ms, CIs (¶288-308) | `outputs/research_experiments/multi_seed/statistical_results.json` (seeds **42/123/2024**) | ⚠ traced; seed-set conflicts with `paper.txt` ([42,123,456]); val metrics degenerate |
| 11 | Backbone table internal acc (CNN-2D 95.26 … ViT 97.52, TAMNet 96.45) + external column (¶142-181) | `outputs/comprehensive_metrics/all_backbones_comprehensive.json` — exact match; **but also hardcoded** in `paper/generate_paper.py:85-91` | ⚠ traced; params conflict (DN121 6.96M vs 7.97M vs ~8M); "97.95" labeled `val_f1` in `Braintumor.txt:153` |
| 12 | Efficiency: **9,034,416 params, 34.46 MB, 48.63 ms, 20.6 FPS** (¶113,¶313-324) | `all_experiment_results.json.efficiency` | ⚠ hardware unstated; MC-pass inclusion unknown |
| 13 | Dataset sizes 7,897/1,857/9,754 (data sheets) | `full_experiment_log.txt` (Train 7897/Val 1394/Test 1857) | ❌ conflicts: disk Train=9,291; `Braintumor.txt` says 11,148 |
| 14 | "73/73 tests" (data sheets) | not re-verified | TODO-RERUN (Phase 6) |

**Additional conflicts found:**
- `paper/results/ablation_all.json` says A=0.9795 while `paper/results/ablation/A_baseline_densenet121_seed42.json` says **0.9499 for the same experiment label**; `paper/tables/*.md` mixed both sources.
- `paper/results/all_results.json` has **empty** `multi_seed` and `external` keys.
- Ref **[3] never cited in body** (0 occurrences); [9] has no authors; [28] is "A. Ovadia" (correct author list to verify: Y. Ovadia et al. — first author **Yehuda** Ovadia, paper says A.); [22] listed as Z. Yang (verify Q. Yang); no citations exist for BraTS 2024, BRISC 2025, the Kaggle set, or Hendrycks & Dietterich corruption benchmark; no `.bib` file for the LaTeX path.

---

## 9. Open FILLs — answers needed from you (or explicit "discover")

1. **Internal dataset provenance (critical):** `data/raw/classification/{Train,Test}` is *not* BraTS (Kaggle-style filenames, has pituitary/no-tumor, counts don't match BraTS tasks). **What dataset is it, and how were the 4 labels produced?** If it's a Kaggle-derived set, the "internal vs Kaggle external" comparison collapses (both may be the same source — consistent with the 839/1311 overlap).
2. **Kaggle dataset exact name/link** (local: `data/external/kaggle_brain_tumor/`; Testing counts match your FILL ✔; likely "Brain Tumor MRI Dataset" — **[VERIFY]**).
3. **3D→2D conversion:** modality/channels, slice rule, patient-level split for the classification task — **not found in the code paths that produced the paper numbers**. Was any BraTS data used for classification at all, or only for segmentation eval?
4. **Classifier training data:** paper runs trained on `data/raw/classification/Train` only (BRISC/Kaggle test used as external eval). Confirm this is the intended statement.
5. **Paper file:** `TAMNet_Paper.docx` (root) is the single source of truth for edits — confirm (superseded IEEE docx left untouched).

---

## 10. What Phase 1 will do (pending your confirmation)

Per the task plan: label-provenance documentation (stop if any class lacks training samples), patient-level/stratified split fix (invalidates current val-based claims — flagged, not deleted), `scripts/leakage_check.py` (exact-hash + pHash across all 4 dataset pairs → `results/leakage_report.json` with thumbnails), `results/dataset_table.csv`, and one shared 1–99%-clip + z-score normalization. **No number will be changed without a `results/` output behind it; current paper numbers stay untouched until re-derived.**

**⚠ Rule-6 notice for Phase 3:** 5-seed retraining of 7 baselines + 7 ablation configs + λ_seg sweep on an RTX 3050 will very likely exceed 2 GPU-hours — I will bring you a cost estimate before launching it.

---

*Phase 0 complete. Awaiting your confirmation to start Phase 1.*
