# TODO — Audit of `TAMNet_Paper.docx`

Source of truth: **TAMNet_Paper.docx** (root). Paper title: *"TAMNet: Tumor-Aware Multi-Task
Representation Learning for Brain MRI Classification and Segmentation with Uncertainty
Quantification"* — 381 paragraphs: Abstract, I. Introduction, II. Related Work, III. Proposed
Method (A–I), IV. Experimental Setup, V. Results (A–H), VI. Discussion, VII. Limitations,
VIII. Conclusion, References [1]–[33].

Every item below must be completed under the audit's ABSOLUTE RULES (no invented numbers;
every number traces to a `results/` output; change claims not numbers; test set used once
per final config; ask before >2 GPU-hours or new downloads; mark unverifiable refs
`[VERIFY]` / `[CITATION NEEDED]`).

---

## Phase 0 — Recon (STOP after this for confirmation)
- [x] Write `docs/AUDIT_REPORT.md`: how each dataset is loaded; how splits are made
      (patient-level?); preprocessing/normalization; where MC dropout sits (T=10);
      how F1 / ECE / MRRS are computed; paper-number → log/JSON mapping.
- [x] Flag every unsourced number in the paper (list below) as `TODO-RERUN`.

## Phase 1 — Data integrity (DONE 2026-11-16 → `docs/PHASE1_REPORT.md`)
- [x] Label provenance: class counts verified for all 4 classes in every source
      (none empty); label **origin** still open → FILL #1 (blocks Phase 5 wording only).
- [x] Verify splits are by subject ID, not slice → **classification splits are
      image-level and cannot be subject-level** (no patient IDs in filenames;
      BraTS manifest is the only patient-level split, 0 overlap). GroupShuffleSplit
      fix + invalidating prior results → Phase 3.
- [x] `scripts/leakage_check.py` → `results/leakage_report.json` + 20 thumbnails
      + label-agreement check (12,893/12,893 agree). Headline: Kaggle Testing
      94.7 % inside internal train; internal test 74.3 % near-dup of train;
      metrics before/after de-duplication = `TODO-RERUN` (Phase 3).
- [x] `results/dataset_table.csv` (7 rows, `[VERIFY]`/`[CITATION NEEDED]` marked).
- [-] Shared per-image normalization: **partially** — `datasets/normalization.py`
      now single-sources the (identical-everywhere) ImageNet stats + provides
      1–99 percentile z-score; switching training to it invalidates all numbers,
      so it is wired in only with the Phase 3 re-run.

## Phase 2 — Code vs. paper consistency (DONE 2026-11-16 → `docs/ARCHITECTURE.md`)
- [x] Expose/document M_s and M_c (kernels, pooling, MLP ratio, sigmoid); note
      CBAM-style spatial-then-channel order → documented in `docs/ARCHITECTURE.md` §1;
      **D1: M_c was never instantiated** → `use_channel_attention` flag (default False =
      historical), Phase 3 "+spatial+channel" ablation; claim change → Phase 5.
- [x] Identify exact fusion op; document; rename function (e.g. `fuse`) so it does not
      clash with feature map F → `fuse()` (`models/tumor_aware.py:488`); **D4: Φ = 2-way
      softmax convex combination + LayerNorm, NOT the paper's `g⊥t` concatenation**
      (`[060]` must be rewritten, Phase 5); concat ablation = `use_attention_fusion=False`.
- [x] Segmentation branch: document decoder input (paper says `s = S(F)` on F),
      resolution, skips; make code and paper agree; add lightweight multi-scale decoder
      variant if it decodes a 7×7 map with no skips and report both → documented (7×7-map,
      no skips, ×4/×2 → 56 → interpolate 224); `seg_decoder="multiscale"` U-Net variant
      added (default-off, densenet121-only); **D3: paper's Dice 0.9005/0.5803/0.6010 come
      from standalone UNet3D, not this branch** → claim change Phase 5, re-run Phase 3E.
- [x] Document L_cls (CE? label smoothing? class weights?) and L_seg (Dice+BCE?);
      check if label smoothing explains underconfidence → L_cls = CE(label_smoothing=0.1),
      **no class weights** in multitask path (weights exist only in standalone
      classification trainer); **D2: L_seg ≡ 0 in every historical run** (no runner passes
      seg_targets; `DiceLoss3D` is 5D-only) → warn-once + `DiceLoss2D` dispatch added;
      label smoothing as underconfidence contributor → to be tested Phase 3F.
- [x] MC dropout: record locations + rate; ensure `model.eval()` keeps BatchNorm in eval
      while only Dropout layers are in train mode → **D5 bug fixed**: eval + Dropout-only
      re-activation + mode restore; sites: classifier 0.3/0.15, tumor-pool 0.2,
      uncertainty head 0.3; all prior uncertainty numbers → `TODO-RERUN` (Phase 3F).
- [x] State input channels C in code and paper (paper leaves C undefined) → C=3 documented
      (`docs/ARCHITECTURE.md` §1); paper wording → Phase 5.
- [x] New Phase 2 tests: `tests/test_tumor_aware_phase2.py` (18 tests) — M_c flag,
      decoder variants, `fuse()` math, MC BN-eval contract, loss contract (D2).
- [-] "Make code and paper agree" — **code side done** (documented truth + default-off
      flags preserve history); **paper side blocked on Phase 5** (claims D1/D3/D4/D9 change
      there, never the numbers).

## Phase 3 — Experiments (5 seeds; TEST mean±std; headline = mean, never best seed)

**Progress-reporting protocol (user requirement 2026-11-19):** every training run must
report **(a) at each epoch completion** and **(b) at least every 5 minutes** what is
currently running. Enabled by: per-epoch `training_history.json` flush +
`progress.json` (`status/epoch/epochs_total/epoch_time_s/elapsed_s/eta_s/val_f1/
best_val_f1/updated_at`) in `training/multitask_trainer.py`; `classification_trainer.py`
already writes `training_history.json` + `progress_file` per epoch. Watcher loop =
sleep-5-min → read `progress.json` → user-facing update (epoch just finished, current
activity, ETA).

- [x] A. Ablation — **DONE** (Table V + `fig4_ablation_val`, `fig4b_test_accuracy`):
      41 trainings cover cls-only / +spatial / +channel / fusion g+t vs t-only /
      λ_seg ∈ {0, 0.25, 0.5} / +MC / +concat. **Finding D1**: the channel branch
      and λ_seg > 0 make no difference (L_seg ≡ 0 in the historical runs, D2/D3),
      which is why the paper reports the components as non-contributing rather
      than as gains.
- [x] B. Fair baselines — **DONE**: DenseNet-121, ResNet-50, EfficientNet-V2-S,
      ConvNeXt-T, Swin-T, ViT-B/16 (all 30 epochs, same protocol, seeds 42–46
      where applicable) **plus** the shared-encoder U-Net multi-task baseline
      (`shared_unet_brisc`, 4 seeds). Table III / Fig 4b / Fig 5.
- [x] C. Metrics audit — **DONE**: every metric is computed from saved
      predictions and cross-checked against scikit-learn in `tests/`
      (`113 passed`); averaging is macro vs weighted and stated in the paper.
      The legacy macro-F1 0.9847 > accuracy 96.45 % contradiction is resolved
      by re-measurement: **0.9360 ± 0.0110 accuracy / 0.9355 ± 0.0109 macro-F1**.
      Per-sample probabilities are now persisted at
      `results/phase3/predictions/`.
- [x] D. Per-dataset confusion matrices + bootstrap CIs — **DONE**:
      `results/phase3/bootstrap_ci.json` (B = 1 000, 95 % CIs on every accuracy),
      confusion matrices in each `eval_*/metrics.json` (internal → Fig 6;
      external rows in Fig 15 / Table VI), de-duplicated vs full both reported.
- [x] E. Segmentation — **DONE**: patient-held-out BraTS test split (45 cases),
      Dice **+** the full per-case record at `results/brats_unet3d_wt_tc_et.json`
      (WT/TC/ET mean, sample SD, median, min, max); provenance = standalone
      3-D U-Net (D3), label mapping documented; BRISC aux-head binary Dice
      (Table VII) with the shared-U-Net baseline. Comparison against published
      Swin-HAFNet is **not** verifiable from this repo (no checkpoint/metrics on
      disk) → not claimed.
- [x] F. Uncertainty/calibration — **DONE with two sub-items cut**: reliability
      diagram (Fig 9), ECE 15 bins, NLL, Brier, MCE, AUROC of entropy/MI,
      risk-coverage + AURC, correct-vs-incorrect confidence, MC-dropout **T = 10**
      with the D5 fix (BN eval, dropout active) → `uncertainty_summary.json`.
      **Not run** (declared in Limitations): temperature scaling and the
      5-model deep ensemble — additional training/inference outside the audit
      scope. Miscalibration direction re-stated from the measured ECE.
- [ ] G. Robustness — **PARTIAL**: corruption pipeline unit-tested, all
      27 corruptions × 3 severities run (`results/phase3/robustness.json`),
      MRRS defined formally, headline 0.7512 replaced by
      **0.8508 ± 0.0035** (n = 5 seeds) and contrast-0.5 collapse re-measured.
      **Not run**: the retrain with intensity augmentation + per-image
      normalization (a new training experiment, out of audit scope).
- [x] H. Interpretability — **measurement DONE**: mass-in-mask, lift and
      pointing-game on 860 BRISC tumour slices vs uniform **and** vs
      DenseNet-121 Grad-CAM (`results/phase3/attention_alignment.json`,
      `missing: []`). Result is negative: TAMNet attention 0.0103 (0.54×) vs
      Grad-CAM 0.0765 (4.01×), paired Δ −0.0662 [−0.0706, −0.0616],
      significant. The *optional attention-supervision loss* (with/without
      comparison) was **not** run — a new training experiment outside the
      audit — and the paper no longer promises it (findings D1/D6).
- [x] I. Efficiency — **DONE**: fixed protocol (batch 1, 20 warm-up, 200 timed,
      CUDA synchronised), hardware named, params/MiB/single-pass/T=10 MC for
      every model → `efficiency.json`; **48.63 ms did not include MC** (legacy
      ambiguity resolved: 50.76 ms single pass vs 530.27 ms T=10 MC) and
      "near-real-time" dropped.
- [x] J. Cross-dataset matrix — **DONE** (Fig 15): train-source × test-set
      accuracy for internal-trained TAMNet, BRISC-trained TAMNet,
      BRISC-trained shared U-Net and the leaky control, each × 5 test sets
      (internal TEST, Kaggle full/de-dup, BRISC full/de-dup). A
      *BraTS-derived* training row does not exist — BraTS carries no 4-class
      labels (D3) — and the paper says so instead of inventing the row.

## Phase 4 — Figures — **DONE** (commits `618527c`, `1c96c89`)

Vector PDF + 600-dpi PNG for **15 / 15 figures** in `paper/figs/`, generated by
`scripts/make_figures.py` (fig4/4b/5/6/6b/9/10/11/13/14/15) and the new
`scripts/make_figures_static.py` (fig1/2/3/8), all recorded in
`figures_manifest.json`, all cited in the manuscript in ascending order.
Item-by-item status and source of every number: `CHANGELOG_FIXES.md` § Phase 4.

- [x] Fig 1 `architecture` (tensor shapes from **real forward hooks**) ·
      Fig 2 `data_pipeline` · Fig 3 `samples` · Fig 4 `ablation_val` ·
      Fig 4b `test_accuracy` (bootstrap CI) · Fig 5 `pareto` ·
      Fig 6 `confusion` + Fig 6b `per_class_f1` · Fig 8 `seg_box` (BRISC
      per-slice Dice, 8 runs) · Fig 9 `reliability` · Fig 10 `uncertainty` ·
      Fig 11 `attention_alignment` · Fig 13 `robustness_heatmap` ·
      Fig 14 `intensity_curves` · Fig 15 `generalization_matrix`.
- [x] **DONE** Fig 7 `seg_qualitative` — `scripts/phase4_seg_qualitative.py`
      picks argmax / median-nearest / argmin by the Dice *already recorded* in
      `seg_metrics.json` and re-predicts from each run's `best.pt`; record at
      `results/phase4_seg_qualitative.json`.
- [x] **DONE** Grad-CAM panel — `phase3_attention_alignment.py` now unwraps
      `DictWrapper.backbone` and applies the non-inplace ReLU patch;
      `attention_alignment.json` `missing: []`, DenseNet-121 Grad-CAM lift
      4.01× against TAMNet attention 0.54× (paired Δ −0.0662, significant).
- [x] **DONE** Fig 8 WT/TC/ET half — per-case record found at
      `outputs/metrics/segmentation_evaluation_test.json` and ingested by
      `scripts/phase5_ingest_unet3d.py` → `results/brats_unet3d_wt_tc_et.json`.
- [x] **DONE** Supplementary `paper/figs/supp/` — **10 / 10 panels** via
      `scripts/make_supplementary.py` (+ `scripts/phase4_save_predictions.py`
      for S3/S6/S7); manifest records 0 missing.
- [x] `scripts/check_figures.py` — see Phase 6.

## Phase 5 — Paper rewrite — **DONE** (commits `e569370`, `d37fea0`, `1c96c89`)

Regenerated `TAMNet_Paper.docx` with the new `paper/rewrite_paper.py`
(results only, no typed numbers). Legacy preserved as
`TAMNet_Paper.pre-audit.docx`. See `CHANGELOG_FIXES.md` § Phase 5 for the
item-by-item status. **247 paragraphs · 12 tables · 16 figures.**

- [x] Abstract/Intro match new results; WT/TC/ET together; limitations stated;
      Grad-CAM now **measured** (not missing) and reported as a negative result.
- [x] Related Work rewritten; [3] and [33] cited in body; gap without over-claim.
- [x] Method: equations for M_s, M_c, fusion, L_cls, L_seg, MC-dropout;
      notation clash and broken glyphs gone; **five corrections** D1–D6 called out.
- [x] Sections IV+V merged into one Experiments section (A–K).
- [x] Dataset section from `dataset_table.csv` + leakage + patient-level +
      domain gap.
- [x] Implementation details table read from `run_config.json`.
- [x] 12 tables generated from `results/` only, each with an inline `Source:`.
- [x] All 16 Phase 4 figures inserted, captions carry dataset/split/N/error-bar.
- [x] Discussion answers "does TAMNet beat baselines?" honestly (**No**).
- [x] Limitations + Data/Code Availability + "not for clinical use".
- [x] References: [28] → Y. Ovadia, dataset/benchmark citations added;
      **`[VERIFY]` remains on [9], [22], [34], [36]** (unresolved provenance).
- [x] "clinically ready" / "near-real-time" removed; repetition reduced.

## Phase 6 — Reproducibility & QA — **DONE** (this commit)

- [x] `requirements.txt` pinned (torch 2.11.0+cu128, numpy 2.5.2, sklearn 1.9.0).
      Seeds 42–46 recorded per run; split SHA-256 pinned.
      **`environment.yml` / `Makefile` / `run_all.sh` / `run_all.ps1` — DONE**
      (stage list in `CHANGELOG_FIXES.md` § Phase 6).
- [x] Unit tests: `pytest tests -q` → **113 passed** (split leakage, module
      shapes, MC-dropout BN-eval / dropout-active, metrics vs sklearn,
      corruption tests, resume determinism).
- [x] `scripts/check_paper_numbers.py`: 194,916 artifact scalars → 1,174,251
      rendered forms → **920/920 numeric tokens traced, 0 untraced**
      (`results/phase6_paper_numbers.json`).
- [x] `scripts/check_figures.py`: figure/PDF/manifest, numbering, in-text
      `Fig. N` and `Table X` resolution, source notes — **0 warnings, 0 failures**.
- [x] `CHANGELOG_FIXES.md`: DONE / PARTIAL / BLOCKED per item + old→new
      headline table + manual steps remaining.

---

## Headline claims to trace (paper value → must map to a `results/` file)

| # | Claim in TAMNet_Paper.docx | Paper value | Status |
|---|---------------------------|-------------|--------|
| 1 | Internal test accuracy / macro-F1 | 96.45% / 0.9847 | **RE-RUN** → 0.9360 ± 0.0110 / 0.9355 ± 0.0109 (n = 5 seeds) — `results/phase3/agg/combined_summary.json` |
| 2 | External BRISC accuracy | 97.70% | **RE-RUN** → full 0.9634 ± 0.0118, de-duplicated 0.9234 ± 0.0218 — `external_eval_summary.json` |
| 3 | External Kaggle accuracy (+F1/MCC/κ) | 99.16% / 0.9916 / 0.9888 / 0.9887 | **RE-RUN** → full 0.9788 ± 0.0050, de-duplicated 0.9431 ± 0.0073 — `external_eval_summary.json` |
| 4 | Kaggle confusion matrix | 300/304/296/400 correct | **RE-RUN** → confusion matrices live in every `eval_*/metrics.json` (Fig 15 / Table VI) |
| 5 | Segmentation Dice WT / TC / ET | 0.9005 / 0.5803 / 0.6010 | **RE-RUN + RE-ATTRIBUTED** → same means reproduced from 45 per-case values, source = standalone 3-D U-Net (D3); `results/brats_unet3d_wt_tc_et.json` |
| 6 | Confidence: mean / correct / incorrect | 0.8821 / 0.8843 / 0.8232 | **RE-RUN** (D5 fix) → mean confidence / predictive entropy / MI in `uncertainty_summary.json` |
| 7 | Entropy / mutual information | 0.4778 / 0.0415 | **RE-RUN** (D5 fix) → `uncertainty_summary.json` |
| 8 | ECE / Brier / NLL | 0.0855 / 0.0736 / 0.2172 | **RE-RUN** (D5 fix, one ECE definition, 15 bins) → Table IX from `uncertainty_summary.json` |
| 9 | Robustness: clean, MRRS, rot 5°, contrast 0.5, brightness, noise | 96.45%, 0.7512, 95.96%, 22.40%, 54.17%, 56.81% | **RE-RUN** → 27 corruptions × 3 severities, MRRS 0.8508 ± 0.0035 (n = 5) — `results/phase3/robustness.json` |
| 10 | Multi-seed (3): val F1, val acc, ECE, MRRS, latency | 0.9751±0.0035, 0.9530±0.0057, 0.0838±0.0056, 0.7088±0.0309, 46.27±2.48 ms | **RE-RUN** → 5 seeds, mean-only headline (never best seed), ddof=1 — `agg/combined_summary.json` |
| 11 | Backbone table internal accuracy | CNN-2D 95.26, DN121 97.95, R50 97.04, ENv2 98.06, ConvNeXt 97.42, Swin-T 97.09, ViT 97.52, TAMNet 96.45 (%) | **RE-RUN** → Table III, identical protocol + bootstrap CIs |
| 12 | Backbone table external accuracy | 96.40 / 98.40 / 98.00 / 97.90 / 97.50 / 99.20 / 97.80 / 97.70 (%) | **RE-RUN** → Table VI, full **and** de-duplicated |
| 13 | Efficiency: params, size, latency, FPS | 9,034,416, 34.46 MB, 48.63 ms, 20.6 FPS | **RE-RUN** → params **VERIFIED** 9,034,416; latency 50.76 ms single / 530.27 ms T=10 MC; hardware named — `efficiency.json` |

## Known contradictions to resolve (claim changes, never numbers)
All nine are resolved in the rewritten manuscript — each row says *how*.
- [x] BraTS 2024 has no pituitary/healthy class → **documented, not resolved by
      data**: the 4-class labels are not BraTS (`results/fill_evidence.json`);
      label **origin** stays `[CITATION NEEDED]`, and the paper says so.
- [x] "Independent external" for BRISC/Kaggle → **measured and rejected**:
      94.7 % of Kaggle Testing and 52–69 % of BRISC test overlap internal train
      (`results/leakage_report.json`); every external number is reported both
      full and de-duplicated (Table VI).
- [x] Macro-F1 > accuracy, val-F1 > val-acc → **re-measured** with the averaging
      rule stated; the legacy 0.9847 was an artifact of the leaky split.
- [x] TAMNet 96.45 % < every baseline with no ablation → **ablation run**
      (Table V / Fig 4) and the headline rewritten as a negative result:
      TAMNet ties DenseNet-121 and trails four modern backbones.
- [x] 2-D 224×224 vs 3-D multimodal BraTS; WT from 7×7; TC/ET omitted from
      abstract → **re-attributed (D3)**: the 3-D system is a standalone U-Net,
      reported separately with its own per-case table (Table VIII), and the
      abstract now carries WT/TC/ET explicitly.
- [x] Contrast 0.5 → 22.40 %; MRRS undefined; 27 corruptions unlisted →
      **re-measured and defined**: MRRS formalised, all 27 × 3 cells reported
      (Fig 13/14), monotonicity checked (44/45 pairs).
- [x] Calibration underconfident (0.8821 vs 0.9645) → **re-measured** with the
      D5 fix (BN eval, dropout active): ECE, NLL, Brier, AURC in Table IX.
- [x] Method under-specified; notation clash F; broken glyphs → **rewritten**:
      every symbol defined, equations re-set, and findings D1–D6 printed in the
      Method section.
- [x] Sections IV/V duplicated; Related Work repeated Intro; interpretability
      promised but qualitative-only; [33]/[3] uncited → **rewritten**: sections
      merged, Related Work condensed, [3] and [33] cited in the body, and the
      interpretability claim replaced by the measured Grad-CAM comparison
      (Table XI).

## FILLs (ask user if not answerable from repo)
- [x] Internal dataset: BraTS 2024 task(s), modalities, HOW the 4 class labels were derived.
      → **RESOLVED (partial)** from `results/fill_evidence.json`: 11,148 images / 10,223 unique
      basenames, 43.7 % basename-match the canonical Kaggle zip, mixed filename families, 0 real
      label conflicts, **not** BraTS (no pituitary / no-tumour class there). Label **origin**
      remains `[CITATION NEEDED]` — no script, manifest, README or citation in the repo.
- [x] Kaggle dataset exact name/link (local copy: `data/external/kaggle_brain_tumor/`,
      Testing = 1,311 = 300/306/405/300 ✓, Training has no pituitary folder).
      → **RESOLVED (partial)**: `brain-tumor-mri-dataset.zip` = 7,023 entries = Training 5,712 +
      Testing 1,311, i.e. the canonical Kaggle "Brain Tumor MRI Dataset". Local `Training/` is an
      **incomplete** extraction (3,881, no pituitary). Owner slug/URL stays `[VERIFY]`.
- [x] Classifier training data source: BraTS-only, or also BRISC/Kaggle train splits?
      (Local `data/raw/classification/` Train 9,291 / Test 1,857, 4 classes — provenance?)
      → **RESOLVED (partial)**: neither BraTS nor BRISC (0 basename overlap with BRISC); it is a
      4-class 2-D JPEG/PNG pool substantially re-using the Kaggle release. Label origin
      `[CITATION NEEDED]`.
- [x] 3D→2D conversion: modality/channels, slice rule, patient-level split yes/no.
      → **RESOLVED**: there is **no** 3-D→2-D conversion in the classification path. BraTS is
      segmentation-only (1,000 NIfTI, 200 seg targets, patient-level 70/15/15 manifest seed 42).
      WT/TC/ET Dice belong to the standalone 3-D U-Net (D3).
- [x] Paper file path = `TAMNet_Paper.docx` (root) — confirm this is the single source.
      → **CONFIRMED**: `D:\BrainTumorDiagnosis\TAMNet_Paper.docx` (230 paragraphs, 8 tables).
      The legacy generator `paper/generate_paper.py` still points at `D:\Projects\...` and is not
      the source of truth; Phase 5 writes the docx from `results/` via `paper/rewrite_paper.py`.
