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

- [ ] A. Ablation (7 configs): DenseNet121 cls-only; +spatial; +spatial+channel;
      +aux seg only; full TAMNet; λ_seg ∈ {0, 0.1, 0.25, 0.5, 1.0}; t-only vs g+t fusion.
      **Progress 2026-11-21**: harness flags shipped (`--channel-attention`,
      `--no-attention-fusion`, `--fusion-mode t`, `--seg-weight`, `--seg-classes`,
      `--protocol brisc`) — commit `f741c9c`. Runs: +mc & +concat queued
      (queue_stage2a), t-only queued (queue_stage2b), λ_seg sweep = 5 BRISC runs
      queued (only local data with aligned masks; no_tumor gets all-zero mask).
      "+aux seg only" is covered by the λ_seg sweep (λ=0 vs λ>0 isolates the
      aux-seg contribution) — full-TAMNet and +spatial rows = the 5-seed headline.
- [ ] B. Fair baselines, identical protocol: DenseNet121, ResNet50, EfficientNetV2-S,
      ConvNeXt-T, Swin-T, ViT-B/16, shared-encoder + U-Net-style multi-task baseline.
      (Paper Table I claims TAMNet 96.45% < all of them — must be reproduced honestly.)
      **Progress**: 6 torchvision backbones queued (queue_stage2a/b, seed 42, 30 ep);
      shared-encoder U-Net multi-task baseline = TODO (needs design).
- [ ] C. Metrics audit from saved predictions with scikit-learn; resolve
      **macro-F1 0.9847 > accuracy 96.45%** and **val F1 0.9751 > val acc 0.9530**;
      state averaging used.
- [ ] D. Per-dataset confusion matrices (internal, BRISC, Kaggle, before/after de-dup)
      + bootstrap 95% CIs (1,000 resamples) on every accuracy.
- [ ] E. Segmentation: dataset/task, WT/TC/ET label mapping, slice selection,
      patient-held-out eval; Dice + HD95 for WT/TC/ET (paper: 0.9005/0.5803/0.6010);
      BRISC segmentation split vs published Swin-HAFNet if verifiable; U-Net baseline.
- [ ] F. Uncertainty/calibration on TEST + each external set: reliability diagram,
      ECE 15 bins, MC dropout T ∈ {5,10,20,30}, MC + temperature scaling (fit on VAL
      only), 5-model deep ensemble; confidence correct vs incorrect; AUROC of entropy/MI;
      risk-coverage + AURC; selective accuracy at 90/95% coverage; state miscalibration
      direction (paper: underconfident).
- [ ] G. Robustness: unit-test corruption pipeline first (8 samples/type, pixel-range
      asserts, identical normalization); explain **contrast 0.5 → 22.40%**; define MRRS
      formally; list all 27 corruptions (type × severity); retrain with intensity
      augmentation + per-image normalization; before/after; MRRS mean±std across seeds
      (resolve headline 0.7512 best-seed vs 0.7088±0.0309 3-seed).
- [ ] H. Interpretability: Grad-CAM [33] + attention-map extraction; attention–mask
      overlap (mass inside tumor mask, pointing game) on BRISC vs random and vs
      DenseNet121 Grad-CAM; add optional attention-supervision loss if not aligned;
      report with/without. (Paper §V-G currently promises qualitative-only.)
- [ ] I. Efficiency: fixed protocol (20 warmup, 200 timed, batch 1, cuda synchronize),
      hardware named; single pass AND full 10-pass MC latency for TAMNet and every
      baseline; params and MiB. Resolve whether **48.63 ms** includes 10 MC passes;
      drop "near-real-time" unless supported.
- [ ] J. Cross-dataset matrix: train BraTS-derived / BRISC-train / both × test on every
      other set; state which setting each paper number comes from.

## Phase 4 — Figures (single script `scripts/make_figures.py` → `paper/figs/`,
## vector PDF + 600-dpi PNG, `figNN_slug`, IEEE widths, ≥8 pt serif, no in-plot titles,
## colorblind-safe model→color map, error bars + N in caption; `captions.md` +
## `figure_sources.json`; log `MISSING:` and skip rather than fabricate)
- [ ] Fig 1 `architecture` (shapes via forward hooks) · Fig 2 `data_pipeline` ·
      Fig 3 `samples` (4 classes × 3 datasets, masks) · Fig 4 `ablation` (7 configs,
      SD over 5 seeds + λ_seg sweep) · Fig 5 `pareto` (params/latency vs accuracy) ·
      Fig 6 `confusion` (row-normalized, internal/BRISC/Kaggle ± de-dup) ·
      Fig 7 `seg_qualitative` (best/median/worst rule-based) · Fig 8 `seg_box`
      (WT/TC/ET + BRISC vs U-Net) · Fig 9 `reliability` (softmax / MC T=10 / MC+temp /
      ensemble) · Fig 10 `uncertainty` (histograms, risk-coverage+AURC, AUROC bars) ·
      Fig 11 `gradcam_attention` · Fig 12 `attention_alignment` ·
      Fig 13 `robustness` (corruption×severity heatmap, original vs augmented) ·
      Fig 14 `intensity_curves` · Fig 15 `generalization` (train-source × test-set)
- [ ] Supplementary `paper/figs/supp/`: S1 training curves · S2 leakage pHash ·
      S3 per-class ROC/PR · S4 per-class P/R/F1 · S5 ECE/AUROC/latency vs MC passes ·
      S6 t-SNE/UMAP of fused z · S7 failure gallery (12 worst) · S8 class counts ·
      S9 per-seed dots + 95% t-interval · S10 latency violins
- [ ] `scripts/check_figures.py`: figure + source JSON exist, N matches caption,
      figures cited in text in order, printed numbers equal table values; re-render 2
      figures at final column width.

## Phase 5 — Paper rewrite (edit TAMNet_Paper.docx source in place)
- [ ] Abstract/Intro match new results; WT/TC/ET together; state contrast-sensitivity
      limitation; remove interpretability promise unless 3H done; cut duplication.
- [ ] Related Work: remove copied intro paragraphs; add verified multi-task
      classification+segmentation methods; rewrite gap without over-claiming.
- [ ] Method: explicit equations for M_s, M_c, fusion, L_cls, L_seg, MC-dropout
      placement/rate; fix duplicated sentence ("Let the input MRI image be
      represented by"), notation clash F, seg-input inconsistency (F vs F_t), and all
      broken glyphs (Ⓧ, ⊙, ┬, "Cdenote"/"xdenotes").
- [ ] Merge Sections IV and V into one Experiments section; each full number appears once.
- [ ] Dataset section from `dataset_table.csv` + leakage results + patient-level split
      statement + domain-gap discussion (BraTS NIfTI multimodal vs JPEG T1 sets).
- [ ] Implementation details (optimizer, LR schedule, epochs, batch, augmentation,
      pretrained weights, early stopping, loss details, hardware, framework versions,
      seeds).
- [ ] Tables generated from `results/` only: dataset; ablation; baselines w/ params and
      latency; per-class; external before/after de-dup; segmentation (+U-Net, +BRISC);
      calibration/uncertainty; corruption list; seeds.
- [ ] Insert all Phase 4 figures with takeaways; captions state dataset, split, N,
      error-bar meaning.
- [ ] Discussion: state honestly whether TAMNet beats plain DenseNet121 (paper Table I:
      96.45% vs 97.95%); reframe contribution if segmentation doesn't help
      classification.
- [ ] Limitations paragraph (public datasets, 2D slices, domain gap, intensity
      sensitivity, no prospective validation) + Data/Code Availability + "research
      prototype, not for clinical use".
- [ ] References: fix [28] → Y. Ovadia (paper already has "A. Ovadia" — verify),
      verify [22] first author (Z. vs Q. Yang), add authors to [9], consistent DOIs,
      cite [3] and [33] in body text, add missing dataset/benchmark citations
      (BraTS 2024, BRISC 2025 Fateh et al. arXiv:2506.14318 / DOI
      10.1038/s41597-026-06753-y, Kaggle set, Hendrycks & Dietterich corruptions);
      mark unverifiable `[VERIFY]`.
- [ ] Reduce "unified/framework" repetition; remove "clinically ready",
      "near-real-time" unless supported.

## Phase 6 — Reproducibility & QA
- [ ] `requirements.txt`/`environment.yml`, `run_all.sh` or Makefile reproducing every
      table/figure; fixed seeds (torch/numpy/random, deterministic cuDNN where possible).
- [ ] Unit tests: split-leakage, module shapes, MC-dropout (BN eval / dropout active),
      metrics vs sklearn, corruption tests.
- [ ] `scripts/check_paper_numbers.py`: every metric in paper == `results/*.json`.
- [ ] Final `CHANGELOG_FIXES.md`: DONE / PARTIAL / BLOCKED per item, old vs new numbers,
      manual steps remaining.

---

## Headline claims to trace (paper value → must map to a `results/` file)

| # | Claim in TAMNet_Paper.docx | Paper value | Status |
|---|---------------------------|-------------|--------|
| 1 | Internal test accuracy / macro-F1 | 96.45% / 0.9847 | TODO-RERUN (val tail = 1,394 × 100% No Tumor → F1 artifact; 74.3% of test near-dup of train — `split_audit.json`, `leakage_report.json`) |
| 2 | External BRISC accuracy | 97.70% | TODO-RERUN (80.3% of BRISC test overlaps internal train, 616 byte-identical — `leakage_report.json`) |
| 3 | External Kaggle accuracy (+F1/MCC/κ) | 99.16% / 0.9916 / 0.9888 / 0.9887 | TODO-RERUN (94.7% of Kaggle Testing inside internal train, 960 byte-identical — `leakage_report.json`) |
| 4 | Kaggle confusion matrix | 300/304/296/400 correct | TODO-RERUN |
| 5 | Segmentation Dice WT / TC / ET | 0.9005 / 0.5803 / 0.6010 | TODO-RERUN (**D3: provenance = standalone UNet3D** `checkpoints/segmentation/unet3d`, not TAMNet's aux branch; TAMNet aux branch had `L_seg≡0` — D2) |
| 6 | Confidence: mean / correct / incorrect | 0.8821 / 0.8843 / 0.8232 | TODO-RERUN (D5: pre-Phase-2 MC path had BN-in-train bug) |
| 7 | Entropy / mutual information | 0.4778 / 0.0415 | TODO-RERUN (D5: MC path bug fixed Phase 2) |
| 8 | ECE / Brier / NLL | 0.0855 / 0.0736 / 0.2172 | TODO-RERUN (D5: MC path bug fixed Phase 2; two non-interchangeable ECE impls — AUDIT §7) |
| 9 | Robustness: clean, MRRS, rot 5°, contrast 0.5, brightness, noise | 96.45%, 0.7512, 95.96%, 22.40%, 54.17%, 56.81% | TODO-RERUN (MRRS undefined, 27 corruptions unlisted) |
| 10 | Multi-seed (3): val F1, val acc, ECE, MRRS, latency | 0.9751±0.0035, 0.9530±0.0057, 0.0838±0.0056, 0.7088±0.0309, 46.27±2.48 ms | TODO-RERUN (val F1 > val acc; best-seed headline 0.7512 vs mean 0.7088) |
| 11 | Backbone table internal accuracy | CNN-2D 95.26, DN121 97.95, R50 97.04, ENv2 98.06, ConvNeXt 97.42, Swin-T 97.09, ViT 97.52, TAMNet 96.45 (%) | TODO-RERUN (no ablation, identical protocol unproven) |
| 12 | Backbone table external accuracy | 96.40 / 98.40 / 98.00 / 97.90 / 97.50 / 99.20 / 97.80 / 97.70 (%) | TODO-RERUN |
| 13 | Efficiency: params, size, latency, FPS | 9,034,416, 34.46 MB, 48.63 ms, 20.6 FPS | TODO-RERUN (hardware unstated; MC-pass inclusion unclear) |

## Known contradictions to resolve (claim changes, never numbers)
- [ ] BraTS 2024 has no pituitary/healthy class, yet 4-class results incl. pituitary are
      reported — document where the 4-class labels actually come from.
- [ ] "Independent external" for BRISC/Kaggle unproven (BRISC collated from public sets;
      Kaggle overlaps internal train).
- [ ] Macro-F1 > accuracy and val-F1 > val-acc — averaging undefined/implausible.
- [ ] TAMNet 96.45% < every baseline with no ablation.
- [ ] 2D 224×224 input vs 3D multimodal BraTS unexplained; WT from 7×7 map; TC/ET
      omitted from abstract.
- [ ] Contrast 0.5 → 22.40% (near chance); MRRS undefined; 27 corruptions unlisted.
- [ ] Calibration underconfident (0.8821 vs 0.9645); correct/incorrect gap tiny.
- [ ] Method under-specified (M_s, M_c, fusion, L_cls, L_seg, C, hyperparameters);
      notation clash F; broken equation glyphs; duplicated sentence.
- [ ] Sections IV/V duplicate results; Related Work repeats Intro; "interpretability
      analysis" promised but qualitative-only; [33]/[3] never cited in body.

## FILLs (ask user if not answerable from repo)
- [ ] Internal dataset: BraTS 2024 task(s), modalities, HOW the 4 class labels were derived.
- [ ] Kaggle dataset exact name/link (local copy: `data/external/kaggle_brain_tumor/`,
      Testing = 1,311 = 300/306/405/300 ✓, Training has no pituitary folder).
- [ ] Classifier training data source: BraTS-only, or also BRISC/Kaggle train splits?
      (Local `data/raw/classification/` Train 9,291 / Test 1,857, 4 classes — provenance?)
- [ ] 3D→2D conversion: modality/channels, slice rule, patient-level split yes/no.
- [ ] Paper file path = `TAMNet_Paper.docx` (root) — confirm this is the single source.
