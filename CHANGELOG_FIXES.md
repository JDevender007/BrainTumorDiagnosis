# CHANGELOG — Audit fixes (`fix/paper-audit`)

Running log of every audit action. Final pass (Phase 6) will restate each line
as DONE / PARTIAL / BLOCKED with old → new numbers.

Rules in force: no invented metrics (unsourced = `TODO-RERUN`); when evidence
contradicts a claim, the **claim** changes, never the number; test set once per
final config; original files never deleted (git history / `legacy/main`).

---

## Phase 0 — Recon (commits `a2666f1`, `4fd1b1d`, `03979b5`, `8b806e9`)

| Item | Status | Notes |
|---|---|---|
| Full repo recon → `docs/AUDIT_REPORT.md` | DONE | 5 biggest findings: degenerate 100 %-No-Tumor val tail; F1 0.9847 = arithmetic artifact of that split; `M_c` never instantiated; Sep reruns contradict paper; MC-dropout `self.train()` BN bug |
| All paper numbers flagged `TODO-RERUN` | DONE | `docs/TODO.md` headline table (13 rows) |
| Structure tidy (docs/, scripts/, results/) | DONE | fixed dead `D:\Projects\...` paths in 4 runner scripts |
| GitHub `main` = this code | DONE | old Aug-2026 snapshot preserved as `legacy/main` (`52f6537`) |

## Phase 1 — Data integrity (commit `72390f0`, 2026-11-16)

Report: **`docs/PHASE1_REPORT.md`** · evidence: `results/leakage_report.json`,
`results/split_audit.json`, `results/dataset_table.csv`,
`results/leakage_thumbs/` (26 images).

| Item | Status | Old → new / notes |
|---|---|---|
| `scripts/leakage_check.py` exact + pHash/dHash across all 6 sources | DONE | 22,340 images, 68 s, 0 unreadable; thresholds eyeball-validated (`boundary_ph4/6.jpg` = same scan) |
| Kaggle Testing independence | **CLAIM CHANGED** | paper "independent external 99.16 %" → **94.7 % (1,242/1,311) of that set is inside internal train**, 960 byte-identical. Number itself → `TODO-RERUN` on de-duplicated set (Phase 3) |
| BRISC test independence | **CLAIM CHANGED** | "independent external 97.70 %" → **80.3 % overlap with internal train** (616 byte-identical). → `TODO-RERUN` |
| Internal test purity | **CLAIM CHANGED** | 96.45 % test → **74.3 % (1,379/1,857) of test images are re-encoded copies of train images** (0 byte-identical). → `TODO-RERUN` |
| F1 0.9847 > accuracy 96.45 % | EXPLAINED | scheme C val tail = **1,394 images, 100 % No Tumor** (`split_audit.json`). Claim to be rewritten (Phase 5); number never touched |
| Subject-level split audit | DONE | classification data has **no patient IDs** → image-level only (paper limitation, Phase 5); BraTS manifest is patient-level, 0 cross-split overlap (129/26/45 cases) |
| Label consistency on shared images | DONE | **12,893 / 12,893** exact cross-source pairs agree; 0 disagreements |
| BRISC provenance | DONE | bundled manifest sha256: **6,000 / 6,000** match; citation = Fateh et al., arXiv:2506.14318 |
| `results/dataset_table.csv` | DONE | 7 rows; provenance gaps marked `[VERIFY]` / `[CITATION NEEDED]` |
| Shared per-image normalization | PARTIAL | `datasets/normalization.py` single-sources the (already-identical) ImageNet stats used by the paper + provides 1–99 percentile z-score; **switching training to it deferred to Phase 3** because it invalidates all numbers (requires full re-run). 5 call sites refactored, 0 behaviour change, suite 82/82 green |
| Metrics before/after de-duplication | **TODO-RERUN** | needs model re-evaluation (Phase 3); deliberately not computed |

Open user FILLs (block Phase 5 wording only): #1 internal dataset provenance,
#2 exact Kaggle dataset name/link, #3 BraTS role in classification.

---

## Phase 2 — Code vs. paper consistency (commit `<pending>`)

Report: **`docs/ARCHITECTURE.md`** (executed-model spec, discrepancy register D1–D9,
param/shape accounting). New tests: `tests/test_tumor_aware_phase2.py` (18).

| Item | Status | Old → new / notes |
|---|---|---|
| M_s / M_c exposed + documented | **CLAIM CHANGED (D1)** | `ChannelAttention` **never instantiated** in any historical run → paper's "spatial **and channel** attention" (abstract/intro/method/conclusion) is historically false. `use_channel_attention` flag added, **default False** (executed model unchanged). Claim rewrite → Phase 5; "+spatial+channel" ablation → Phase 3A |
| Fusion op identified + renamed `fuse()` | **CLAIM CHANGED (D4)** | Φ = **2-way softmax-weighted convex combination of projected g,t + LayerNorm** — not the concatenation `[060]` asserts. `fuse()` documents exact math; concat ablation = `use_attention_fusion=False`. `[060]` rewrite → Phase 5 |
| Segmentation branch documented | **CLAIM CHANGED (D3)** | paper's Dice **0.9005/0.5803/0.6010 `[231]`** trace to **standalone `models/segmentation.py::UNet3D`** (`checkpoints/segmentation/unet3d`, metadata epoch 30 val_dice_mean 0.5243) — not TAMNet's aux branch. Executed decoder = 7×7-map, no skips, ×4/×2→56→224 (documented). `seg_decoder="multiscale"` U-Net variant added **default-off**. Attribution fix → Phase 5; aux-branch Dice re-run → Phase 3E |
| L_cls / L_seg documented | **CLAIM CHANGED (D2)** | L_cls = CE(**label_smoothing=0.1, no class weights** in multitask path). **L_seg ≡ 0 in every historical run** (no runner passes `seg_targets`; `master_pipeline.py:415` passes `None`; `DiceLoss3D` is 5D-only vs 4D `seg_logits`) → "region-level supervision" claim false for training. Code now: warn-once `RuntimeWarning` + `DiceLoss2D` 2D/3D dispatch (path was dead → 0 historical behaviour change). Paper claim → Phase 5; λ_seg sweep needs real masks (Phase 3A/E) |
| MC dropout BN bug | **FIXED (D5)** | old `predict_with_uncertainty` called `self.train()` → BN in train mode for all T=10 passes. Now: `eval()` + Dropout-only re-activation + mode restore. **All prior uncertainty numbers (ECE, entropy, MI, T=10 curves) → `TODO-RERUN`** (Phase 3F) — no silent number swap |
| C=3 + implementation details | DOCUMENTED (D8) | C=3, global BN, t-MLP (BN+ReLU+Dropout 0.2), classifier 0.3/0.15, CBAM order, T=10, mask path dead (D6), `uncertainty_logits` unused (D7) → `docs/ARCHITECTURE.md` §1; paper wording → Phase 5 |
| Parameter claim | **VERIFIED** | probe: **9,034,416 total** = paper's 9.03M exactly (seg_head 1,254,212; tumor_pool 263,010; fusion 460,546; classifier 99,716; uncertainty_head 1,028; encoder+global_bn 6,955,904); multiscale decoder variant = 12,094,576 |
| Historical numbers | UNTOUCHED | all new flags default-off → executed model byte-identical by construction; no metric/checkpoint/`results/` artifact altered; original files preserved |
## Phase 3 — Experiments, 5 seeds _(pending; needs GPU-budget approval)_
## Phase 4 — Figures _(pending)_
## Phase 5 — Paper rewrite _(pending)_
## Phase 6 — Reproducibility & QA _(pending)_
