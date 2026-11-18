# TAMNet — Executed-Model Architecture & Code↔Paper Sync

**Phase 2 deliverable.** This document is the *authoritative description of the model that
actually ran* (as of commit on `fix/paper-audit`), the complete register of divergences
between that code and `TAMNet_Paper.docx`, and the hand-off note for Phase 3 (experiments)
and Phase 5 (paper rewrite). Every number here was produced by a probe or a committed
`results/` artifact — nothing is estimated. To re-derive the parameter/shape tables, run
the commands in §8.

Paper paragraphs are cited as `[NNN]` (extracted from `TAMNet_Paper.docx` via
`scripts/extract_paper.py`; full text: temp `paper_text.txt`).

---

## §1 Executed model — `TumorAwareMultiTaskNet` (`models/tumor_aware.py`)

Defaults below are the **historically executed** configuration (`TumorAwareConfig()`
defaults). New Phase 2 flags exist but default to the historical behaviour, so no prior
number changes by construction.

| Stage | Exact operation (code) | Paper §III says | Sync |
|---|---|---|---|
| Input | `C = 3`, 224×224 (probe `zeros(1,3,224,224)` at `models/tumor_aware.py:448`) | "input resolution of pixels" `[116]` — C never stated | Phase 5: state C=3, H=W=224 |
| Encoder | torchvision **DenseNet121**, `classifier → Identity`; forward hook on `encoder.features` captures **F** = 1024ch @ 7×7 (`:407-434`) | "shared DenseNet121-based encoder" `[117]` | ✅ |
| Global pool | `g = BN(GAP(F))`, 1024-dim; `global_bn = BatchNorm1d(1024)` (`:321-322, :517`) | global features `g`, BN not stated | Phase 5: state global BN |
| `M_s` (spatial) | `SpatialAttention(7)`: conv 7×7 over `cat[channel-avg, channel-max]` → sigmoid → elementwise multiply (`:52-66`) | spatial attention (§III.C) | ✅ exists |
| `M_c` (channel) | `ChannelAttention` SE-style, reduction=16, shared MLP on avg+max pools → sigmoid (`:69-88`) — **constructed only when `use_channel_attention=True`, default `False`** (`:41, :121-123`) | "spatial **and channel** attention" `[001, 009, 026, 110, 117, 345]` | ❌ **D1**: not in executed model; flag added (default off) for Phase 3 ablation "+spatial+channel" |
| Attention order | CBAM order: **spatial THEN channel** (`:138-140`) | not stated | Phase 5: state order |
| `t` (tumor vector) | `t = MLP(GAP(M_s(...)))` = `Linear(1024→256) → BatchNorm1d → ReLU → Dropout(0.2)` (`:124-130, :141-142`); mask branch `x·mask` (bilinear-interpolated) only if `seg_mask` given (`:132-137`) | `t = GAP(F_t)` — MLP/BN/Dropout unstated | Phase 5: state MLP spec |
| Mask input | `forward(x, seg_mask=None, ...)` — **no caller ever passes `seg_mask`** | mask-assisted pooling implied | ❌ **D6**: mask path dead; document or ablate (Phase 3H uses seg supervision instead) |
| Fusion Φ | `AttentionFusion`: `g' = W_g g`, `t' = W_t t` (Linear→256); `w = softmax(MLP([g';t']))` (512→256→2); `z = LayerNorm(w₀·g' + w₁·t')` (`:145-171, :488-508`) | "z = Φ(g, t)"; `[060]` "where [⊥] denotes **concatenation** and [Φ] represents the implemented feature-fusion operation" | ❌ **D4**: executed op is a **2-way softmax-weighted convex combination + LayerNorm, not concatenation**. `fuse()` now names the op; ablation `use_attention_fusion=False` = plain concat |
| Classifier | `256 → BN → ReLU → Dropout(0.3) → 128 → ReLU → Dropout(0.15) → 4` (`:336-345`) | 4-class head, details unstated | Phase 5: state |
| Seg head (executed) | `SegmentationHead`: conv3×3 1024→128 +BN+ReLU → bilinear ×4 → conv3×3 128→64 +BN+ReLU → bilinear ×2 → conv1×1 →4ch@56 → `F.interpolate` to 224 (`:174-208, :533-540`). **No skips, decodes the 7×7 map alone** | "s = S(F) on F" `[117]` — resolution/skips unstated | ⚠ input matches paper; **D3/D8**: 7×7-map decoder + no-skips must be stated; `seg_decoder="multiscale"` variant (U-Net taps `transition1/2/3`, densenet121-only, `:211-276, :436-445`) added default-off for the Phase 3 decoder ablation |
| Uncertainty head | `Dropout(0.3) + Linear(256→4)` applied to fused z (`:279-289, :542-543`) — output stored as `uncertainty_logits` and **never consumed by any training/eval path** | MC dropout "extends the deterministic classifier" `[110]` | ❌ **D7**: unused head; MC uncertainty in practice comes from the *classifier's* dropout sites via `predict_with_uncertainty` |
| MC dropout | `predict_with_uncertainty(x, num_passes=10)`: `model.eval()` → re-activate **only** Dropout layers → T=10 forward passes under `no_grad` → mean/var/entropy/MI; restores prior mode (`:547-600`). **Phase 2 fix**: old code called `self.train()`, putting BatchNorm in train mode for the 10 passes (batch-statistics noise + mutated running stats) | MC dropout T=10 (§III.G) | ✅ now correct; **all pre-Phase-2 uncertainty numbers → `TODO-RERUN` (D5)**. Dropout sites sampled: classifier 0.3/0.15, tumor-pool MLP 0.2, uncertainty head 0.3 |

Forward output shapes (probe, batch=2): `logits (2,4)`, `probabilities (2,4)`,
`global_features (2,1024)`, `tumor_features (2,256)`, `fused_features (2,256)`,
`seg_logits (2,4,224,224)`, `uncertainty_logits (2,4)`.

---

## §2 Losses actually executed (`training/multitask_trainer.py`)

| Term | Executed | Paper §III.F | Sync |
|---|---|---|---|
| `L_cls` | `nn.CrossEntropyLoss(label_smoothing=0.1)`, **no class weights** in the multitask path (`:73-98, :161-166`). (The *standalone* classification trainer supports inverse-frequency class weights — `classification_trainer.py:402-441` — a different trainer; which path produced the paper's numbers is pinned in Phase 3.) | "L_cls" (CE implied); smoothing/weights not stated | Phase 5: state smoothing 0.1, no class weights (multitask) |
| `L_seg` | **≡ 0 in every historical run** — see D2. `MultiTaskLoss` now (a) warns **once** per run when `seg_logits` arrive without `seg_targets` (`:100-117`), and (b) dispatches `DiceLoss2D` for 4D logits / `DiceLoss3D` for 5D volumes when targets *are* supplied (`:119-127`) | `L_seg` = Dice-based; `λ_seg = 0.5` | ❌ **D2**: claim must change in Phase 5; code fixed so the branch *can* be trained (Phase 3E) |
| Total | `1.0·L_cls + 0.5·L_seg` (`cls_weight=1.0`, `seg_weight=0.5` defaults `:76-77`) — with `L_seg ≡ 0` this reduces to **`1.0·L_cls`** | `λ_cls=1.0, λ_seg=0.5` | weights match; effective loss was CE-only |

---

## §3 Discrepancy register (paper claim → code reality → resolution)

D1–D9: D1–D7 are Phase 0/2 audit findings; D8 = underspecified details; D9 = derived
claim-level consequence of D1. "Resolution owner" says *which phase* fixes it.

| ID | Paper claim | Code reality (evidence) | Resolution |
|---|---|---|---|
| **D1** | "shared convolutional encoder with **spatial and channel attention**" — abstract `[001]`, intro `[009, 026]`, method `[110, 117]`, conclusion `[345]` | `ChannelAttention` defined but **never instantiated**; no call site anywhere in repo (Phase 0 §5) | **Claim changes (Phase 5)**: executed model = spatial attention only. Channel attention kept as opt-in flag → Phase 3 ablation "+spatial+channel" |
| **D2** | "auxiliary segmentation branch provides **region-level supervision**" `[001, 009, 110, 117, 345]`; `L_seg` term in total objective `[117]` | `criterion(outputs, labels)` — no runner passes `seg_targets`; `master_pipeline.py:415` passes `None` explicitly; other runners use plain CE; even with targets, `DiceLoss3D` rejects the 4D `seg_logits` → **`L_seg ≡ 0` in all historical runs, aux branch got no gradient** | **Claim changes (Phase 5)**: segmentation was evaluation-only, not a training loss (or re-run with real masks, Phase 3E). Code now warns + dispatches `DiceLoss2D` |
| **D3** | "TAMNet obtained Dice scores of 0.9005 (WT), 0.5803 (TC), 0.6010 (ET)" `[231]` | Numbers trace to the **standalone `models/segmentation.py::UNet3D`** (`checkpoints/segmentation/unet3d/best`, metadata: epoch 30, `val_dice_mean` 0.5243) — *not* TAMNet's aux branch | **Claim changes (Phase 5)**: attribute Dice to the U-Net3D baseline/segmentation model; TAMNet aux-branch Dice must be re-run (Phase 3E) or dropped |
| **D4** | `z = Φ(g, t)`; `[060]`: "[⊥] denotes **concatenation** and [Φ] represents the implemented feature-fusion operation" | Φ = `AttentionFusion`: 2-way softmax-weighted convex combination of projected g,t + LayerNorm (`:145-171, :488-508`) — **not concatenation** | **Claim changes (Phase 5)**: document the real op with an equation; report concat ablation (`use_attention_fusion=False`, Phase 3A) |
| **D5** | MC dropout T=10 for uncertainty `[110]` | Old `predict_with_uncertainty` called `self.train()` → **BatchNorm in train mode during all 10 passes** (noise + mutated running stats); only `evaluation/uncertainty.py` had the correct pattern | **Fixed (Phase 2)**: eval + Dropout-only re-activation + mode restore (`:547-600`); **all prior uncertainty numbers (ECE, entropy, MI, T=10 curves) → `TODO-RERUN`** |
| **D6** | Mask-assisted tumor pooling implied by "tumor-aware" pooling | `seg_mask` parameter exists but **no caller ever passes it** — mask path dead | Phase 5: state pooling is attention-only, or Phase 3 ablates mask input |
| **D7** | "MC dropout extends the deterministic classifier" `[110]` | `uncertainty_logits` head computed but **unused by any downstream path**; MC samples the classifier's own dropout | Phase 5: describe MC on the classifier; head either removed or wired (Phase 3F) |
| **D8** | Under-specified: C=3, global BN, t-MLP (BN+ReLU+Dropout 0.2), 7×7-map no-skip decoder, classifier 0.3/0.15 drops, `label_smoothing=0.1`, no class weights (multitask), CBAM order, T=10 | All present in code (§1) | Phase 5: implementation-details paragraph |
| **D9** | "spatial **and channel** attention" in abstract/intro/conclusion (same sentences as D1) | Follows from D1 — historically false | Phase 5: same rewrite as D1 |

---

## §4 Parameter & shape accounting (traced probe)

Probe (re-runnable, §8): `TumorAwareMultiTaskNet(TumorAwareConfig(pretrained=False))`,
`sum(p.numel())` overall and per top-level module:

| Module | Params |
|---|---:|
| Encoder (DenseNet121 minus its 1000-way classifier) + `global_bn` | 6,955,904 |
| `seg_head` (lightweight decoder) | 1,254,212 |
| `tumor_pool` (M_s + MLP) | 263,010 |
| `fusion` (AttentionFusion) | 460,546 |
| `classifier` | 99,716 |
| `uncertainty_head` | 1,028 |
| **Total (all trainable)** | **9,034,416** |

- **9,034,416 = the paper's stated "approximately 9.03 million trainable parameters"**
  `[001]` and Table "params" row — the parameter claim is **verified exactly**.
- With `seg_decoder="multiscale"`: **12,094,576** total (+3,060,160 vs lightweight) —
  report both if the decoder ablation is kept (Phase 3A).
- `use_channel_attention=True` adds ~131k (SE MLP on 1024ch) — Phase 3 "+spatial+channel".

---

## §5 Phase 3 / Phase 5 hand-off

- **Phase 3 (experiments)** — every config below is a *new* run; test set is touched once
  per final config; headline = 5-seed mean±std, never the best seed.
  - Ablations map 1:1 to the flags: `use_channel_attention` (D1), `use_attention_fusion`
    concat vs Φ (D4), `seg_decoder` lightweight vs multiscale, `λ_seg ∈ {0, .1, .25, .5, 1}`
    — **λ_seg>0 requires real masks** (internal/kaggle sets have none; BRISC has aligned
    masks — dataset choice is a Phase 3 design decision).
  - All pre-Phase-2 numbers are `TODO-RERUN`: uncertainty suite (D5), segmentation (D2/D3),
    anything touching the leaked splits (Phase 1).
- **Phase 5 (paper rewrite)** — rewrite from §1–§3, never from memory: state C=3; replace
  the "spatial and channel attention" sentences (D1/D9); replace `[060]`'s concatenation
  sentence with the softmax-convex-combination equation (D4); re-attribute or drop the
  `[231]` Dice numbers (D3); state `L_seg` history honestly (D2); write the
  implementation-details paragraph from §1/D8.

---

## §6 Evidence trail

| Claim here | Evidence |
|---|---|
| D1 M_c never instantiated | Phase 0 `docs/AUDIT_REPORT.md` §5; grep `ChannelAttention` → no construction site |
| D2 L_seg ≡ 0 | call-site audit: `master_pipeline.py:415` (`None`), runners use `criterion(outputs, labels)`; `DiceLoss3D` 5D-only contract vs 4D `seg_logits` |
| D3 Dice = UNet3D | `evaluation/segmentation_evaluator.py` + `models/segmentation.py::UNet3D` + `checkpoints/segmentation/unet3d/metadata.json` (epoch 30, val_dice_mean 0.5243) |
| D4 fusion op | `models/tumor_aware.py:145-171, 488-508`; paper `[060]` |
| D5 MC fix | `models/tumor_aware.py:547-600`; prior bug per Phase 0 §5 |
| Param totals | §8 probe (this commit) |
| Dice 0.9005/0.5803/0.6010 | paper `[231]` (numbers untouched; provenance per D3) |
| Test coverage | `tests/test_tumor_aware_phase2.py` — 18 tests (M_c flag, decoder variants, `fuse()`, MC BN eval, loss contract) |

## §7 What Phase 2 deliberately did NOT change

- No metric, checkpoint, or `results/` artifact was altered or regenerated.
- All new code paths are default-off (`use_channel_attention=False`,
  `seg_decoder="lightweight"`, `use_attention_fusion=True`) → executed model identical
  to pre-Phase-2 by construction; `MC` fix changes *future* uncertainty numbers only
  (hence `TODO-RERUN`, never a silent number swap).
- Original files preserved (git history / `legacy/main`).

## §8 Reproduce the probes

```powershell
# parameter counts + forward shapes (from repo root, .venv)
.venv\Scripts\python.exe -X utf8 -c "import torch,sys; sys.path.insert(0,'.'); \
 from models.tumor_aware import TumorAwareConfig, TumorAwareMultiTaskNet; \
 m=TumorAwareMultiTaskNet(TumorAwareConfig(pretrained=False)); \
 print('total', sum(p.numel() for p in m.parameters())); \
 [print(n, sum(p.numel() for p in m.parameters() if k==n)) for n in ('seg_head','uncertainty_head','tumor_pool','fusion','classifier') for k in [None]]" 2>$null

# full suite (includes 18 Phase 2 tests)
.venv\Scripts\python.exe -X utf8 -m pytest tests -q
```
