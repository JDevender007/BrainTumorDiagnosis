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
## Phase 3 — Experiments, 5 seeds (in progress)

Split/harness: `scripts/phase3_split.py` → `results/phase3/split_manifest_clean.json`
(de-dup grouped 70/15/15 = 7,804/1,672/1,672, 0 clusters cross-split), harness
`phase3_{train,evaluate,queue}.py` + `phase3_status.ps1`.

| Item | Status | Notes |
|---|---|---|
| 30-epoch cap, all training | DONE (`681f52a`) | user directive 2026-11-19; prior 50-epoch work kept as `*_ep50budget*` dirs |
| GPU speed-ups (protocol-neutral) | DONE (`63b027b`) | cudnn.benchmark + TF32 + persistent workers + non_blocking H2D; single run was 68 % util / data-starved |
| Parallel queues (2 trainings at once) | DONE (`c60e1da`) | `--status-file` per queue; GPU → **99–100 % util**, ~5.5 min/epoch per run vs 7.45 sequential |
| `progress.json` best-lag fix | DONE (`9d43d0f`) | write progress AFTER best/patience update (reporting-only) |
| **Per-epoch checkpoint + auto-resume** | DONE (this commit) | `checkpoint_last.pt` after every epoch (atomic tmp→replace): model/optimizer/scheduler/scaler/best/patience/history/elapsed + all RNG (torch/cuda/numpy/python) + shuffle-generator state; `phase3_train.py` auto-resumes when the file exists and `training_result.json` doesn't (original `run_config.json` never overwritten); `--no-resume` to opt out. New tests `tests/test_multitask_resume.py` (4, incl. crash→resume ≡ uninterrupted). Suite **104/104** |
| Seed s42 (30-ep protocol) | DONE | best val F1(w) **0.9463 @ ep17**, 24/30 epochs (early stop), 2.98 h, `results/phase3/runs/tamnet_clean_s42/` |
| Seeds s43+s45 (parallel) | **CRASHED (OOM)** @ ep17/30 | 2026-11-19 22:26 IST: Windows error **1455 (paging file too small)** killed both trainers — `PBZ.exe` game + 2 trainings × 5 procs exhausted RAM. Preserved `*_ep17_oom_partial`: s43 best **0.9402 @ ep15**, s45 best **0.9457 @ ep15** (full history + best.pt). Old code had **no per-epoch checkpoint** → cannot resume mid-run; must restart from epoch 1 with new code. Lesson applied: resume feature ships in this commit |
| Seeds s44, s46 | see rows below | s44 completed; s46 crash→resume→crash→resume (both restarts via checkpointing) |
| Seed s44 | DONE | best val F1(w) **0.9522 @ ep26**, full 30 epochs, 1.97 h, `results/phase3/runs/tamnet_clean_s44/` |
| s46 crash #1 (`0xC0000409` @ ep8) | RESUMED | 2026-11-20 native fail-fast at ep9; `checkpoint_last.pt` present → relaunch continued at **ep9/30** with zero epoch loss (first live proof of the resume feature) |
| s46/+mc crash #2 (Win32 **1455**) | RESUMED | 2026-11-21 06:04 UTC: a **pytest run executed concurrently with both trainings** pushed commit into "low virtual memory" (Event 2004: python 5.18 GB + 5.15 GB + test-runner 2.6 GB) → dataloader collate failed in BOTH runs. s46 resumed at **ep16/30** (best 0.9444@11), `tamnet_clean_s42+mc` at **ep5/30**. Lesson: never run the test suite while trainings are live |
| Stage 2 queue A launched | RUNNING | `results/phase3/queue_stage2a.json` (5 runs, seed 42, 30 ep): `+mc` (D1), `+concat` (D4), densenet121, resnet50, efficientnet_v2_s — runs sequentially beside queue B (max 2 concurrent) |
| **Phase 3A wiring: λ_seg + t-only + BRISC** | DONE (`f741c9c`) | `--seg-weight` (λ_seg), `--seg-classes` (4 default / 2 for binary BRISC), `--fusion-mode g+t\|t` (t-only ablation, `TumorAwareConfig.fusion_mode`, `fuse()` returns `t`); `--protocol brisc` builds the paired split at runtime (labels from `classification_task/`, real masks from `segmentation_task/`, no_tumor → generated all-zero mask `results/phase3/brisc_empty_mask.png` since BRISC only annotates tumor slices); dataset yields `(img,label,mask)` triples → trainer passes `seg_targets` → **L_seg active for the first time in repo history** (smoke: `seg=0.6452`); val/eval loaders use index access (2-tuple contract unchanged for old runs); 9 new tests `tests/test_phase3_seg_fusion.py` |
| Seed s46 | DONE | best val F1(w) **0.9444 @ ep11**, early stop @ ep18/30 (2 restarts), 1.85 h total, `results/phase3/runs/tamnet_clean_s46/` |
| **Val-loader memory fix** | DONE (`6f800d8`) | every Windows DataLoader worker commits ~2 GB (torch import footprint); 4 train + 4 val workers × 2 parallel runs = 18 pythons ≈ **50 GB commit** → Win32 1455 cliff. Val now runs in-process (`num_workers=0`, one pass/epoch, no quality impact); train keeps its 4-worker pool. Commit with both runs live: 41.8/43.3 GB (was 50.1/50.9) |
| **Stage 1 headline: TEST eval, 5 seeds** | **DONE 2026-11-21** | test split used ONCE per seed via `phase3_evaluate.py` (n=1672), aggregated by `scripts/phase3_aggregate.py` → **`results/phase3/stage1_test_summary.json`**: accuracy **0.9360 ± 0.0110**, macro-F1 **0.9355 ± 0.0109**, weighted-F1 0.9361 ± 0.0110 (min 0.9258 / max 0.9478 macro-F1). **Headline = mean, never best seed.** Paper's 96.45% accuracy claim → rewrite in Phase 5 (measured 93.60 ± 1.10%) |
| Stage 2 queues relaunched | RUNNING | queue 2a (`+mc` resumes ep14, then +concat, densenet121, resnet50, efficientnet_v2_s) + queue 2b (t-only, leaky, convnext_tiny, swin_t, vit_b_16, 5× BRISC λ_seg sweep) — 2 concurrent, 30 epochs each |
| Stage 2a+2b COMPLETE (baselines/ablations, seed 42, val F1(w) best@ep) | **DONE 2026-11-23** | efficientnet_v2_s 0.9688@18 · swin_t 0.9641@26 · resnet50 0.9636@18 · convnext_tiny 0.9623@27 · +mc 0.9564@27 · +concat 0.9525@23 · densenet121 0.9514@7 · **TAMNet baseline 0.9463@17** · t-only 0.9383@25 · leaky-split 0.9871@29 (different split — explains historical ~96%) · **vit_b_16 0.9392@26**. Every clean-protocol baseline except vit beats TAMNet baseline → Table I heading for rewrite in Phase 5 |
| Win32 1455 recurrences (2026-11-23) | MITIGATED | 5 crashes/day from DataLoader collate `shared file mapping` when commit hit ~44/44 GB (Sekiro also stole VRAM → CUDA OOM once; killed with user approval). Fix: **all queues `--num-workers` 4→2** (worker commit halved; data order unchanged — seeded sampler) + games stay closed |
| **3-concurrent training** | ADOPTED (user directive 2026-11-23) | measured: 2-at-once 0.56 ep/min vs **3-at-once 0.71 ep/min (+28%)**, VRAM 2673/4096 MiB fits; queues split to keep entries disjoint (2d=[λ0.5], 2e=[λ1.0], 2f=[shared_unet]) |
| BRISC λ_seg sweep | **5/5 DONE 2026-11-23** | λ0 **0.9829@26** · λ0.1 **0.9829@27** · λ0.25 **0.9839@26** (best) · λ0.5 **0.9829@24** · λ1.0 **0.9779@14** (early stop ep21) — all full provenance in `results/phase3/runs/tamnet_brisc_*`; flat 0→0.5 plateau then drop at 1.0 → Phase 5 wording |
| **Ablation/baseline seed extension (Fig 4 SD over 5 seeds)** | RUNNING 2026-11-24 | restart recovery: server restart 2026-11-23 13:02 UTC killed all 3 queues; all 3 t-only runs resumed exactly from `checkpoint_last.pt` (s43 ep16, s44 ep14, s45 ep11) with zero epoch loss. **s44 t-only DONE 0.9515@15** (22 ep, early stop); s43 t-only ep24+ (best 0.9401@23), s45 t-only ep18+ (best 0.9390@15); queue 2h advanced to +mc s44 |
| 1455 crash #6 + workers 2→1 | MITIGATED 2026-11-24 | all 3 runs died together at commit 43.7/44.1 (pagefile then shrank to 33 GB, 3 runs need ~27 GB). Fix: **`--num-workers 1`** in queues 2g/2h/2i (profiling: workers idle at 3-4 % of one core; trainer main process is the real bottleneck at ~94 % of one core, so worker count does not change throughput). Commit dropped 43.7 → 37/39 GB, epoch time unchanged (~7.3-8 min). All 3 resumed from checkpoints, zero epoch loss |
| **s43/s44/s45 t-only DONE** | **DONE 2026-11-24** | s43 **0.9401@23** (30 ep) · s44 **0.9515@15** (22 ep early stop) · s45 **0.9419@23** (30 ep) — t-only ablation now has 4 seeds (s42 0.9383, s43 0.9401, s44 0.9515, s45 0.9419) |
| User-requested reboot 2026-11-24 | DONE | machine restarted after s45 t-only completed (no run could finish within 10 min); fresh pagefile raises the commit ceiling. **Resume after boot:** relaunch the 3 queues — `python -X utf8 scripts\phase3_queue.py --queue results\phase3\queue_stage2g.json --status-file results\phase3\queue_status_2g.json` (same for `2h`, `2i`); each skips completed runs and auto-resumes in-flight ones from `checkpoint_last.pt` |
| **Phase 3B: shared-encoder U-Net baseline** | LAUNCHED 2026-11-23 | `models/shared_unet.py` — same pretrained DenseNet-121 encoder as the cls-only row + U-Net decoder with 5 skips (11.94M params) + GAP/FC head; dict contract identical to TAMNet → same `MultiTaskLoss` (CE + λ·(Dice2D+BCE)); `phase3_train.py --model shared_unet` wiring; queue 2f (brisc, λ=0.25, seed 42, 30 ep, num-workers 2). Smoke-verified: logits (B,4), seg_logits (B,2,224,224) |
| AMD Radeon iGPU training | DECLINED (2026-11-23) | laptop iGPU (~512 MB, DirectML-only) can't run under installed CUDA torch 2.11 without downgrading the shared env; ~5-10× slower than RTX 3050 and adds RAM/commit pressure → 1455 risk. User advised to skip |
## Phase 3 — Shutdown checkpoint (2026-11-25 05:03 IST)

**Cause.** Server restart at 2026-11-24T17:07Z killed every phase3 process (queue driver + trainers). The scheduled midnight stop, `scripts/phase3_stop_after_epoch.py --timeout-min 20`, was therefore run with nothing live and exited `STOP: no active runs (nothing marked 'running' without a result)`. **No `results/phase3/shutdown_*.json` exists** — none was written, and none was fabricated.

**Verification (2026-11-25 10:33 IST).**
- phase3 python processes remaining: **0**
- 28 runs complete on disk (`training_result.json` present); none zero-byte.
- 3 runs were in flight; all have intact, non-zero checkpoints:

| run | epoch | best val F1 | `checkpoint_last.pt` | `best.pt` |
|---|---|---|---|---|
| `densenet121_clean_s45` | 1/30 | 0.934774 @ ep1 | 84.5 MB | 84.3 MB |
| `tamnet_clean_s43+concat` | 4/30 | 0.853742 @ ep4 | 97.0 MB | 96.9 MB |
| `tamnet_clean_s44+concat` | 7/30 | 0.914534 @ ep6 | 97.0 MB | 96.9 MB |

- Historical `*_partial*` dirs (`s43_ep17_oom_partial`, `s43_ep50budget_partial`, `s43_parallel_setup_partial`, `s43_workers4_partial`, `s45_ep17_oom_partial`) have **no `checkpoint_last.pt`** — expected: they predate the per-epoch checkpoint feature or were killed before it. Kept as-is, not deleted. `tamnet_clean_s45+mc_stale_20261124T1248` also has no checkpoints (stale dir, in no queue).

**Pending queue** — `results/phase3/queue_stage4_fast.json`, 12 entries, `--num-workers 4`:

- resumable (3): `densenet121_clean_s45` (ep1), `tamnet_clean_s43+concat` (ep4), `tamnet_clean_s44+concat` (ep7)
- not started (9): `densenet121_clean_s43`, `densenet121_clean_s44`, `shared_unet_brisc_s43+ls0.25+sc2`, `shared_unet_brisc_s45+ls0.25+sc2`, `shared_unet_brisc_s44+ls0.25+sc2`, `efficientnet_v2_s_clean_s43`, `efficientnet_v2_s_clean_s44`, `swin_t_clean_s43`, `swin_t_clean_s44`

**Gap.** `tamnet_clean_s45+concat` is pending in `queue_stage2i.json` but **absent from `queue_stage4_fast.json`** — it must be added when the fast queue is rebuilt, or the `+concat` ablation stays at n=3 instead of n=4.

**Resume commands** (verbatim from `queue_stage4_fast.json` args; `phase3_train.py` auto-resumes the 3 in-flight runs from `checkpoint_last.pt`, and `phase3_queue.py` skips the 28 completed runs):

```text
python -X utf8 scripts/phase3_train.py --model densenet121 --protocol clean --seed 45 --epochs 30 --num-workers 4 --out results/phase3/runs/densenet121_clean_s45
python -X utf8 scripts/phase3_train.py --model densenet121 --protocol clean --seed 43 --epochs 30 --num-workers 4 --out results/phase3/runs/densenet121_clean_s43
python -X utf8 scripts/phase3_train.py --model densenet121 --protocol clean --seed 44 --epochs 30 --num-workers 4 --out results/phase3/runs/densenet121_clean_s44
python -X utf8 scripts/phase3_train.py --model shared_unet --protocol brisc --seed 43 --epochs 30 --num-workers 4 --seg-weight 0.25 --seg-classes 2 --out results/phase3/runs/shared_unet_brisc_s43+ls0.25+sc2
python -X utf8 scripts/phase3_train.py --model shared_unet --protocol brisc --seed 45 --epochs 30 --num-workers 4 --seg-weight 0.25 --seg-classes 2 --out results/phase3/runs/shared_unet_brisc_s45+ls0.25+sc2
python -X utf8 scripts/phase3_train.py --model shared_unet --protocol brisc --seed 44 --epochs 30 --num-workers 4 --seg-weight 0.25 --seg-classes 2 --out results/phase3/runs/shared_unet_brisc_s44+ls0.25+sc2
python -X utf8 scripts/phase3_train.py --model efficientnet_v2_s --protocol clean --seed 43 --epochs 30 --num-workers 4 --out results/phase3/runs/efficientnet_v2_s_clean_s43
python -X utf8 scripts/phase3_train.py --model efficientnet_v2_s --protocol clean --seed 44 --epochs 30 --num-workers 4 --out results/phase3/runs/efficientnet_v2_s_clean_s44
python -X utf8 scripts/phase3_train.py --model swin_t --protocol clean --seed 43 --epochs 30 --num-workers 4 --out results/phase3/runs/swin_t_clean_s43
python -X utf8 scripts/phase3_train.py --model swin_t --protocol clean --seed 44 --epochs 30 --num-workers 4 --out results/phase3/runs/swin_t_clean_s44
python -X utf8 scripts/phase3_train.py --model tamnet --protocol clean --seed 43 --epochs 30 --num-workers 1 --no-attention-fusion --out results/phase3/runs/tamnet_clean_s43+concat
python -X utf8 scripts/phase3_train.py --model tamnet --protocol clean --seed 44 --epochs 30 --num-workers 1 --no-attention-fusion --out results/phase3/runs/tamnet_clean_s44+concat
python -X utf8 scripts/phase3_train.py --model tamnet --protocol clean --seed 45 --epochs 30 --num-workers 1 --no-attention-fusion --out results/phase3/runs/tamnet_clean_s45+concat
```

Queue form (preferred — one command, skips completed, auto-resumes in-flight):

```text
python -X utf8 scripts/phase3_queue.py --queue results/phase3/queue_stage4_fast.json --status-file results/phase3/queue_status_4.json
```

## Phase 3 — Reboot + relaunch day (2026-11-25)

| Item | Status | Notes |
|---|---|---|
| Pagefile restored | DONE (`758cd6d`) | 29 GB fixed pagefile on D: (`PagingFiles = D:\pagefile.sys 29696 32768`, registry-persisted) → commit ceiling 44.26 GB (was 39 GB); runtime attach impossible (`NtCreatePagefile` not exported) |
| Auto-start after reboot | DONE | scheduled task `TAMNet_ResumeNext3` (AtLogOn, non-elevated) → `scripts/resume_next3_after_reboot.py` with fail-safes (ceiling < 40 GB → start nothing; phase3 already running → no duplicates) |
| 12:51 task-launch crash (3 trainers) | **FIXED** (`32181bc`) | `forrtl: error (200)` window-CLOSE = Task Scheduler console teardown, **not memory**; evidence kept as `results/phase3/logs/*.bootfail1252`; checkpoints verified intact, nothing re-run. Fix: `phase3_queue.py` spawns trainers with `DETACHED_PROCESS \| CREATE_NEW_PROCESS_GROUP` (no console to close) |
| Triple simultaneous cold-start | **FIXED** (`32181bc`) | s43 died `RuntimeError: bad allocation` (stale `run_config.json` moved aside as `.badalloc1329`, never deleted); fix: resume script staggers queue launches 120 s + waits for ≥16 GB commit headroom (`HEADROOM_TIMEOUT_S=900`) |
| Win32 **1455** killed s44 @ ep7 (14:11) | ROOT-CAUSED | **Sekiro was running** (~9.75 GB commit + ~2.4 GB VRAM of the 4 GB): commit peaked **46.6/47.0** (pagefile auto-grew to its 32 GB max), torch shared-memory mapping failed. s44 checkpoint saved at ep6 → resumed ep7, **zero epoch loss**. Games-closed rule reaffirmed |
| Monitor saw "0 trainers" | **FIXED** (`32181bc`) | `phase3_monitor.py` now derives run name from `--model/--protocol/--seed` + suffix rules (queue never passes `--out`) |
| Concurrency policy (empirical) | 2 normal / 3 max | pre-reboot: 3 stacks peaked 37.2/39.0 ✓; 2 stacks saturate GPU (99 %); 3rd slot only with ≥14 GB commit headroom, else `--num-workers 0` |
| **`densenet121_clean_s45` DONE** | **DONE 14:32** | best val F1(w) **0.9452 @ ep3**, early stop at 10/30 epochs, 52.6 min, provenance `training_result.json` (git `32181bc`) |
| **`densenet121_clean_s44` DONE** | **DONE 15:12** | best val F1(w) **0.9471 @ ep6**, early stop at 13/30 epochs, 42 min, resumed from ep7 after the 1455 kill |
| **`densenet121_clean_s43`** | RUNNING | relaunched 15:25 after the 15:21 worker death → **auto-resumed from checkpoint ep8** (best 0.9488 @ ep8), `queue_status_next3_b.json` |
| 15:21 **DataLoader worker deaths** killed 2 runs | **FIXED** | 3 stacks at **43.75/44.26 (0.51 GB free)** → workers exited unexpectedly (`RuntimeError: DataLoader worker ... exited unexpectedly`) in `densenet121_clean_s43` (checkpoint saved @ ep8 → resumed, zero epoch loss) and `shared_unet_brisc_s45` (only a `run_config.json` → moved aside as `.workerdied1521`, fresh start). Measured per-stack commit: **nw4 ≈ 12.4 GB, nw0 ≈ 10 GB** → policy: **3rd concurrent slot always `--num-workers 0`**; steady state after fix **36.6/44.3 with 3 alive** (7.7 GB free) |
| 3rd slot (revised) | RUNNING | `queue_stage4_d.json`: s45 shared_unet relaunched at **nw0** → `tamnet_clean_s45+concat` next (fills the `+concat` n=4 gap); `queue_stage4_c3.json` = s43 shared_unet (nw0) |

## Phase 4 — Figures _(pending)_
## Phase 5 — Paper rewrite _(pending)_
## Phase 6 — Reproducibility & QA _(pending)_
