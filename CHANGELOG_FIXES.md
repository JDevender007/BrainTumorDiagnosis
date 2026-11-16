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

## Phase 2 — Code vs. paper consistency _(pending)_
## Phase 3 — Experiments, 5 seeds _(pending; needs GPU-budget approval)_
## Phase 4 — Figures _(pending)_
## Phase 5 — Paper rewrite _(pending)_
## Phase 6 — Reproducibility & QA _(pending)_
