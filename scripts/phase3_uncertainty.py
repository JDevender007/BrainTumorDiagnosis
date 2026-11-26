"""Phase 3F — MC-Dropout uncertainty, calibration (ECE/MCE/Brier/NLL), selective prediction.

Why this exists
---------------
Every uncertainty number in the legacy paper (ECE = 0.0855, entropy 0.52, MI,
"accuracy vs ECE" bar) came from the **buggy** MC-dropout path that called
``model.train()``: BatchNorm was also put in train mode, so the T stochastic
passes mixed batch-statistics noise into the predictive distribution and
mutated the running statistics.  After the Phase 2 / D5 fix the correct
protocol is:

    model.eval()                      # BatchNorm frozen, running stats used
    only Dropout modules .train()     # stochasticity comes from dropout alone

This script implements that protocol for **any** run in ``results/phase3/runs``
(TAMNet, shared-UNet and every timm backbone), on the internal TEST split, and
writes one artifact per run:

    results/phase3/runs/<run>/eval_internal_test/uncertainty.json

plus a grouped summary (mean ± sample SD over seeds, ddof=1):

    results/phase3/uncertainty_summary.json

Reported per run
----------------
* ``accuracy`` — argmax of the mean predictive distribution (MC average)
* ``accuracy_deterministic`` — single pass, dropout off (sanity: MC must not
  change accuracy materially)
* ``nll``, ``brier``, ``ece_15``, ``mce_15`` — calibration of the MC mean
* ``mean_confidence``, ``mean_predictive_entropy``, ``mean_mutual_information``,
  ``mean_aleatoric_variance`` — the uncertainty decomposition
  (predictive entropy = aleatoric + epistemic, MI = epistemic)
* ``selective_accuracy`` at coverage 1.0/0.9/0.8/0.7/0.5, ranked by confidence
* ``error_detection_auroc`` — AUROC of MI for separating errors from correct
  predictions (0.5 = uncertainty carries no error signal)

Timing: the TEST images are decoded and normalised **once** and kept on the
GPU, so the T passes are pure forward passes.  Expect ~2–3 min per run.

Usage:
    python -X utf8 scripts/phase3_uncertainty.py --dry-run
    python -X utf8 scripts/phase3_uncertainty.py --passes 10
    python -X utf8 scripts/phase3_uncertainty.py --runs tamnet_clean_s42
    python -X utf8 scripts/phase3_uncertainty.py --all
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datasets.classification import (  # noqa: E402
    CLASS_NAMES,
    BrainTumorClassificationDataset,
)
from datasets.normalization import build_classification_transform  # noqa: E402
from scripts.phase3_aggregate_all import GROUPS  # noqa: E402
from scripts.phase3_evaluate import build_model, sha256_file  # noqa: E402

RUNS = ROOT / "results" / "phase3" / "runs"
SUMMARY = ROOT / "results" / "phase3" / "uncertainty_summary.json"
DEFAULT_SPLIT = ROOT / "results" / "phase3" / "split_manifest_clean.json"

# calibration / uncertainty in the main comparison table
DEFAULT_GROUPS = (
    "tamnet_clean",
    "tamnet_clean_mc",
    "densenet121_clean",
    "efficientnet_v2_s_clean",
    "swin_t_clean",
    "resnet50_clean",
    "convnext_tiny_clean",
    "vit_b_16_clean",
    "shared_unet_brisc",
)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> tuple[float, float]:
    """Equal-width ECE and MCE over the confidence axis."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    mce = 0.0
    n = len(conf)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf < hi) if hi < 1.0 else (conf >= lo) & (conf <= hi)
        if not mask.any():
            continue
        gap = abs(correct[mask].mean() - conf[mask].mean())
        ece += mask.sum() / n * gap
        mce = max(mce, gap)
    return float(ece), float(mce)


def brier_multi(probs: np.ndarray, y: np.ndarray) -> float:
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def nll(probs: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, None))))


def selective_accuracy(conf: np.ndarray, correct: np.ndarray, coverage: float) -> float:
    k = max(1, int(round(coverage * len(conf))))
    idx = np.argsort(-conf)[:k]
    return float(correct[idx].mean())


def entropy(p: np.ndarray) -> np.ndarray:
    return -np.sum(p * np.log(np.clip(p, 1e-12, None)), axis=-1)


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #
def enable_dropout_only(model: torch.nn.Module) -> None:
    """Phase 2 / D5 protocol: eval everywhere, Dropout layers stochastic."""
    model.eval()
    for module in model.modules():
        if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout1d, torch.nn.Dropout2d, torch.nn.Dropout3d)):
            module.train()


def load_test_tensor(device: torch.device) -> tuple[torch.Tensor, np.ndarray]:
    manifest = json.loads(DEFAULT_SPLIT.read_text(encoding="utf-8"))
    entries = [e for e in manifest["entries"] if e["split"] == "test"]

    class _S:
        __slots__ = ("image_path", "class_index", "class_name", "split", "mask_path")

        def __init__(self, path: Path, cls: str):
            self.image_path = path
            self.class_name = cls
            self.class_index = CLASS_NAMES.index(cls)
            self.split = "test"
            self.mask_path = None

    samples = [_S(ROOT / e["path"], e["class"]) for e in entries]
    ds = BrainTumorClassificationDataset(
        samples, transform=build_classification_transform((224, 224))
    )
    xs, ys = [], []
    for i in range(len(ds)):
        item = ds[i]
        xs.append(item[0])
        ys.append(int(item[1]))
    return torch.stack(xs).to(device), np.asarray(ys, dtype=np.int64)


@torch.no_grad()
def mc_predict(model, x: torch.Tensor, passes: int, batch: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mean_probs (N,C), per-pass probs (T,N,C), deterministic probs (N,C))."""
    model.eval()
    logits = []
    for i in range(0, len(x), batch):
        out = model(x[i:i + batch])
        logits.append((out["probabilities"] if isinstance(out, dict) else torch.softmax(out, 1)).float())
    det = torch.cat(logits).cpu().numpy()

    enable_dropout_only(model)
    runs = []
    for _ in range(passes):
        logits = []
        for i in range(0, len(x), batch):
            out = model(x[i:i + batch])
            logits.append((out["probabilities"] if isinstance(out, dict) else torch.softmax(out, 1)).float())
        runs.append(torch.cat(logits).cpu().numpy())
    model.eval()
    stacked = np.stack(runs, axis=0)          # (T, N, C)
    return stacked.mean(axis=0), stacked, det


def evaluate_run(run_dir: Path, args, x: torch.Tensor, y: np.ndarray) -> dict:
    run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    ckpt_path = run_dir / "best.pt"
    model = build_model(run_config)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(x.device).eval()

    mean_p, stacked, det = mc_predict(model, x, args.passes, args.batch_size)

    pred = mean_p.argmax(axis=1)
    correct = (pred == y).astype(np.float64)
    conf = mean_p.max(axis=1)
    pe = entropy(mean_p)
    mi = pe - stacked_entropy_mean(stacked)
    ale = float(np.mean(stacked.var(axis=0).sum(axis=1)))

    ece, mce = expected_calibration_error(conf, correct, args.bins)
    try:
        auroc = float(roc_auc_score(1.0 - correct, mi))
    except ValueError:
        auroc = None

    per_class = {}
    for ci, name in enumerate(CLASS_NAMES):
        mask = y == ci
        if mask.any():
            per_class[name] = {
                "support": int(mask.sum()),
                "mean_entropy": float(pe[mask].mean()),
                "mean_confidence": float(conf[mask].mean()),
                "accuracy": float(correct[mask].mean()),
            }

    payload = {
        "kind": "phase3_uncertainty",
        "run": run_dir.name,
        "n": int(len(y)),
        "passes": args.passes,
        "protocol": "model.eval() with only Dropout layers in train mode (Phase 2 / D5 fix)",
        "accuracy": float(correct.mean()),
        "accuracy_deterministic": float((det.argmax(axis=1) == y).mean()),
        "nll": nll(mean_p, y),
        "brier": brier_multi(mean_p, y),
        "ece_15": ece,
        "mce_15": mce,
        "mean_confidence": float(conf.mean()),
        "mean_predictive_entropy": float(pe.mean()),
        "mean_mutual_information": float(mi.mean()),
        "mean_aleatoric_variance": ale,
        "selective_accuracy": {
            str(c): selective_accuracy(conf, correct, c) for c in (1.0, 0.9, 0.8, 0.7, 0.5)
        },
        "error_detection_auroc": auroc,
        "per_class": per_class,
        "provenance": {
            "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "run_name": run_config.get("run_name"),
            "run_git": run_config.get("git"),
            "seed": run_config.get("seed"),
            "checkpoint": str(ckpt_path.relative_to(ROOT)),
            "checkpoint_sha256": sha256_file(ckpt_path),
            "run_config_sha256": sha256_file(run_dir / "run_config.json"),
            "test_manifest": str(DEFAULT_SPLIT.relative_to(ROOT)),
            "test_manifest_sha256": sha256_file(DEFAULT_SPLIT),
            "script": "scripts/phase3_uncertainty.py",
        },
    }
    out_dir = run_dir / "eval_internal_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "uncertainty.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def stacked_entropy_mean(stacked: np.ndarray) -> np.ndarray:
    return np.mean(entropy(stacked), axis=0)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def resolve_runs(args) -> list[Path]:
    if args.runs:
        return [RUNS / r for r in args.runs]
    if args.all:
        return [d for d in sorted(RUNS.iterdir())
                if (d / "best.pt").is_file() and (d / "training_result.json").is_file()]
    wanted = [n for slug, _, names in GROUPS if slug in args.groups for n in names]
    return [RUNS / n for n in wanted if (RUNS / n / "best.pt").is_file()]


def aggregate(run_dirs: list[Path], payloads: list[dict], args) -> dict:
    by_name = {p["run"]: p for p in payloads}
    groups_out = []
    for slug, description, names in GROUPS:
        present = [by_name[n] for n in names if n in by_name]
        if not present:
            continue
        field_summary = {}
        for key in ("accuracy", "ece_15", "mce_15", "nll", "brier",
                    "mean_confidence", "mean_predictive_entropy",
                    "mean_mutual_information", "mean_aleatoric_variance",
                    "error_detection_auroc"):
            vals = [p[key] for p in present if p.get(key) is not None]
            if not vals:
                continue
            field_summary[key] = {
                "mean": statistics.fmean(vals),
                "std_sample_ddof1": statistics.stdev(vals) if len(vals) > 1 else None,
                "min": min(vals),
                "max": max(vals),
            }
        groups_out.append({
            "slug": slug,
            "description": description,
            "n_seeds": len(present),
            "runs": [p["run"] for p in present],
            "summary": field_summary,
            "per_seed_accuracy": {p["run"]: p["accuracy"] for p in present},
            "per_seed_ece": {p["run"]: p["ece_15"] for p in present},
        })
    return {
        "kind": "phase3_uncertainty_summary",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": "MC-Dropout T passes, BN in eval mode (D5 fix); TEST split used once per config",
        "passes": args.passes,
        "headline_rule": "mean over seeds (never best seed); std = sample ddof=1",
        "groups": groups_out,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--groups", nargs="*", default=list(DEFAULT_GROUPS))
    ap.add_argument("--all", action="store_true", help="every completed run")
    ap.add_argument("--passes", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--bins", type=int, default=15)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    run_dirs = resolve_runs(args)
    missing = [d for d in run_dirs if not (d / "best.pt").is_file()]
    if missing:
        for d in missing:
            print(f"MISSING best.pt: {d.name}")
        return 2
    jobs = [d for d in run_dirs
            if args.force or not (d / "eval_internal_test" / "uncertainty.json").is_file()]

    print(f"uncertainty: {len(run_dirs)} runs selected, {len(jobs)} to compute "
          f"(T={args.passes}, batch={args.batch_size})")
    for d in jobs:
        print(f"  {d.name}")
    if args.dry_run:
        print("\n(dry run - nothing executed)")
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} | preloading TEST images ...", flush=True)
    t0 = time.time()
    x, y = load_test_tensor(device)
    print(f"  {tuple(x.shape)} on {device} ({time.time() - t0:.0f}s)", flush=True)

    payloads: list[dict] = []
    failed = []
    for i, d in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}] {d.name} ...", flush=True)
        t = time.time()
        try:
            p = evaluate_run(d, args, x, y)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
            failed.append(d.name)
            continue
        del p["per_class"]  # keep the console record small; it is in the artifact
        payloads.append(p)
        print(f"  acc {p['accuracy']:.4f} (det {p['accuracy_deterministic']:.4f}) "
              f"ECE {p['ece_15']:.4f} NLL {p['nll']:.4f} "
              f"MI {p['mean_mutual_information']:.4f} "
              f"AUROC {p['error_detection_auroc']:.3f} ({time.time() - t:.0f}s)", flush=True)

    # merge every on-disk artifact so partial batches still produce a full summary
    all_payloads = []
    for d in run_dirs + [p for p in RUNS.iterdir() if p.is_dir()]:
        f = d / "eval_internal_test" / "uncertainty.json"
        if f.is_file():
            try:
                all_payloads.append(json.loads(f.read_text(encoding="utf-8")))
            except ValueError:
                pass
    dedup = {p["run"]: p for p in all_payloads}

    summary = aggregate(run_dirs, list(dedup.values()), args)
    summary["failed_this_batch"] = failed
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 100)
    print(f" Phase 3F uncertainty / calibration (MC-Dropout T={args.passes}, TEST n={len(y)})")
    print("=" * 100)
    print(f" {'group':<24} {'n':>2} {'accuracy':<8} {'ECE':<14} {'NLL':<14} {'MI':<14} {'AUROC':<6}")
    print("-" * 100)
    for g in sorted(summary["groups"], key=lambda g: -g["summary"]["accuracy"]["mean"]):
        s = g["summary"]

        def cell(k, fmt="{:.4f}"):
            d = s.get(k)
            if not d:
                return "-"
            if d["std_sample_ddof1"] is None:
                return fmt.format(d["mean"])
            return (fmt.format(d["mean"]) + "±" + fmt.format(d["std_sample_ddof1"]))

        print(f" {g['slug'][:24]:<24} {g['n_seeds']:>2} {cell('accuracy'):<8} "
              f"{cell('ece_15'):<14} {cell('nll'):<14} {cell('mean_mutual_information'):<14} "
              f"{cell('error_detection_auroc', '{:.3f}'):<6}")
    print("-" * 100)
    print(f" summary -> {SUMMARY.relative_to(ROOT)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
