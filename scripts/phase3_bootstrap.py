"""Phase 3D — bootstrap 95 % CIs on every TEST accuracy + paired model comparisons.

The paper reports point accuracies with no interval and compares models on
differences of ~0.1–1 pp.  With n = 1,672 test images the binomial standard
error of a 0.95 accuracy is ~0.0053, so several of the paper's rankings are
inside the noise band.  This script quantifies that.

Sources (nothing is recomputed from scratch): each run's
``eval_internal_test/predictions.csv`` written by ``scripts/phase3_evaluate.py``.

Outputs
-------
``results/phase3/runs/<run>/eval_internal_test/bootstrap_ci.json``
    per-run point metrics + 95 % percentile-bootstrap CI (B = 1000) for
    accuracy, macro-F1, weighted-F1 and MCC, plus per-class ROC-AUC / PR-AUC.

``results/phase3/bootstrap_ci.json``
    grouped summary (mean over seeds of the CI bounds) and the **paired**
    bootstrap for each (TAMNet, baseline) pair evaluated on the same test set:

        delta = acc(TAMNet seed s) - acc(baseline seed s)

    resampling test indices with replacement, same indices for both models
    (paired).  A CI that contains 0 means the two models are not
    distinguishable on this test set, whatever the point estimate says.

Usage:
    python -X utf8 scripts/phase3_bootstrap.py
    python -X utf8 scripts/phase3_bootstrap.py --splits 1000 --runs tamnet_clean_s42
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    auc,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results" / "phase3" / "runs"
OUT = ROOT / "results" / "phase3" / "bootstrap_ci.json"
CLASS_NAMES = ("Glioma", "Meningioma", "Pituitary", "No Tumor")

# pairs compared to the headline configuration (same seed when both exist)
PAIR_BASES = (
    "densenet121_clean_s42",
    "resnet50_clean_s42",
    "efficientnet_v2_s_clean_s42",
    "convnext_tiny_clean_s42",
    "swin_t_clean_s42",
    "vit_b_16_clean_s42",
    "shared_unet_brisc_s42+ls0.25+sc2",
)


def read_predictions(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ys, ps, probs = [], [], []
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        prob_cols = [c for c in reader.fieldnames if c.startswith("p_")]
        for row in reader:
            ys.append(int(row["y_true"]))
            ps.append(int(row["y_pred"]))
            probs.append([float(row[c]) for c in prob_cols])
    return (
        np.asarray(ys, dtype=np.int64),
        np.asarray(ps, dtype=np.int64),
        np.asarray(probs, dtype=np.float64),
    )


def point_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, p, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y, p)),
    }


def bootstrap_ci(y: np.ndarray, p: np.ndarray, splits: int, seed: int, alpha: float = 0.05) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y)
    accs, mf1s, wf1s, mccs = [], [], [], []
    for _ in range(splits):
        idx = rng.integers(0, n, n)
        yy, pp = y[idx], p[idx]
        accs.append(accuracy_score(yy, pp))
        mf1s.append(f1_score(yy, pp, average="macro", zero_division=0))
        wf1s.append(f1_score(yy, pp, average="weighted", zero_division=0))
        mccs.append(matthews_corrcoef(yy, pp))
    lo, hi = alpha / 2 * 100, (1 - alpha / 2) * 100
    out = {}
    for name, vals in (("accuracy", accs), ("macro_f1", mf1s), ("weighted_f1", wf1s), ("mcc", mccs)):
        point = point_metrics(y, p)[name]
        out[name] = {
            "point": point,
            "ci95_low": float(np.percentile(vals, lo)),
            "ci95_high": float(np.percentile(vals, hi)),
            "boot_mean": float(np.mean(vals)),
            "boot_std": float(np.std(vals, ddof=1)),
        }
    return out


def paired_delta(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, splits: int, seed: int) -> dict:
    """Bootstrap the accuracy difference on identical resampled indices."""
    rng = np.random.default_rng(seed)
    n = len(y)
    correct_a = (p_a == y).astype(np.float64)
    correct_b = (p_b == y).astype(np.float64)
    deltas = np.empty(splits)
    for i in range(splits):
        idx = rng.integers(0, n, n)
        deltas[i] = correct_a[idx].mean() - correct_b[idx].mean()
    delta = float(correct_a.mean() - correct_b.mean())
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "delta": delta,
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "p_two_sided": float(2 * min((deltas <= 0).mean(), (deltas >= 0).mean())),
        "significant": bool(lo > 0 or hi < 0),
    }


def per_class_auc(y: np.ndarray, probs: np.ndarray) -> dict:
    out = {}
    for ci, name in enumerate(CLASS_NAMES):
        binary = (y == ci).astype(int)
        if binary.min() == binary.max():
            continue
        pr, rc, _ = precision_recall_curve(binary, probs[:, ci])
        out[name] = {
            "support": int(binary.sum()),
            "roc_auc": float(roc_auc_score(binary, probs[:, ci])),
            "pr_auc": float(auc(rc, pr)),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--no-pairs", action="store_true")
    args = ap.parse_args()

    run_dirs = [RUNS / r for r in args.runs] if args.runs else [
        d for d in sorted(RUNS.iterdir())
        if (d / "eval_internal_test" / "predictions.csv").is_file()
    ]
    print(f"bootstrap: {len(run_dirs)} runs, B={args.splits}")

    per_run = {}
    skipped = []
    for d in run_dirs:
        pred_path = d / "eval_internal_test" / "predictions.csv"
        if not pred_path.is_file():
            skipped.append(d.name)
            continue
        y, p, probs = read_predictions(pred_path)
        payload = {
            "kind": "phase3_bootstrap_ci",
            "run": d.name,
            "n": int(len(y)),
            "splits": args.splits,
            "method": "percentile bootstrap over test images (i.i.d. resampling)",
            "caveat": "resamples test images, not patients; splits are image-level (see Phase 1)",
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **bootstrap_ci(y, p, args.splits, args.seed),
            "per_class": per_class_auc(y, probs),
            "source": pred_path.relative_to(ROOT).as_posix(),
        }
        (pred_path.parent / "bootstrap_ci.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        per_run[d.name] = {"y": y, "pred": p, "metrics": payload}
        m = payload["accuracy"]
        print(f"  {d.name:<34} acc {m['point']:.4f} "
              f"[{m['ci95_low']:.4f}, {m['ci95_high']:.4f}]")

    # ---- paired comparisons against the headline seeds -------------------
    pairs = []
    if not args.no_pairs:
        headline = [n for n in per_run if n.startswith("tamnet_clean_s")
                    and "+" not in n and "-" not in n and not n.endswith("budget")]
        print("\npaired bootstrap (same test indices, headline TAMNet seeds):")
        for base in PAIR_BASES:
            if base not in per_run:
                continue
            seed = base.rsplit("_s", 1)[-1][:2]
            cand = f"tamnet_clean_s{seed}"
            if cand not in per_run:
                continue
            a, b = per_run[cand], per_run[base]
            if not np.array_equal(a["y"], b["y"]):
                print(f"  SKIP {cand} vs {base}: different test sets")
                continue
            res = paired_delta(a["y"], a["pred"], b["pred"], args.splits, args.seed)
            res.update({"tamnet": cand, "baseline": base})
            pairs.append(res)
            flag = "significant" if res["significant"] else "NOT distinguishable from 0"
            print(f"  {cand:<20} - {base:<34} "
                  f"delta {res['delta']:+.4f} [{res['ci95_low']:+.4f}, {res['ci95_high']:+.4f}]  {flag}")

    payload = {
        "kind": "phase3_bootstrap_summary",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "splits": args.splits,
        "method": "percentile bootstrap over test images; paired for model comparisons",
        "runs": {name: {"metrics": v["metrics"], "source": v["metrics"]["source"]}
                 for name, v in per_run.items()},
        "paired_tamnet_vs_baseline": pairs,
        "runs_skipped_no_predictions": skipped,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
