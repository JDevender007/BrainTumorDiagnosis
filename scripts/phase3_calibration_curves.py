"""Phase 3F (part 2) — reliability diagrams and risk–coverage curves.

``scripts/phase3_uncertainty.py`` stores scalar summaries (ECE, MCE, Brier,
NLL, entropy, MI, AUROC, selective accuracy at 5 coverages).  A reliability
*diagram* and a full risk–coverage curve need the binned/per-sample data, which
this script writes:

    results/phase3/reliability.json
      curves[run]  = {ece, bins: [{mean_confidence, accuracy, n}], n}
      selective[run] = {coverage: [...], accuracy: [...], aurc}

Protocol is identical to the uncertainty script (T MC passes, BatchNorm frozen,
only Dropout active — Phase 2 / D5 fix), so the two artifacts are consistent.

    python -X utf8 scripts/phase3_calibration_curves.py
    python -X utf8 scripts/phase3_calibration_curves.py --runs tamnet_clean_s42 densenet121_clean_s42
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.phase3_evaluate import build_model, sha256_file  # noqa: E402
from scripts.phase3_uncertainty import (  # noqa: E402
    DEFAULT_SPLIT,
    enable_dropout_only,
    entropy,
    expected_calibration_error,
    load_test_tensor,
)

RUNS = ROOT / "results" / "phase3" / "runs"
OUT = ROOT / "results" / "phase3" / "reliability.json"

DEFAULT_RUNS = ("tamnet_clean_s42", "densenet121_clean_s42", "efficientnet_v2_s_clean_s42")
BINS = 15


@torch.no_grad()
def mc_mean_probs(model, x: torch.Tensor, passes: int, batch: int) -> np.ndarray:
    """MC mean predictive probabilities; batches move to the model's device.

    ``x`` is host-resident (see phase3_uncertainty.load_test_tensor) because a
    full 1 GiB TEST tensor does not coexist with activations on a 4 GiB card.
    """
    device = next(model.parameters()).device
    model.eval()
    acc = None
    enable_dropout_only(model)
    for _ in range(passes):
        chunks = []
        for i in range(0, len(x), batch):
            out = model(x[i:i + batch].to(device))
            p = out["probabilities"] if isinstance(out, dict) else torch.softmax(out, 1)
            chunks.append(p.float().cpu())
        probs = torch.cat(chunks).numpy()
        acc = probs if acc is None else acc + probs
    model.eval()
    return acc / passes


def reliability(y: np.ndarray, probs: np.ndarray, bins: int) -> dict:
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y).astype(np.float64)
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf < hi) if hi < 1.0 else (conf >= lo) & (conf <= hi)
        rows.append({
            "lo": round(float(lo), 4), "hi": round(float(hi), 4),
            "n": int(mask.sum()),
            "mean_confidence": float(conf[mask].mean()) if mask.any() else None,
            "accuracy": float(correct[mask].mean()) if mask.any() else None,
        })
    ece, mce = expected_calibration_error(conf, correct, bins)
    return {"n": int(len(y)), "ece": ece, "mce": mce, "bins": rows}


def risk_coverage(y: np.ndarray, probs: np.ndarray, n_points: int = 20) -> dict:
    """Sort by confidence (desc); accuracy of the retained set vs coverage."""
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y).astype(np.float64)
    order = np.argsort(-conf)
    sorted_correct = correct[order]
    coverages = np.linspace(1.0, 0.1, n_points)
    accs = []
    for c in coverages:
        k = max(1, int(round(c * len(y))))
        accs.append(float(sorted_correct[:k].mean()))
    # AURC over the full curve (lower is better)
    full = np.concatenate([[0.0], np.cumsum(sorted_correct) / np.arange(1, len(y) + 1)])
    aurc = float(np.trapezoid(full, dx=1.0 / len(y)))
    return {
        "coverage": [round(float(c), 3) for c in coverages],
        "accuracy": accs,
        "aurc": aurc,
        "note": "coverage 1.0 = keep everything; 0.1 = keep the 10% most confident",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=list(DEFAULT_RUNS))
    ap.add_argument("--passes", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--bins", type=int, default=BINS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    payload = {"curves": {}, "selective": {}}
    if OUT.is_file() and not args.force:
        try:
            payload = json.loads(OUT.read_text(encoding="utf-8"))
        except ValueError:
            pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading TEST pixels on {device} ...", flush=True)
    x, y = load_test_tensor(device)

    for name in args.runs:
        d = RUNS / name
        if not (d / "best.pt").is_file():
            print(f"MISSING best.pt: {name}")
            continue
        cfg = json.loads((d / "run_config.json").read_text(encoding="utf-8"))
        model = build_model(cfg)
        model.load_state_dict(torch.load(d / "best.pt", map_location="cpu", weights_only=False)["model_state_dict"])
        model = model.to(device).eval()
        probs = mc_mean_probs(model, x, args.passes, args.batch_size)
        rel = reliability(y, probs, args.bins)
        rc = risk_coverage(y, probs)
        payload["curves"][name] = rel
        payload["selective"][name] = rc
        print(f"  {name:<30} ECE {rel['ece']:.4f}  MCE {rel['mce']:.4f}  AURC {rc['aurc']:.4f}",
              flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    payload.update({
        "kind": "phase3_reliability_curves",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": "MC-Dropout T=%d, BatchNorm frozen, only Dropout active (D5 fix)" % args.passes,
        "test_manifest": str(DEFAULT_SPLIT.relative_to(ROOT)),
        "bins": args.bins,
        "checkpoints": {
            n: sha256_file(RUNS / n / "best.pt") for n in args.runs
            if (RUNS / n / "best.pt").is_file()
        },
    })
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f" wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
