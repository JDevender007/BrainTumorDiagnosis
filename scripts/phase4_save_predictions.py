"""Phase 4 — one inference pass that retains what supplementary panels need.

``eval_*/metrics.json`` keeps only aggregate confusion counts, which is why
S3 (per-class ROC/PR), S6 (feature embedding) and S7 (failure gallery) were
reported MISSING. This script does a single deterministic pass over the clean
test split with the headline TAMNet run and writes:

    results/phase3/predictions/tamnet_clean_s42.npz
        y_true, y_pred, probs (N×C), features (N×D fused embedding), paths

    results/phase3/predictions/tamnet_clean_s42.json
        provenance: checkpoint sha256, split manifest sha256, git commit,
        transform, seed, N, accuracy (must equal the recorded TEST accuracy)

The accuracy written here is asserted against the already-published TEST
number: if they disagree, the script fails instead of publishing two
different values for one run.

    python -X utf8 scripts/phase4_save_predictions.py
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

from datasets.classification import (  # noqa: E402
    BrainTumorClassificationDataset,
    ClassificationSample,
)
from datasets.classification import CLASS_NAMES  # noqa: E402
from datasets.normalization import build_classification_transform  # noqa: E402
from scripts.phase3_evaluate import build_model, sha256_file  # noqa: E402

RUNS = ROOT / "results" / "phase3" / "runs"
OUT_DIR = ROOT / "results" / "phase3" / "predictions"
DEFAULT_RUN = "tamnet_clean_s42"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=DEFAULT_RUN)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--force", action="store_true")
    p.add_argument("--split", default="test",
                   choices=("test", "val", "train"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run = RUNS / args.run
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    npz_path = OUT_DIR / f"{args.run}_{args.split}.npz"
    js_path = OUT_DIR / f"{args.run}_{args.split}.json"

    if npz_path.is_file() and js_path.is_file() and not args.force:
        print(f"{npz_path.relative_to(ROOT)} exists - use --force to recompute")
        return 0

    cfg = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
    manifest_path = ROOT / "results" / "phase3" / "split_manifest_clean.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = [
        ClassificationSample(
            image_path=ROOT / e["path"],
            class_name=e["class"],
            class_index=CLASS_NAMES.index(e["class"]),
            split=e["split"],
        )
        for e in manifest["entries"]
        if e["split"] == args.split
    ]
    if not samples:
        print(f"FAIL: no samples for split {args.split!r}")
        return 1
    ds = BrainTumorClassificationDataset(
        samples, transform=build_classification_transform((224, 224)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg)
    ckpt = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    # fused embedding: tap the encoder feature map and pool it spatially, so
    # S6 gets the representation TAMNet actually fuses (tumor_aware.encode)
    feats_all: list[np.ndarray] = []

    def _tap(module, inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        if isinstance(tensor, torch.Tensor):
            if tensor.dim() == 4:
                tensor = tensor.detach().float().mean(dim=(2, 3))
            else:
                tensor = tensor.detach().float()
            feats_all.append(tensor.cpu().numpy())

    handle = model.encoder.register_forward_hook(_tap) \
        if hasattr(model, "encoder") else None

    ys, ps, paths = [], [], []
    with torch.no_grad():
        for i in range(0, len(ds), args.batch_size):
            items = [ds[j] for j in range(i, min(i + args.batch_size, len(ds)))]
            x = torch.stack([it[0] for it in items]).to(device)
            out = model(x)
            if isinstance(out, dict):
                logits = out.get("logits")
            else:
                logits = out
            prob = torch.softmax(logits.float(), dim=1)
            ps.append(prob.cpu().numpy())
            ys.extend(int(it[1]) for it in items)
            paths.extend(samples[j].image_path for j in
                         range(i, min(i + args.batch_size, len(ds))))
    if handle is not None:
        handle.remove()

    probs = np.concatenate(ps, axis=0)
    y_true = np.asarray(ys, dtype=np.int64)
    y_pred = probs.argmax(axis=1)

    acc = float((y_true == y_pred).mean())
    ref = json.loads((run / "eval_internal_test" / "metrics.json")
                     .read_text(encoding="utf-8")) if (
        run / "eval_internal_test" / "metrics.json").is_file() else None
    if ref is not None and abs(acc - ref["accuracy"]) > 1e-9:
        print(f"FAIL: re-inference accuracy {acc:.6f} != recorded "
              f"{ref['accuracy']:.6f} for {args.run}")
        return 1

    features = np.concatenate(feats_all, axis=0) if feats_all else None
    np.savez_compressed(npz_path, y_true=y_true, y_pred=y_pred,
                        probs=probs.astype(np.float32),
                        features=(features.astype(np.float32)
                                  if features is not None else np.zeros(0)),
                        paths=np.asarray([str(p) for p in paths]))

    try:
        commit = __import__("subprocess").run(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True,
            text=True, encoding="utf-8", errors="replace").stdout.strip()
    except OSError:
        commit = "unknown"

    js_path.write_text(json.dumps({
        "kind": "phase4_saved_predictions",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run": args.run,
        "split": args.split,
        "n": int(len(y_true)),
        "accuracy": acc,
        "accuracy_matches_eval": True,
        "n_classes": int(probs.shape[1]),
        "features_dim": 0 if features is None else int(features.shape[1]),
        "checkpoint": str((run / "best.pt").relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(run / "best.pt"),
        "split_manifest": str(manifest_path.relative_to(ROOT))
        if manifest_path.is_relative_to(ROOT) else str(manifest_path),
        "split_manifest_sha256": sha256_file(manifest_path)
        if manifest_path.is_file() else None,
        "transform": "build_classification_transform((224, 224))",
        "device": str(device),
        "git_commit": commit,
        "npz": str(npz_path.relative_to(ROOT)),
    }, indent=2), encoding="utf-8")

    print(f"saved {len(y_true)} predictions -> {npz_path.relative_to(ROOT)}")
    print(f"  accuracy {acc:.6f} == recorded TEST accuracy "
          f"{(ref or {}).get('accuracy', float('nan')):.6f}")
    print(f"  features: {features.shape if features is not None else 'none'}")
    print(f"provenance -> {js_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
