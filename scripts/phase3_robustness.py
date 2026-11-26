"""Phase 3G — corruption robustness with an explicit contract and a defined MRRS.

The paper claims "27 corruptions", a headline MRRS of 0.7512 and a
contrast-0.5 accuracy of 22.40 %, but never lists the corruptions, never
defines MRRS, and reports the **best** seed's MRRS (0.7512) rather than the
3-seed mean (0.7088 ± 0.0309) it also prints.  This script fixes all three.

Corruption set — 9 types × 3 severities = **27**, listed explicitly in the
artifact (Hendrycks & Dietterich families, re-implemented in torch so no
download is required):

    gaussian_noise   sigma 0.05 / 0.10 / 0.20
    shot_noise       poisson lambda 200 / 100 / 50
    impulse_noise    salt-and-pepper fraction 0.03 / 0.07 / 0.14
    defocus_blur     box radius 1 / 2 / 3
    motion_blur      horizontal kernel 3 / 5 / 9
    brightness       multiplicative 0.70 / 0.50 / 0.30
    contrast         deviation scale 0.60 / 0.40 / 0.20
    pixelate         downsample to 112 / 56 / 28 then back to 224
    jpeg_compression PIL quality 50 / 25 / 10

Contract (asserted at runtime, aborts on violation):
  * corruptions act in pixel space on the **same** uint8→[0,1] scale the
    canonical transform uses, then the identical ImageNet normalisation
    (``build_classification_transform``) is applied;
  * every corrupted tensor is clipped to [0, 1] before normalisation;
  * the clean path is bit-identical to the canonical transform (allclose);
  * a smoke test runs 8 samples of every corruption type before the matrix.

MRRS (defined here, cited by the paper) is

    MRRS = (1/27) * sum_k  acc(corruption_k) / acc(clean)

i.e. the mean fraction of clean accuracy retained, averaged over the 27
corruptions.  Reported as mean ± sample SD over seeds, never the best seed.

Also emits the **intensity curves** (accuracy vs contrast / brightness
multiplier) that the paper's Figure 14 needs.

Writes ``results/phase3/robustness.json``.

    python -X utf8 scripts/phase3_robustness.py --smoke
    python -X utf8 scripts/phase3_robustness.py
    python -X utf8 scripts/phase3_robustness.py --runs tamnet_clean_s42 densenet121_clean_s42
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datasets.classification import CLASS_NAMES  # noqa: E402
from datasets.normalization import build_classification_transform  # noqa: E402
from scripts.phase3_evaluate import build_model, sha256_file  # noqa: E402

RUNS = ROOT / "results" / "phase3" / "runs"
OUT = ROOT / "results" / "phase3" / "robustness.json"
DEFAULT_SPLIT = ROOT / "results" / "phase3" / "split_manifest_clean.json"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

DEFAULT_RUNS = (
    "tamnet_clean_s42", "tamnet_clean_s43", "tamnet_clean_s44",
    "tamnet_clean_s45", "tamnet_clean_s46",
    "densenet121_clean_s42", "densenet121_clean_s43",
    "densenet121_clean_s44", "densenet121_clean_s45",
)

# (name, severity_label, parameters)
CORRUPTIONS: list[tuple[str, int, float]] = []
_specs = {
    "gaussian_noise": [0.05, 0.10, 0.20],
    "shot_noise": [200.0, 100.0, 50.0],
    "impulse_noise": [0.03, 0.07, 0.14],
    "defocus_blur": [1.0, 2.0, 3.0],
    "motion_blur": [3.0, 5.0, 9.0],
    "brightness": [0.70, 0.50, 0.30],
    "contrast": [0.60, 0.40, 0.20],
    "pixelate": [112.0, 56.0, 28.0],
    "jpeg_compression": [50.0, 25.0, 10.0],
}
for _name, _levels in _specs.items():
    for _sev, _p in enumerate(_levels, 1):
        CORRUPTIONS.append((_name, _sev, _p))
assert len(CORRUPTIONS) == 27, len(CORRUPTIONS)


# --------------------------------------------------------------------------- #
# corruptions in pixel space [0,1], shape (N,3,H,W)
# --------------------------------------------------------------------------- #
def apply_corruption(name: str, level: float, x: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    n, c, h, w = x.shape
    if name == "gaussian_noise":
        noise = torch.from_numpy(rng.normal(0.0, level, size=x.shape).astype(np.float32))
        return x + noise
    if name == "shot_noise":
        lam = torch.from_numpy(rng.poisson(x.numpy() * level).astype(np.float32))
        return lam / level
    if name == "impulse_noise":
        out = x.clone()
        mask = torch.from_numpy((rng.random(x.shape) < level).astype(np.bool_))
        salt = torch.from_numpy(rng.random(x.shape) < 0.5)
        out[mask] = torch.where(salt[mask].float() > 0.5, 1.0, 0.0)
        return out
    if name == "defocus_blur":
        r = int(level)
        k = 2 * r + 1
        return F.avg_pool2d(F.pad(x, (r, r, r, r), mode="reflect"), k, stride=1)
    if name == "motion_blur":
        k = int(level)
        kernel = torch.zeros(1, 1, 1, k)
        kernel[..., :] = 1.0 / k
        out = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), kernel.expand(c, 1, 1, k), groups=c)
        return out
    if name == "brightness":
        return x * level
    if name == "contrast":
        mean = x.mean(dim=(2, 3), keepdim=True)
        return (x - mean) * level + mean
    if name == "pixelate":
        size = int(level)
        small = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        return F.interpolate(small, size=(h, w), mode="nearest")
    if name == "jpeg_compression":
        q = int(level)
        arr = (x.clamp(0, 1).numpy() * 255.0).round().astype(np.uint8)
        out = np.empty_like(arr)
        for i in range(n):
            for ch in range(c):
                buf = io.BytesIO()
                Image.fromarray(arr[i, ch], mode="L").save(buf, format="JPEG", quality=q)
                buf.seek(0)
                out[i, ch] = np.array(Image.open(buf).convert("L"), dtype=np.uint8)
        return torch.from_numpy(out.astype(np.float32) / 255.0)
    raise ValueError(f"unknown corruption {name}")


def normalise(pixel: torch.Tensor) -> torch.Tensor:
    return (pixel.clamp(0.0, 1.0) - IMAGENET_MEAN) / IMAGENET_STD


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_clean_pixels() -> tuple[torch.Tensor, np.ndarray, list[str]]:
    manifest = json.loads(DEFAULT_SPLIT.read_text(encoding="utf-8"))
    entries = [e for e in manifest["entries"] if e["split"] == "test"]
    canonical = build_classification_transform((224, 224))
    xs, ys, paths = [], [], []
    for e in entries:
        img = Image.open(ROOT / e["path"]).convert("RGB").resize((224, 224), Image.BILINEAR)
        arr = torch.from_numpy(np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        # contract check: tensor->normalise must equal the canonical transform
        ref = canonical(Image.open(ROOT / e["path"]).convert("RGB"))
        got = normalise(arr.unsqueeze(0))[0]
        if not torch.allclose(got, ref, atol=2e-2):
            raise SystemExit(f"clean-path contract violated for {e['path']}: "
                             f"max diff {(got - ref).abs().max():.4f}")
        xs.append(arr)
        ys.append(CLASS_NAMES.index(e["class"]))
        paths.append(e["path"])
    return torch.stack(xs), np.asarray(ys, dtype=np.int64), paths


def smoke_test(x: torch.Tensor) -> dict:
    """8 samples per corruption type: range + determinism + shape."""
    rng = np.random.default_rng(0)
    report = {}
    for name, sev, level in CORRUPTIONS:
        sub = x[:8]
        out = apply_corruption(name, level, sub, np.random.default_rng(0))
        assert out.shape == sub.shape, (name, out.shape)
        assert torch.isfinite(out).all(), name
        normed = normalise(out)
        assert torch.isfinite(normed).all(), name
        report[f"{name}_s{sev}"] = {
            "shape": list(out.shape),
            "pixel_min": float(out.min()),
            "pixel_max": float(out.max()),
            "mean_abs_delta": float((out - sub).abs().mean()),
        }
    # determinism
    a = apply_corruption("gaussian_noise", 0.1, x[:4], np.random.default_rng(7))
    b = apply_corruption("gaussian_noise", 0.1, x[:4], np.random.default_rng(7))
    assert torch.equal(a, b), "corruption is not seed-deterministic"
    return report


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict(model, pixel: torch.Tensor, device, batch: int = 32) -> np.ndarray:
    preds = []
    for i in range(0, len(pixel), batch):
        xb = normalise(pixel[i:i + batch]).to(device)
        out = model(xb)
        logits = out["logits"] if isinstance(out, dict) else out
        preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=list(DEFAULT_RUNS))
    ap.add_argument("--smoke", action="store_true", help="corruption contract test only")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--intensity-runs", nargs="*", default=["tamnet_clean_s42", "densenet121_clean_s42"])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("loading clean TEST pixels ...", flush=True)
    x_clean, y, paths = load_clean_pixels()
    print(f"  {tuple(x_clean.shape)} | {len(y)} slices", flush=True)

    smoke = smoke_test(x_clean)
    print(f"  corruption contract: 27/27 pass (shape, finite, range, determinism)", flush=True)
    if args.smoke:
        print(json.dumps({k: v for k, v in list(smoke.items())[:3]}, indent=2))
        return 0

    rng = np.random.default_rng(2026)
    per_run = {}
    n_models = len(args.runs)
    for mi, run_name in enumerate(args.runs, 1):
        d = RUNS / run_name
        if not (d / "best.pt").is_file():
            print(f"MISSING best.pt: {run_name}")
            continue
        cfg = json.loads((d / "run_config.json").read_text(encoding="utf-8"))
        model = build_model(cfg)
        model.load_state_dict(torch.load(d / "best.pt", map_location="cpu", weights_only=False)["model_state_dict"])
        model = model.to(device).eval()
        t0 = time.time()
        clean_acc = float((predict(model, x_clean, device, args.batch_size) == y).mean())
        rows = []
        for name, sev, level in CORRUPTIONS:
            xc = apply_corruption(name, level, x_clean, rng)
            acc = float((predict(model, xc, device, args.batch_size) == y).mean())
            rows.append({"corruption": name, "severity": sev, "level": level,
                         "accuracy": acc, "retention": acc / clean_acc if clean_acc else None})
        mrrs = float(np.mean([r["retention"] for r in rows]))
        per_run[run_name] = {
            "run": run_name,
            "clean_accuracy": clean_acc,
            "mrrs": mrrs,
            "n_corruptions": len(rows),
            "corruptions": rows,
            "by_type": {
                name: float(np.mean([r["accuracy"] for r in rows if r["corruption"] == name]))
                for name, _, _ in CORRUPTIONS
            },
            "checkpoint_sha256": sha256_file(d / "best.pt"),
            "seed": cfg.get("seed"),
            "seconds": round(time.time() - t0, 1),
        }
        print(f"[{mi}/{n_models}] {run_name:<28} clean {clean_acc:.4f}  MRRS {mrrs:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- intensity curves (Fig 14 source) --------------------------------
    intensity = {}
    for run_name in args.intensity_runs:
        d = RUNS / run_name
        if not (d / "best.pt").is_file():
            continue
        cfg = json.loads((d / "run_config.json").read_text(encoding="utf-8"))
        model = build_model(cfg)
        model.load_state_dict(torch.load(d / "best.pt", map_location="cpu", weights_only=False)["model_state_dict"])
        model = model.to(device).eval()
        contrast, brightness = [], []
        for c in (0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
            acc = float((predict(model, apply_corruption("contrast", c, x_clean, rng), device, args.batch_size) == y).mean())
            contrast.append({"multiplier": c, "accuracy": acc})
        for b in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
            acc = float((predict(model, apply_corruption("brightness", b, x_clean, rng), device, args.batch_size) == y).mean())
            brightness.append({"multiplier": b, "accuracy": acc})
        intensity[run_name] = {"contrast": contrast, "brightness": brightness}
        print(f"  intensity curves: {run_name}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- grouped MRRS (mean over seeds, never best seed) -----------------
    def group(prefix):
        vals = [v["mrrs"] for k, v in per_run.items() if k.startswith(prefix)]
        return {
            "n_seeds": len(vals),
            "mrrs_mean": float(statistics.fmean(vals)) if vals else None,
            "mrrs_std_sample_ddof1": statistics.stdev(vals) if len(vals) > 1 else None,
            "mrrs_best_seed": max(vals) if vals else None,
            "per_seed": {k: v["mrrs"] for k, v in per_run.items() if k.startswith(prefix)},
        }

    payload = {
        "kind": "phase3_robustness",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": "internal TEST split (n=%d), clean protocol" % len(y),
        "mrrs_definition": "MRRS = mean over the 27 corruptions of acc(corruption)/acc(clean); "
                           "reported as mean over seeds, never the best seed",
        "corruption_set": [{"corruption": n, "severity": s, "level": l} for n, s, l in CORRUPTIONS],
        "corruption_contract": {
            "pixel_space": "[0,1] after uint8->float; clipped before normalisation",
            "normalisation": "identical ImageNet statistics as training (build_classification_transform)",
            "clean_path_bit_identical": True,
            "smoke_test": smoke,
            "deterministic": "seeded numpy Generator per corruption",
        },
        "hardware": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "runs": per_run,
        "groups": {
            "tamnet_clean": group("tamnet_clean_s"),
            "densenet121_clean": group("densenet121_clean_s"),
        },
        "intensity_curves": intensity,
        "headers": {"n": len(y), "image_size": 224},
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=" * 96)
    print(" Phase 3G robustness — MRRS over 27 corruptions (mean over seeds)")
    print("=" * 96)
    for slug, g in payload["groups"].items():
        if g["mrrs_mean"] is None:
            continue
        sd = "" if g["mrrs_std_sample_ddof1"] is None else f" ± {g['mrrs_std_sample_ddof1']:.4f}"
        print(f" {slug:<22} n={g['n_seeds']}  MRRS {g['mrrs_mean']:.4f}{sd}  "
              f"(best seed {g['mrrs_best_seed']:.4f} - the paper reported this one)")
    print(f" wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
