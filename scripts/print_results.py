"""Print full results summary."""
import json, numpy as np

ab = json.load(open("paper/results/ablation_all.json"))
ms = json.load(open("paper/results/multi_seed_all.json"))
ext = json.load(open("paper/results/external_validation.json"))

print("=" * 80)
print("ABLATION RESULTS (10 epochs, GPU, RTX 3050)")
print("=" * 80)
fmt = "{:<40s} {:>8s} {:>8s} {:>8s} {:>8s}"
print(fmt.format("Model", "Acc", "F1", "AUC", "Params"))
print("-" * 74)
for k in sorted(ab):
    r = ab[k]
    t = r["test"]
    e = r["efficiency"]
    name = r.get("experiment", r.get("model_name", k))
    print(f"{name:40s} {t['accuracy']:.4f}  {t['macro_f1']:.4f}  {t['roc_auc']:.4f}  {e['params']/1e6:.1f}M")

print()
print("=" * 80)
print("MULTI-SEED RESULTS (mean +/- std)")
print("=" * 80)
model_groups = {}
for k, v in ms.items():
    mn = v.get("model_name", k.rsplit("_seed", 1)[0])
    model_groups.setdefault(mn, []).append(v)
for mn, runs in sorted(model_groups.items()):
    accs = [r["test"]["accuracy"] for r in runs]
    f1s = [r["test"]["macro_f1"] for r in runs]
    aucs = [r["test"]["roc_auc"] for r in runs]
    print(f"  {mn}:")
    print(f"    Acc: {np.mean(accs):.4f}+/-{np.std(accs):.4f}  F1: {np.mean(f1s):.4f}+/-{np.std(f1s):.4f}  AUC: {np.mean(aucs):.4f}+/-{np.std(aucs):.4f}")

print()
print("=" * 80)
print("EXTERNAL VALIDATION")
print("=" * 80)
for k in sorted(ext):
    v = ext[k]
    print(f"  {v['model']:35s} {v['dataset']:8s} Acc={v['accuracy']:.4f} F1={v['macro_f1']:.4f} AUC={v['roc_auc']:.4f}")

print()
print("=" * 80)
print("FIGURES & TABLES GENERATED")
print("=" * 80)
from pathlib import Path
for p in sorted(Path("paper/figures").glob("fig*.png")):
    print(f"  {p.name}")
for p in sorted(Path("paper/tables").glob("table*.md")):
    print(f"  {p.name}")
