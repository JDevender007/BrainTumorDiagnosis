import json, os
f = "paper/results/ablation_all.json"
d = json.load(open(f)) if os.path.exists(f) else {}
print(f"{len(d)} results saved")
for k in sorted(d.keys()):
    v = d[k]
    print(f"  {k}: acc={v['test']['accuracy']:.4f} f1={v['test']['macro_f1']:.4f} auc={v['test']['roc_auc']:.4f}")

# Also check individual result files
ablation_dir = Path("paper/results/ablation") if os.path.exists("paper/results/ablation") else None
if ablation_dir:
    files = sorted(ablation_dir.glob("*_seed42.json"))
    print(f"\nIndividual files: {len(files)}")
    for f in files:
        print(f"  {f.name}")
