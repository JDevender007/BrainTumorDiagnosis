"""Save all current data and generate final reports. No training."""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

# Load all existing results
ablation = {}
ms = {}
ext = {}

p1 = Path("paper/results/ablation_all.json")
p2 = Path("paper/results/multi_seed_all.json")
p3 = Path("paper/results/external_validation.json")

if p1.exists(): ablation = json.load(open(p1))
if p2.exists(): ms = json.load(open(p2))
if p3.exists(): ext = json.load(open(p3))

print(f"Loaded: {len(ablation)} ablation, {len(ms)} multi-seed, {len(ext)} external")

# Save combined results
combined = {"ablation": ablation, "multi_seed": ms, "external": ext, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
with open("paper/results/all_results.json", "w") as f:
    json.dump(combined, f, indent=2)
print("Saved paper/results/all_results.json")
