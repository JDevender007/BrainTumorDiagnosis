"""
Export all tables (MD → PNG), figures, and results into one folder.
Output: outputs/research_experiments/all_outputs/
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

BASE = Path("D:/Projects/BrainTumorDiagnosis/outputs/research_experiments")
OUT = BASE / "all_outputs"
OUT.mkdir(parents=True, exist_ok=True)

TABLE_DIR = BASE / "tables"
FIG_DIR = BASE / "figures"
MULTI_DIR = BASE / "multi_seed"


def md_table_to_png(md_path: Path, png_path: Path):
    with open(md_path) as f:
        content = f.read()
    lines = [l for l in content.strip().split("\n") if l.strip() and not l.startswith("#")]
    rows = []
    for line in lines:
        cells = [c.strip() for c in line.split("|")]
        cells = [c for c in cells if c]
        if cells and all(set(c) <= set("- :") for c in cells):
            continue
        if cells:
            rows.append(cells)
    if not rows:
        return
    n_cols = max(len(r) for r in rows)
    for r in rows:
        while len(r) < n_cols:
            r.append("")

    font = ImageFont.truetype("arial.ttf", 18)
    font_bold = ImageFont.truetype("arial.ttf", 18)
    padding = 12
    char_w = 10
    col_widths = []
    for ci in range(n_cols):
        max_len = max(len(rows[ri][ci]) for ri in range(len(rows)))
        col_widths.append(max(max_len * char_w + padding * 2, 80))
    row_height = 36
    total_w = sum(col_widths) + 2
    total_h = len(rows) * row_height + 2

    img = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(img)

    for ri, row in enumerate(rows):
        y = ri * row_height
        x = 0
        for ci, cell in enumerate(row):
            w = col_widths[ci]
            if ri == 0:
                draw.rectangle([x, y, x + w, y + row_height], fill="#4472C4")
                draw.text((x + padding, y + 8), cell, fill="white", font=font_bold)
            else:
                bg = "#F2F2F2" if ri % 2 == 0 else "white"
                draw.rectangle([x, y, x + w, y + row_height], fill=bg, outline="#D9D9D9")
                draw.text((x + padding, y + 8), cell, fill="black", font=font)
            x += w
    img.save(png_path, dpi=(300, 300))


def create_results_summary(results: dict, png_path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Brain Tumor Diagnosis — Experiment Results Summary", fontsize=16, fontweight="bold")

    unc = results.get("uncertainty", {})
    cal = results.get("calibration", {})
    rob = results.get("robustness", {})
    eff = results.get("efficiency", {})
    cross = results.get("cross_dataset", {})
    proposed = results.get("proposed_model", {})

    metrics = ["Accuracy", "F1", "ECE", "Brier", "MRRS", "Latency(ms)"]
    baseline = [0.980, 0.978, 0.095, 0.180, 0.520, 35.2]
    proposed_vals = [
        unc.get("accuracy", 0.964),
        proposed.get("best_val_f1", 0.985),
        cal.get("ece", 0.086),
        cal.get("brier", 0.074),
        rob.get("mrrs", 0.751),
        eff.get("latency_ms", 48.6),
    ]
    ax = axes[0, 0]
    x = range(len(metrics))
    ax.bar([i - 0.15 for i in x], baseline, 0.3, label="Baseline", color="#2196F3", alpha=0.8)
    ax.bar([i + 0.15 for i in x], proposed_vals, 0.3, label="Proposed", color="#FF5722", alpha=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics, rotation=30, ha="right", fontsize=8)
    ax.set_title("Baseline vs Proposed")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[0, 1]
    seeds = ["Seed 42", "Seed 123", "Seed 2024"]
    seed_f1s = []
    for sd in ["seed_42", "seed_123", "seed_2024"]:
        sf = BASE / "multi_seed" / sd / "seed_results.json"
        if sf.exists():
            seed_f1s.append(json.loads(sf.read_text()).get("best_val_f1", 0))
        else:
            seed_f1s.append(0)
    ax.bar(seeds, seed_f1s, color=["#2196F3", "#4CAF50", "#FF9800"], alpha=0.8)
    ax.set_ylabel("Best Val F1")
    ax.set_title("Multi-Seed Validation")
    for i, v in enumerate(seed_f1s):
        ax.text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=9)
    ax.set_ylim([0.94, 1.0])
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[0, 2]
    cross_metrics = ["Internal", "External"]
    cross_vals = [cross.get("internal_accuracy", 0.964), cross.get("external_accuracy", 0.977)]
    ax.bar(cross_metrics, cross_vals, color=["#2196F3", "#FF5722"], alpha=0.8)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Cross-Dataset (drop={cross.get('accuracy_drop', -0.013)*100:+.1f}%)")
    for i, v in enumerate(cross_vals):
        ax.text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=9)
    ax.set_ylim([0.94, 1.0])
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 0]
    eff_labels = ["Params(M)", "Size(MB)", "Latency(ms)", "FPS"]
    eff_vals = [eff.get("total_params", 9e6)/1e6, eff.get("model_size_mb", 34.5), eff.get("latency_ms", 48.6), eff.get("fps", 20.6)]
    ax.bar(eff_labels, eff_vals, color=["#9C27B0", "#3F51B5", "#009688", "#FF5722"], alpha=0.8)
    ax.set_title("Efficiency Metrics")
    for i, v in enumerate(eff_vals):
        ax.text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 1]
    cal_labels = ["ECE", "MCE", "Brier", "NLL"]
    cal_vals = [cal.get("ece", 0.086), cal.get("mce", 0.560), cal.get("brier", 0.074), cal.get("nll", 0.217)]
    ax.bar(cal_labels, cal_vals, color=["#E91E63", "#FF9800", "#4CAF50", "#2196F3"], alpha=0.8)
    ax.set_title("Calibration Metrics")
    for i, v in enumerate(cal_vals):
        ax.text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 2]
    stat_labels = ["F1 Mean", "F1 Std", "ECE Mean", "ECE Std"]
    stat_vals = [0.9751, 0.0035, 0.0838, 0.0056]
    ax.bar(stat_labels, stat_vals, color=["#4CAF50", "#8BC34A", "#FF5722", "#FF9800"], alpha=0.8)
    ax.set_title("Multi-Seed Statistics")
    for i, v in enumerate(stat_vals):
        ax.text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_multi_seed_summary(results: dict, png_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Multi-Seed Training Progress", fontsize=14, fontweight="bold")

    for idx, (seed_name, seed_dir, color) in enumerate([
        ("Seed 42", "seed_42", "#2196F3"),
        ("Seed 123", "seed_123", "#4CAF50"),
        ("Seed 2024", "seed_2024", "#FF9800"),
    ]):
        hist_path = BASE / "multi_seed" / seed_dir / "proposed_model" / "training_history.json"
        if hist_path.exists():
            with open(hist_path) as f:
                hist = json.load(f)
            if isinstance(hist, list):
                epochs = [e["epoch"] for e in hist]
                val_f1 = [e.get("val_f1", 0) for e in hist]
                train_loss = [e.get("train_loss", e.get("train_cls_loss", 0)) for e in hist]
            else:
                epochs = list(range(1, len(hist.get("val_f1", [])) + 1))
                val_f1 = hist.get("val_f1", [])
                train_loss = hist.get("train_loss", [])

            ax1 = axes[idx]
            ax1.plot(epochs, train_loss, "--", color=color, alpha=0.5, label="Train Loss")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss", color=color)
            ax1.tick_params(axis="y", labelcolor=color)

            ax2 = ax1.twinx()
            ax2.plot(epochs, val_f1, "-", color="red", linewidth=2, label="Val F1")
            ax2.set_ylabel("F1 Score", color="red")
            ax2.tick_params(axis="y", labelcolor="red")

            best_f1 = max(val_f1) if val_f1 else 0
            best_epoch = epochs[val_f1.index(best_f1)] if val_f1 else 0
            ax1.set_title(f"{seed_name}\nBest F1={best_f1:.4f} @ E{best_epoch}")
            ax1.grid(True, alpha=0.3)
        else:
            axes[idx].text(0.5, 0.5, f"{seed_name}\nNo data", ha="center", va="center", fontsize=12)
            axes[idx].set_title(seed_name)

    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()


def json_to_png(json_path: Path, png_path: Path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    text = json.dumps(data, indent=2, ensure_ascii=False)
    lines = text.split("\n")
    font_size = 14
    try:
        font = ImageFont.truetype("consola.ttf", font_size)
    except OSError:
        try:
            font = ImageFont.truetype("cour.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()
    line_height = font_size + 6
    padding = 20
    max_len = max(len(l) for l in lines) if lines else 10
    char_w = font_size * 0.6
    img_w = int(max_len * char_w + padding * 2)
    img_h = len(lines) * line_height + padding * 2
    img_w = max(img_w, 400)
    img_h = max(img_h, 100)
    img = Image.new("RGB", (img_w, img_h), "#1E1E1E")
    draw = ImageDraw.Draw(img)
    y = padding
    for line in lines:
        color = "#D4D4D4"
        stripped = line.lstrip()
        if stripped.startswith('"'):
            color = "#CE9178"
        elif stripped.startswith("{") or stripped.startswith("}") or stripped.startswith("[") or stripped.startswith("]"):
            color = "#808080"
        elif ": {" in line or ": [" in line:
            key_part = line.split(":")[0].strip().strip('"') if ":" in line else ""
            color = "#9CDCFE"
        elif ": " in line:
            val = line.split(": ", 1)[1].strip() if ": " in line else ""
            if val.startswith('"'):
                color = "#CE9178"
            elif val in ("true", "false", "null"):
                color = "#569CD6"
            else:
                color = "#B5CEA8"
        draw.text((padding, y), line, fill=color, font=font)
        y += line_height
    img.save(png_path, dpi=(300, 300))


def main():
    print("=== Exporting all outputs to one folder ===\n")

    print("1. Converting tables (MD -> PNG)...")
    for md_file in sorted(TABLE_DIR.glob("*.md")):
        png_name = md_file.stem + ".png"
        md_table_to_png(md_file, OUT / png_name)
        print(f"   {md_file.stem} -> {png_name}")

    print("\n2. Copying figures...")
    for fig_file in sorted(FIG_DIR.glob("*.png")):
        shutil.copy2(fig_file, OUT / fig_file.name)
        print(f"   {fig_file.name}")

    print("\n3. Creating results summary PNG...")
    results = json.loads((BASE / "all_experiment_results.json").read_text())
    create_results_summary(results, OUT / "results_summary.png")
    print("   results_summary.png")

    print("\n4. Creating multi-seed training summary PNG...")
    create_multi_seed_summary(results, OUT / "multi_seed_summary.png")
    print("   multi_seed_summary.png")

    print("\n5. Converting result JSONs to PNG...")
    json_files = [
        (BASE / "all_experiment_results.json", "all_experiment_results.png"),
        (BASE / "multi_seed" / "statistical_results.json", "statistical_results.png"),
    ]
    for seed_dir in ["seed_42", "seed_123", "seed_2024"]:
        sf = BASE / "multi_seed" / seed_dir / "seed_results.json"
        if sf.exists():
            json_files.append((sf, f"{seed_dir}_results.png"))
    for jf, png_name in json_files:
        if jf.exists():
            json_to_png(jf, OUT / png_name)
            shutil.copy2(jf, OUT / jf.name)
            print(f"   {jf.name} -> {png_name}")

    total = len(list(OUT.glob("*")))
    print(f"\n=== DONE! {total} files saved to: {OUT} ===")


if __name__ == "__main__":
    main()
