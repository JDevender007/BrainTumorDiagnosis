"""
==============================================================================
BRAIN TUMOR DIAGNOSIS - COMPLETE PIPELINE RUNNER
==============================================================================
This script runs the ENTIRE project pipeline from start to finish.

What it does:
  1. Checks environment and dependencies
  2. Verifies dataset exists
  3. Runs full experiment pipeline (7 phases)
  4. Runs multi-seed validation (3 seeds)
  5. Generates all 12 tables
  6. Generates all 16 figures
  7. Generates IEEE paper (LaTeX)
  8. Exports all outputs to one folder
  9. Tests on external Kaggle dataset
  10. Generates evaluation PNGs for new dataset

Usage:
  python run_everything.py

Output:
  outputs/research_experiments/all_outputs/  (42 files)
  outputs/kaggle_evaluation/                 (9 PNGs + JSON)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Ensure we're in the project root
os.chdir(Path(__file__).parent)
sys.path.insert(0, ".")


def print_header(text):
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_step(num, text):
    print(f"\n[{num}] {text}")
    print("-" * 50)


def check_environment():
    """Check that all dependencies are installed."""
    print_header("STEP 1: CHECKING ENVIRONMENT")

    required = [
        "torch", "torchvision", "numpy", "sklearn", "matplotlib",
        "PIL", "tqdm", "json", "pathlib"
    ]

    missing = []
    for pkg in required:
        try:
            __import__(pkg)
            print(f"  [OK] {pkg}")
        except ImportError:
            missing.append(pkg)
            print(f"  [MISSING] {pkg}")

    if missing:
        print(f"\nERROR: Missing packages: {missing}")
        print("Run: pip install -r requirements.txt")
        return False

    import torch
    print(f"\n  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  Using CPU (will be slower)")

    return True


def check_datasets():
    """Check that datasets exist."""
    print_header("STEP 2: CHECKING DATASETS")

    datasets = {
        "Internal (BraTS)": "data/raw/classification",
        "External (BRISC)": "data/external/brisc2025/brisc2025/classification_task/test",
        "External (Kaggle)": "data/external/kaggle_brain_tumor/Testing",
    }

    for name, path in datasets.items():
        if Path(path).exists():
            count = sum(1 for _ in Path(path).rglob("*") if _.is_file() and _.suffix.lower() in (".jpg", ".jpeg", ".png"))
            print(f"  [OK] {name}: {count} images ({path})")
        else:
            print(f"  [--] {name}: Not found ({path})")

    return True


def run_full_experiments():
    """Run the complete experiment pipeline."""
    print_header("STEP 3: RUNNING FULL EXPERIMENT PIPELINE")
    print("  This will train models and run all 7 phases.")
    print("  Estimated time: 2-4 hours on GPU, 6-10 hours on CPU.")

    from scripts.run_research_experiments import run_full_experiment
    from scripts.run_research_experiments import ExperimentConfig

    config = ExperimentConfig()
    results = run_full_experiment(config)
    return results


def run_multi_seed():
    """Run multi-seed validation."""
    print_header("STEP 4: RUNNING MULTI-SEED VALIDATION")
    print("  Training proposed model across 3 seeds (42, 123, 2024).")
    print("  Estimated time: 1-2 hours on GPU, 3-5 hours on CPU.")

    # Check if already done
    seed_file = Path("outputs/research_experiments/multi_seed/all_seed_results.json")
    if seed_file.exists():
        print("  [SKIP] Multi-seed results already exist.")
        with open(seed_file) as f:
            data = json.load(f)
        print(f"  Seeds found: {list(data.get('seeds', {}).keys())}")
        return data

    from scripts.run_multi_seed import main as run_multi_seed_main
    run_multi_seed_main()


def generate_tables():
    """Generate all 12 tables."""
    print_header("STEP 5: GENERATING TABLES")

    try:
        from scripts.generate_tables import main as gen_tables
        gen_tables()
        print("  All 12 tables generated.")
    except Exception as e:
        print(f"  Error generating tables: {e}")
        print("  Continuing...")


def generate_figures():
    """Generate all 16 figures."""
    print_header("STEP 6: GENERATING FIGURES")

    try:
        from scripts.generate_figures import main as gen_figures
        gen_figures()
        print("  All 16 figures generated.")
    except Exception as e:
        print(f"  Error generating figures: {e}")
        print("  Continuing...")


def generate_paper():
    """Generate IEEE paper."""
    print_header("STEP 7: GENERATING PAPER")

    try:
        from scripts.generate_paper import main as gen_paper
        gen_paper()
        print("  IEEE LaTeX paper generated.")
    except Exception as e:
        print(f"  Error generating paper: {e}")
        print("  Continuing...")


def export_all_outputs():
    """Export all outputs to one folder."""
    print_header("STEP 8: EXPORTING ALL OUTPUTS")

    try:
        from scripts.export_all_outputs import main as export_main
        export_main()
        print("  All outputs exported.")
    except Exception as e:
        print(f"  Error exporting outputs: {e}")
        print("  Continuing...")


def test_kaggle_dataset():
    """Test on external Kaggle dataset."""
    print_header("STEP 9: TESTING ON KAGGLE DATASET")

    kaggle_path = Path("data/external/kaggle_brain_tumor/Testing")
    if not kaggle_path.exists():
        print("  [SKIP] Kaggle dataset not found.")
        print("  Download from: https://zenodo.org/records/12735702")
        print(f"  Extract to: data/external/kaggle_brain_tumor/")
        return

    try:
        import torch
        import torch.nn.functional as F
        from torchvision import transforms
        from torch.utils.data import DataLoader
        from PIL import Image
        import numpy as np
        from sklearn.metrics import accuracy_score, f1_score

        from models.tumor_aware import TumorAwareMultiTaskNet, TumorAwareConfig
        from utils.device import get_device

        device = get_device()
        print(f"  Device: {device}")

        # Load model
        config = TumorAwareConfig(backbone="densenet121", pretrained=False)
        model = TumorAwareMultiTaskNet(config).to(device)
        ckpt = torch.load("outputs/research_experiments/proposed_model/best.pt",
                          map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        print("  Model loaded")

        # Load data
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        from scripts.test_quick import SimpleDataset
        dataset = SimpleDataset(kaggle_path, transform=transform)
        loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

        # Evaluate
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                outputs = model(images)
                if isinstance(outputs, dict):
                    preds = outputs["logits"].argmax(dim=1)
                else:
                    preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        preds = np.array(all_preds)
        labels = np.array(all_labels)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="weighted", zero_division=0)

        print(f"\n  Kaggle Dataset Results:")
        print(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        print(f"  F1:       {f1:.4f}")

    except Exception as e:
        print(f"  Error testing Kaggle dataset: {e}")


def generate_kaggle_evaluation():
    """Generate evaluation PNGs for Kaggle dataset."""
    print_header("STEP 10: GENERATING KAGGLE EVALUATION PNGS")

    kaggle_path = Path("data/external/kaggle_brain_tumor/Testing")
    if not kaggle_path.exists():
        print("  [SKIP] Kaggle dataset not found.")
        return

    try:
        from scripts.evaluate_kaggle_full import main as eval_main
        eval_main()
    except Exception as e:
        print(f"  Error: {e}")
        print("  Continuing...")


def print_final_summary():
    """Print final summary of all outputs."""
    print_header("FINAL SUMMARY")

    outputs = {
        "Research Experiments": "outputs/research_experiments/all_outputs",
        "Kaggle Evaluation": "outputs/kaggle_evaluation",
        "Tables": "outputs/research_experiments/tables",
        "Figures": "outputs/research_experiments/figures",
        "Paper": "outputs/research_experiments/paper",
        "Multi-Seed": "outputs/research_experiments/multi_seed",
    }

    total_files = 0
    for name, path in outputs.items():
        p = Path(path)
        if p.exists():
            files = list(p.rglob("*"))
            file_count = len([f for f in files if f.is_file()])
            total_files += file_count
            print(f"  {name:25s}: {file_count:3d} files ({path})")
        else:
            print(f"  {name:25s}: [--] Not found")

    print(f"\n  Total output files: {total_files}")

    # Key results
    results_file = Path("outputs/research_experiments/all_experiment_results.json")
    if results_file.exists():
        with open(results_file) as f:
            data = json.load(f)
        print("\n  Key Results:")
        if "proposed_model" in data:
            pm = data["proposed_model"]
            if "best_val_f1" in pm:
                print(f"    Proposed Model Val F1: {pm['best_val_f1']:.4f}")
        if "cross_dataset" in data:
            cd = data["cross_dataset"]
            print(f"    Internal Accuracy: {cd.get('internal_accuracy', 'N/A')}")
            print(f"    External Accuracy: {cd.get('external_accuracy', 'N/A')}")

    kaggle_results = Path("outputs/kaggle_evaluation/results.json")
    if kaggle_results.exists():
        with open(kaggle_results) as f:
            data = json.load(f)
        print(f"    Kaggle Accuracy: {data['metrics']['accuracy']:.4f}")

    print("\n" + "=" * 70)
    print("  PROJECT STATUS: COMPLETE")
    print("=" * 70)


def main():
    """Main runner."""
    start_time = time.time()

    print("=" * 70)
    print("  BRAIN TUMOR DIAGNOSIS - COMPLETE PIPELINE")
    print("  Tumor-Aware Self-Supervised Multi-Task Architecture")
    print("=" * 70)

    # Step 1: Check environment
    if not check_environment():
        return

    # Step 2: Check datasets
    check_datasets()

    # Ask user what to run
    print("\n" + "=" * 70)
    print("  WHAT TO RUN?")
    print("=" * 70)
    print("  1. Run everything (full pipeline - takes hours)")
    print("  2. Run only evaluation & figures (if models already trained)")
    print("  3. Run only new dataset test (Kaggle)")
    print("  4. Show summary only (no training)")

    choice = input("\n  Enter choice (1-4): ").strip()

    if choice == "1":
        run_full_experiments()
        run_multi_seed()
        generate_tables()
        generate_figures()
        generate_paper()
        export_all_outputs()
        test_kaggle_dataset()
        generate_kaggle_evaluation()

    elif choice == "2":
        generate_tables()
        generate_figures()
        generate_paper()
        export_all_outputs()

    elif choice == "3":
        test_kaggle_dataset()
        generate_kaggle_evaluation()

    elif choice == "4":
        pass  # Just show summary

    else:
        print("  Invalid choice. Running summary only.")

    # Final summary
    print_final_summary()

    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
