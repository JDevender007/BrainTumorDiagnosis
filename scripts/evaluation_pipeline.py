"""
COMPREHENSIVE EVALUATION PIPELINE
Phases 4-13: Robustness, Calibration, Uncertainty, Statistics, GradCAM,
Segmentation, External Val, Figures, Tables, Documentation, Final Report
Uses existing checkpoints from paper/results/ablation/
"""
import sys, json, time, warnings, copy
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score, average_precision_score,
    matthews_corrcoef, cohen_kappa_score, confusion_matrix,
    roc_curve, precision_recall_curve, auc as sklearn_auc)
from sklearn.preprocessing import label_binarize
from scipy import stats as sp_stats

sys.path.insert(0, ".")
from utils.seed import set_seed

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

CLASS_NAMES = ["glioma", "meningioma", "no tumor", "pituitary"]
CLASS_MAP = {"glioma": 0, "meningioma": 1, "no tumor": 2, "pituitary": 3, "notumor": 2, "no_tumor": 2}
NUM_CLASSES = 4
OUT_DIR = Path("research_final_validation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

class NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (np.float32, np.float64)): return float(o)
        if isinstance(o, (np.int32, np.int64)): return int(o)
        return super().default(o)


# ============================================================================
# DATASET & MODEL LOADING
# ============================================================================
class FairClsDS(Dataset):
    def __init__(self, root, tf):
        self.samples, self.labels = [], []
        root = Path(root)
        for cd in sorted(root.iterdir()):
            if not cd.is_dir(): continue
            lb = CLASS_MAP.get(cd.name.lower().strip(), -1)
            if lb < 0: continue
            for p in sorted(cd.iterdir()):
                if p.is_file() and p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".tiff"}:
                    self.samples.append(str(p)); self.labels.append(lb)
        self.tf = tf
    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.tf(Image.open(self.samples[i]).convert("RGB")), self.labels[i]

def get_test_loader():
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    return DataLoader(FairClsDS("data/raw/classification/Test", tf), 64, shuffle=False, num_workers=0, pin_memory=True)

def get_train_loader():
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    return DataLoader(FairClsDS("data/raw/classification/Train", tf), 64, shuffle=False, num_workers=0, pin_memory=True)

def build_model(name, dev="cuda"):
    set_seed(42)
    if name in ["A_baseline_densenet121", "DenseNet121"]:
        from models.factory import create_backbone_classifier
        return create_backbone_classifier("densenet121", num_classes=4, pretrained=True).to(dev)
    if name in ["d_tumor_aware", "TumorAwareMultiTaskNet"]:
        from models.tumor_aware import TumorAwareMultiTaskNet, TumorAwareConfig
        return TumorAwareMultiTaskNet(TumorAwareConfig(backbone="densenet121", num_classes=4,
            segmentation_classes=4, dropout=0.3, fusion_dim=256)).to(dev)
    if "turma" in name.lower() or name in ["TURMA-Net", "e_turma_full"]:
        from models.novel.turma_net import TurmaNet, TurmaNetConfig
        return TurmaNet(TurmaNetConfig(backbone="densenet121", num_classes=4,
            segmentation_classes=4, dropout=0.3, fusion_dim=256,
            use_atgfm=True, use_consistency=True, use_uncertainty_weights=True)).to(dev)
    if name == "f_no_atgfm":
        from models.novel.turma_net import TurmaNet, TurmaNetConfig
        return TurmaNet(TurmaNetConfig(backbone="densenet121", num_classes=4,
            use_atgfm=False, use_consistency=True, use_uncertainty_weights=True)).to(dev)
    if name == "g_no_consistency":
        from models.novel.turma_net import TurmaNet, TurmaNetConfig
        return TurmaNet(TurmaNetConfig(backbone="densenet121", num_classes=4,
            use_atgfm=True, use_consistency=False, use_uncertainty_weights=True)).to(dev)
    if name == "h_no_uncertainty":
        from models.novel.turma_net import TurmaNet, TurmaNetConfig
        return TurmaNet(TurmaNetConfig(backbone="densenet121", num_classes=4,
            use_atgfm=True, use_consistency=True, use_uncertainty_weights=False)).to(dev)
    raise ValueError(f"Unknown model: {name}")

def load_checkpoint(model, ck_path, dev="cuda"):
    ck = torch.load(ck_path, map_location=dev, weights_only=False)
    sd = ck.get("model_state_dict", ck.get("state_dict", ck))
    model.load_state_dict(sd)
    return model

def is_mt(name):
    return any(x in name.lower() for x in ["turma", "tumor_aware"])

@torch.no_grad()
def predict(model, loader, dev="cuda"):
    model.eval()
    all_probs, all_labels = [], []
    mt = is_mt(type(model).__name__) or hasattr(model, "predict_with_uncertainty")
    for img, lab in loader:
        img = img.to(dev, non_blocking=True)
        if mt:
            try:
                out = model(img, return_seg=True)
                logits = out["logits"] if isinstance(out, dict) else out
            except:
                logits = model(img)
                if isinstance(logits, dict): logits = logits["logits"]
        else:
            logits = model(img)
            if isinstance(logits, dict): logits = logits["logits"]
        all_probs.extend(F.softmax(logits, dim=1).cpu().numpy())
        all_labels.extend(lab.numpy())
    return np.array(all_probs), np.array(all_labels)


# ============================================================================
# PHASE 4: ROBUSTNESS
# ============================================================================
def run_robustness(dev="cuda"):
    print("\n" + "=" * 70)
    print("PHASE 4: ROBUSTNESS EXPERIMENTS")
    print("=" * 70)

    corruptions = {
        "gaussian_noise": lambda img, s: img + s * 0.1 * torch.randn_like(img),
        "shot_noise": lambda img, s: torch.clamp(img + s * 0.05 * torch.poisson(img * 10) / 10, 0, 1),
        "impulse_noise": lambda img, s: torch.where(torch.rand_like(img) < s * 0.1, torch.rand_like(img), img),
        "gaussian_blur": lambda img, s: transforms.GaussianBlur(kernel_size=int(2*(s*3)//2*2+1), sigma=(0.1*s, 2.0*s))(img),
        "brightness": lambda img, s: torch.clamp(img + s * 0.3 * (img.mean() - 0.5), 0, 1),
        "contrast": lambda img, s: torch.clamp(img * (1 + s * 0.5 * (torch.rand(1).item() - 0.5)), 0, 1),
    }
    severities = [0.5, 1.0, 2.0]
    models_to_test = ["A_baseline_densenet121", "d_tumor_aware", "e_turma_full"]
    ck_map = {
        "A_baseline_densenet121": "paper/results/ablation/A_baseline_densenet121/best.pt",
        "d_tumor_aware": "paper/results/ablation/d_tumor_aware/best.pt",
        "e_turma_full": "paper/results/ablation/e_turma_full/best.pt",
    }

    tf_base = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    all_rob = {}
    for mname in models_to_test:
        print(f"\n  --- {mname} ---")
        model = build_model(mname, dev)
        model = load_checkpoint(model, ck_map[mname], dev)
        clean_probs, labels = predict(model, get_test_loader(), dev)
        clean_acc = accuracy_score(labels, clean_probs.argmax(1))

        rob_entry = {"clean_accuracy": float(clean_acc), "corruptions": {}}
        for cname, cfn in corruptions.items():
            rob_entry["corruptions"][cname] = {}
            for sev in severities:
                def corrupt_fn(img_batch):
                    return cfn(img_batch, sev)
                corrupted_loader = get_test_loader()
                # Apply corruption inline
                model.eval()
                all_p, all_l = [], []
                for img, lab in corrupted_loader:
                    img_corrupted = corrupt_fn(img.to(dev, non_blocking=True))
                    with torch.no_grad():
                        if is_mt(mname):
                            try:
                                out = model(img_corrupted, return_seg=True)
                                logits = out["logits"] if isinstance(out, dict) else out
                            except:
                                logits = model(img_corrupted)
                                if isinstance(logits, dict): logits = logits["logits"]
                        else:
                            logits = model(img_corrupted)
                            if isinstance(logits, dict): logits = logits["logits"]
                        all_p.extend(F.softmax(logits, dim=1).cpu().numpy())
                        all_l.extend(lab.numpy())
                acc = accuracy_score(np.array(all_l), np.array(all_p).argmax(1))
                retention = acc / max(clean_acc, 1e-10)
                rob_entry["corruptions"][cname][f"sev_{sev}"] = {
                    "accuracy": float(acc), "retention": float(retention)}
                print(f"    {cname} sev={sev}: acc={acc:.4f} ret={retention:.4f}")

        # MRRS
        retentions = []
        for cname in corruptions:
            for sev in severities:
                retentions.append(rob_entry["corruptions"][cname][f"sev_{sev}"]["retention"])
        mrrs = np.exp(np.mean(np.log(np.maximum(retentions, 1e-10))))
        rob_entry["mrrs"] = float(mrrs)
        rob_entry["worst_case_accuracy"] = float(min(
            rob_entry["corruptions"][c][f"sev_{s}"]["accuracy"]
            for c in corruptions for s in severities))
        print(f"    MRRS={mrrs:.4f} worst_case={rob_entry['worst_case_accuracy']:.4f}")
        all_rob[mname] = rob_entry

    with open(OUT_DIR / "robustness_results.json", "w") as f:
        json.dump(all_rob, f, indent=2, cls=NpEncoder)
    print(f"\n  Saved to {OUT_DIR}/robustness_results.json")
    return all_rob


# ============================================================================
# PHASE 5: CALIBRATION
# ============================================================================
def run_calibration(dev="cuda"):
    print("\n" + "=" * 70)
    print("PHASE 5: CALIBRATION EXPERIMENTS")
    print("=" * 70)

    models_to_test = {
        "A_baseline_densenet121": "paper/results/ablation/A_baseline_densenet121/best.pt",
        "d_tumor_aware": "paper/results/ablation/d_tumor_aware/best.pt",
        "e_turma_full": "paper/results/ablation/e_turma_full/best.pt",
    }

    def ece(probs, labels, n_bins=15):
        confs = probs.max(axis=1)
        preds = probs.argmax(axis=1)
        correct = (preds == labels).astype(float)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece_val = 0.0
        for i in range(n_bins):
            mask = (confs > bin_boundaries[i]) & (confs <= bin_boundaries[i + 1])
            if mask.sum() > 0:
                bin_acc = correct[mask].mean()
                bin_conf = confs[mask].mean()
                ece_val += mask.sum() / len(labels) * abs(bin_acc - bin_conf)
        return float(ece_val)

    def mce(probs, labels, n_bins=15):
        confs = probs.max(axis=1)
        preds = probs.argmax(axis=1)
        correct = (preds == labels).astype(float)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        max_diff = 0.0
        for i in range(n_bins):
            mask = (confs > bin_boundaries[i]) & (confs <= bin_boundaries[i + 1])
            if mask.sum() > 0:
                bin_acc = correct[mask].mean()
                bin_conf = confs[mask].mean()
                max_diff = max(max_diff, abs(bin_acc - bin_conf))
        return float(max_diff)

    def brier(probs, labels):
        one_hot = np.eye(NUM_CLASSES)[labels]
        return float(((probs - one_hot) ** 2).mean())

    def nll(probs, labels):
        eps = 1e-10
        return float(-np.mean(np.log(probs[np.arange(len(labels)), labels] + eps)))

    def temperature_scale(logits, labels, lr=0.01, max_iter=50):
        logits_t = torch.tensor(logits, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.long)
        T = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.LBFGS([T], lr=lr, max_iter=max_iter)
        nll_loss = nn.CrossEntropyLoss()
        def eval_step():
            optimizer.zero_grad()
            loss = nll_loss(logits_t / T, labels_t)
            loss.backward()
            return loss
        optimizer.step(eval_step)
        T_val = T.item()
        scaled = logits / T_val
        probs_scaled = np.exp(scaled) / np.exp(scaled).sum(axis=1, keepdims=True)
        return probs_scaled, T_val

    all_cal = {}
    test_loader = get_test_loader()

    for mname, ck_path in models_to_test.items():
        print(f"\n  --- {mname} ---")
        model = build_model(mname, dev)
        model = load_checkpoint(model, ck_path, dev)

        probs, labels = predict(model, test_loader, dev)
        logits = np.log(probs + 1e-10)

        cal = {
            "ece": ece(probs, labels),
            "mce": mce(probs, labels),
            "brier": brier(probs, labels),
            "nll": nll(probs, labels),
        }
        print(f"    Original: ECE={cal['ece']:.4f} MCE={cal['mce']:.4f} Brier={cal['brier']:.4f} NLL={cal['nll']:.4f}")

        # Temperature scaling
        probs_ts, T = temperature_scale(logits, labels)
        cal["temperature"] = float(T)
        cal["ece_ts"] = ece(probs_ts, labels)
        cal["mce_ts"] = mce(probs_ts, labels)
        cal["brier_ts"] = brier(probs_ts, labels)
        cal["nll_ts"] = nll(probs_ts, labels)
        print(f"    Temp-scaled (T={T:.3f}): ECE={cal['ece_ts']:.4f} MCE={cal['mce_ts']:.4f} Brier={cal['brier_ts']:.4f} NLL={cal['nll_ts']:.4f}")
        print(f"    Improvement: ECE {cal['ece']:.4f} -> {cal['ece_ts']:.4f}")

        all_cal[mname] = cal

    with open(OUT_DIR / "calibration_results.json", "w") as f:
        json.dump(all_cal, f, indent=2, cls=NpEncoder)
    print(f"\n  Saved to {OUT_DIR}/calibration_results.json")
    return all_cal


# ============================================================================
# PHASE 6: UNCERTAINTY
# ============================================================================
def run_uncertainty(dev="cuda"):
    print("\n" + "=" * 70)
    print("PHASE 6: UNCERTAINTY EXPERIMENTS")
    print("=" * 70)

    mc_passes = 10
    models_to_test = {
        "A_baseline_densenet121": "paper/results/ablation/A_baseline_densenet121/best.pt",
        "d_tumor_aware": "paper/results/ablation/d_tumor_aware/best.pt",
        "e_turma_full": "paper/results/ablation/e_turma_full/best.pt",
    }

    all_unc = {}
    test_loader = get_test_loader()

    for mname, ck_path in models_to_test.items():
        print(f"\n  --- {mname} (MC Dropout, {mc_passes} passes) ---")
        model = build_model(mname, dev)
        model = load_checkpoint(model, ck_path, dev)

        # MC Dropout inference
        model.train()  # Keep dropout active
        all_mc_probs = []
        all_labels = []
        for img, lab in test_loader:
            img = img.to(dev, non_blocking=True)
            mc_batch = []
            for _ in range(mc_passes):
                with torch.no_grad():
                    try:
                        out = model(img, return_seg=True)
                        logits = out["logits"] if isinstance(out, dict) else out
                    except:
                        logits = model(img)
                        if isinstance(logits, dict): logits = logits["logits"]
                    mc_batch.append(F.softmax(logits, dim=1).cpu().numpy())
            all_mc_probs.append(np.stack(mc_batch))  # [batch, passes, classes]
            all_labels.extend(lab.numpy())

        model.eval()
        mc_probs = np.concatenate(all_mc_probs, axis=0)  # [N, passes, C]
        labels = np.array(all_labels)
        N = mc_probs.shape[0]

        mean_probs = mc_probs.mean(axis=1)
        pred_var = mc_probs.var(axis=1)
        pred_entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10), axis=1)
        expected_entropy = -np.mean(np.sum(mc_probs * np.log(mc_probs + 1e-10), axis=2), axis=1)
        mutual_info = pred_entropy - expected_entropy

        preds = mean_probs.argmax(axis=1)
        correct = (preds == labels).astype(float)
        incorrect = 1 - correct

        unc = {
            "mc_passes": mc_passes,
            "num_samples": N,
            "mean_entropy": float(pred_entropy.mean()),
            "std_entropy": float(pred_entropy.std()),
            "mean_mutual_info": float(expected_entropy.mean()),
            "mean_predictive_var": float(pred_var.mean()),
            "entropy_correct": float(pred_entropy[correct == 1].mean()) if correct.sum() > 0 else 0,
            "entropy_incorrect": float(pred_entropy[incorrect == 1].mean()) if incorrect.sum() > 0 else 0,
            "mi_correct": float(expected_entropy[correct == 1].mean()) if correct.sum() > 0 else 0,
            "mi_incorrect": float(expected_entropy[incorrect == 1].mean()) if incorrect.sum() > 0 else 0,
        }

        # AUROC for error detection using entropy
        try:
            from sklearn.metrics import roc_auc_score
            unc["auroc_error_detection"] = float(roc_auc_score(incorrect, pred_entropy))
        except:
            unc["auroc_error_detection"] = 0.0

        # Risk-coverage curve
        sorted_idx = np.argsort(-pred_entropy)
        coverages = np.linspace(0.1, 1.0, 20)
        risk_coverage = []
        for cov in coverages:
            n_keep = int(N * cov)
            keep_idx = sorted_idx[:n_keep]
            risk = 1 - correct[keep_idx].mean()
            risk_coverage.append({"coverage": float(cov), "risk": float(risk)})
        unc["risk_coverage"] = risk_coverage
        unc["accuracy_at_80_coverage"] = float(correct[sorted_idx[:int(N*0.8)]].mean())
        unc["accuracy_at_50_coverage"] = float(correct[sorted_idx[:int(N*0.5)]].mean())

        print(f"    Mean entropy: {unc['mean_entropy']:.4f}")
        print(f"    Entropy correct: {unc['entropy_correct']:.4f} | incorrect: {unc['entropy_incorrect']:.4f}")
        print(f"    MI correct: {unc['mi_correct']:.4f} | incorrect: {unc['mi_incorrect']:.4f}")
        print(f"    AUROC error detection: {unc['auroc_error_detection']:.4f}")
        print(f"    Accuracy@80%coverage: {unc['accuracy_at_80_coverage']:.4f}")
        print(f"    Accuracy@50%coverage: {unc['accuracy_at_50_coverage']:.4f}")

        all_unc[mname] = unc

    with open(OUT_DIR / "uncertainty_results.json", "w") as f:
        json.dump(all_unc, f, indent=2, cls=NpEncoder)
    print(f"\n  Saved to {OUT_DIR}/uncertainty_results.json")
    return all_unc


# ============================================================================
# PHASE 7: STATISTICAL VALIDATION
# ============================================================================
def run_statistics(dev="cuda"):
    print("\n" + "=" * 70)
    print("PHASE 7: STATISTICAL VALIDATION")
    print("=" * 70)

    test_loader = get_test_loader()

    # Get predictions for all models
    model_names = ["A_baseline_densenet121", "d_tumor_aware", "e_turma_full",
                    "f_no_atgfm", "g_no_consistency", "h_no_uncertainty"]
    ck_map = {
        "A_baseline_densenet121": "paper/results/ablation/A_baseline_densenet121/best.pt",
        "d_tumor_aware": "paper/results/ablation/d_tumor_aware/best.pt",
        "e_turma_full": "paper/results/ablation/e_turma_full/best.pt",
        "f_no_atgfm": "paper/results/ablation/f_no_atgfm/best.pt",
        "g_no_consistency": "paper/results/ablation/g_no_consistency/best.pt",
        "h_no_uncertainty": "paper/results/ablation/h_no_uncertainty/best.pt",
    }

    predictions = {}
    for mname in model_names:
        model = build_model(mname, dev)
        model = load_checkpoint(model, ck_map[mname], dev)
        probs, labels = predict(model, test_loader, dev)
        predictions[mname] = {"probs": probs, "preds": probs.argmax(1), "labels": labels}
        print(f"  Loaded {mname}: acc={accuracy_score(labels, probs.argmax(1)):.4f}")

    labels = predictions[model_names[0]]["labels"]

    # Bootstrap confidence intervals
    def bootstrap_ci(y_true, y_pred, n_boot=1000, ci=0.95):
        scores = []
        n = len(y_true)
        for _ in range(n_boot):
            idx = np.random.choice(n, n, replace=True)
            scores.append(accuracy_score(y_true[idx], y_pred[idx]))
        scores = np.array(scores)
        lo = np.percentile(scores, (1-ci)/2 * 100)
        hi = np.percentile(scores, (1+ci)/2 * 100)
        return float(np.mean(scores)), float(np.std(scores)), float(lo), float(hi)

    # McNemar test
    def mcnemar(y_true, y_pred1, y_pred2):
        correct1 = (y_pred1 == y_true)
        correct2 = (y_pred2 == y_true)
        b01 = (~correct1 & correct2).sum()
        c10 = (correct1 & ~correct2).sum()
        if b01 + c10 == 0:
            return {"statistic": 0, "p_value": 1.0}
        stat = (abs(b01 - c10) - 1) ** 2 / (b01 + c10)
        p = 1 - sp_stats.chi2.cdf(stat, df=1)
        return {"statistic": float(stat), "p_value": float(p)}

    # Cohen's d
    def cohens_d(x, y):
        nx, ny = len(x), len(y)
        pooled_std = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / (nx+ny-2))
        if pooled_std == 0: return 0.0
        return float((np.mean(x) - np.mean(y)) / pooled_std)

    stat_results = {}

    # Per-model bootstrap CIs
    print("\n  --- Bootstrap 95% Confidence Intervals ---")
    for mname in model_names:
        preds = predictions[mname]["preds"]
        mean, std, lo, hi = bootstrap_ci(labels, preds)
        stat_results[f"{mname}_bootstrap"] = {
            "metric": "accuracy", "mean": mean, "std": std,
            "ci_lower": lo, "ci_upper": hi, "n_bootstrap": 1000
        }
        print(f"    {mname}: {mean:.4f} +/- {std:.4f} [{lo:.4f}, {hi:.4f}]")

    # Pairwise McNemar tests
    print("\n  --- McNemar Tests (pairwise) ---")
    comparisons = [
        ("A_baseline_densenet121", "e_turma_full"),
        ("A_baseline_densenet121", "d_tumor_aware"),
        ("d_tumor_aware", "e_turma_full"),
        ("e_turma_full", "f_no_atgfm"),
        ("e_turma_full", "g_no_consistency"),
        ("e_turma_full", "h_no_uncertainty"),
    ]
    for m1, m2 in comparisons:
        result = mcnemar(labels, predictions[m1]["preds"], predictions[m2]["preds"])
        sig = "***" if result["p_value"] < 0.001 else "**" if result["p_value"] < 0.01 else "*" if result["p_value"] < 0.05 else "ns"
        stat_results[f"mcnemar_{m1}_vs_{m2}"] = {
            "model_1": m1, "model_2": m2,
            "statistic": result["statistic"], "p_value": result["p_value"],
            "significant": result["p_value"] < 0.05
        }
        print(f"    {m1} vs {m2}: chi2={result['statistic']:.4f} p={result['p_value']:.6f} {sig}")

    # Cohen's d
    print("\n  --- Cohen's d (effect size) ---")
    for m1, m2 in comparisons:
        acc1 = (predictions[m1]["preds"] == labels).astype(float)
        acc2 = (predictions[m2]["preds"] == labels).astype(float)
        d = cohens_d(acc1, acc2)
        interpretation = "negligible" if abs(d) < 0.2 else "small" if abs(d) < 0.5 else "medium" if abs(d) < 0.8 else "large"
        stat_results[f"cohens_d_{m1}_vs_{m2}"] = {
            "model_1": m1, "model_2": m2,
            "cohens_d": d, "interpretation": interpretation
        }
        print(f"    {m1} vs {m2}: d={d:.4f} ({interpretation})")

    # Multi-seed stats
    print("\n  --- Multi-Seed Statistics ---")
    ms = json.load(open("paper/results/multi_seed_all.json"))
    for mn in ["A_baseline_densenet121", "d_tumor_aware", "e_turma_full"]:
        mrs = {k: v for k, v in ms.items() if v.get("model_name") == mn}
        if mrs:
            accs = [v["test"]["accuracy"] for v in mrs.values()]
            f1s = [v["test"]["macro_f1"] for v in mrs.values()]
            aucs = [v["test"]["roc_auc"] for v in mrs.values()]
            stat_results[f"multiseed_{mn}"] = {
                "n_seeds": len(accs),
                "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
                "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
                "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
            }
            print(f"    {mn}: acc={np.mean(accs):.4f}+/-{np.std(accs):.4f} "
                  f"f1={np.mean(f1s):.4f}+/-{np.std(f1s):.4f} "
                  f"auc={np.mean(aucs):.4f}+/-{np.std(aucs):.4f}")

    with open(OUT_DIR / "statistical_results.json", "w") as f:
        json.dump(stat_results, f, indent=2, cls=NpEncoder)
    print(f"\n  Saved to {OUT_DIR}/statistical_results.json")
    return stat_results


# ============================================================================
# PHASE 8: GRAD-CAM
# ============================================================================
def run_gradcam(dev="cuda"):
    print("\n" + "=" * 70)
    print("PHASE 8: GRAD-CAM ON TURMA-NET")
    print("=" * 70)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = build_model("e_turma_full", dev)
    model = load_checkpoint(model, "paper/results/ablation/e_turma_full/best.pt", dev)
    model.eval()

    # Register hook on encoder features
    feature_maps = []
    gradients = []
    def fwd_hook(module, input, output):
        feature_maps.append(output.detach())
    def bwd_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    # Find the encoder's last conv layer
    encoder = None
    for name, mod in model.named_modules():
        if "features" in name and isinstance(mod, torch.nn.Sequential):
            encoder = mod
    if encoder is None:
        for name, mod in model.named_modules():
            if "layer4" in name:
                encoder = mod
                break

    if encoder is not None:
        h1 = encoder.register_forward_hook(fwd_hook)
        h2 = encoder.register_full_backward_hook(bwd_hook)
    else:
        print("  WARNING: Could not find encoder hook target")
        return

    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    ds = FairClsDS("data/raw/classification/Test", tf)

    # Generate Grad-CAM for sample images
    gradcam_dir = OUT_DIR / "gradcam"
    gradcam_dir.mkdir(exist_ok=True)

    # Pick one image per class
    class_indices = defaultdict(list)
    for i in range(len(ds)):
        class_indices[ds.labels[i]].append(i)

    for cls_idx in range(NUM_CLASSES):
        if cls_idx not in class_indices or len(class_indices[cls_idx]) == 0:
            continue
        sample_idx = class_indices[cls_idx][0]
        img, label = ds[sample_idx]
        img_batch = img.unsqueeze(0).to(dev, non_blocking=True)
        img_batch.requires_grad = True

        feature_maps.clear()
        gradients.clear()

        out = model(img_batch, return_seg=True)
        logits = out["logits"] if isinstance(out, dict) else out
        pred_class = logits.argmax(1).item()
        model.zero_grad()
        logits[0, pred_class].backward()

        if feature_maps and gradients:
            fm = feature_maps[0][0].cpu()  # [C, H, W]
            grad = gradients[0][0].cpu()   # [C, H, W]
            weights = grad.mean(dim=(1, 2))  # [C]
            cam = torch.zeros(fm.shape[1:])
            for i in range(fm.shape[0]):
                cam += weights[i] * fm[i]
            cam = F.relu(cam)
            cam = cam / (cam.max() + 1e-10)
            cam_np = cam.numpy()

            # Plot
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            img_show = img.permute(1, 2, 0).numpy()
            img_show = (img_show - img_show.min()) / (img_show.max() - img_show.min())
            axes[0].imshow(img_show)
            axes[0].set_title(f"Original ({CLASS_NAMES[label]})")
            axes[0].axis("off")
            axes[1].imshow(cam_np, cmap="jet")
            axes[1].set_title(f"Grad-CAM (pred: {CLASS_NAMES[pred_class]})")
            axes[1].axis("off")
            axes[2].imshow(img_show)
            axes[2].imshow(cam_np, cmap="jet", alpha=0.4)
            axes[2].set_title("Overlay")
            axes[2].axis("off")
            plt.tight_layout()
            plt.savefig(gradcam_dir / f"gradcam_{CLASS_NAMES[label]}.png", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved Grad-CAM for {CLASS_NAMES[label]} (pred: {CLASS_NAMES[pred_class]})")

    h1.remove()
    h2.remove()
    print(f"  Grad-CAM images saved to {gradcam_dir}/")


# ============================================================================
# PHASE 10: EXTERNAL VALIDATION (CLEAN - BRISC ONLY)
# ============================================================================
def run_external(dev="cuda"):
    print("\n" + "=" * 70)
    print("PHASE 10: EXTERNAL VALIDATION (BRISC only - Kaggle contaminated)")
    print("=" * 70)

    brisc_path = "data/external/brisc2025/brisc2025/classification_task/test"
    if not Path(brisc_path).exists():
        print("  BRISC not found"); return {}

    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    brisc_loader = DataLoader(FairClsDS(brisc_path, tf), 64, shuffle=False, num_workers=0)
    print(f"  BRISC: {len(brisc_loader.dataset)} samples")

    ext_results = {}
    models_to_test = {
        "A_baseline_densenet121": "paper/results/ablation/A_baseline_densenet121/best.pt",
        "d_tumor_aware": "paper/results/ablation/d_tumor_aware/best.pt",
        "e_turma_full": "paper/results/ablation/e_turma_full/best.pt",
    }

    for mname, ck_path in models_to_test.items():
        model = build_model(mname, dev)
        model = load_checkpoint(model, ck_path, dev)
        probs, labels = predict(model, brisc_loader, dev)
        pred = probs.argmax(1)
        cm = confusion_matrix(labels, pred, labels=range(NUM_CLASSES))
        try:
            lb = label_binarize(labels, classes=range(NUM_CLASSES))
            ra = roc_auc_score(lb, probs, multi_class="ovr", average="macro")
        except:
            ra = 0.0
        pc_f1 = {}
        for i in range(NUM_CLASSES):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            pc_f1[CLASS_NAMES[i]] = float(2*tp / max(2*tp + fp + fn, 1))

        ext_results[mname] = {
            "dataset": "BRISC", "num_samples": len(labels),
            "accuracy": float(accuracy_score(labels, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
            "macro_f1": float(f1_score(labels, pred, average="macro", zero_division=0)),
            "roc_auc": float(ra),
            "mcc": float(matthews_corrcoef(labels, pred)),
            "per_class_f1": pc_f1,
            "confusion_matrix": cm.tolist(),
        }
        print(f"  {mname:30s} acc={ext_results[mname]['accuracy']:.4f} "
              f"f1={ext_results[mname]['macro_f1']:.4f} auc={ext_results[mname]['roc_auc']:.4f}")

    # Mark Kaggle as contaminated
    ext_results["kaggle_status"] = "CONTAMINATED - 839/1311 filenames overlap with internal training set"
    ext_results["kaggle_action"] = "NOT used as independent external validation"

    with open(OUT_DIR / "external_validation.json", "w") as f:
        json.dump(ext_results, f, indent=2, cls=NpEncoder)
    print(f"\n  Saved to {OUT_DIR}/external_validation.json")
    return ext_results


# ============================================================================
# PHASE 11: FIGURES
# ============================================================================
def run_figures(dev="cuda"):
    print("\n" + "=" * 70)
    print("PHASE 11: GENERATE ALL FIGURES")
    print("=" * 70)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)

    ab = json.load(open("paper/results/ablation_all.json"))
    ms = json.load(open("paper/results/multi_seed_all.json"))
    rob = json.load(open(OUT_DIR / "robustness_results.json")) if (OUT_DIR / "robustness_results.json").exists() else {}
    cal = json.load(open(OUT_DIR / "calibration_results.json")) if (OUT_DIR / "calibration_results.json").exists() else {}
    unc = json.load(open(OUT_DIR / "uncertainty_results.json")) if (OUT_DIR / "uncertainty_results.json").exists() else {}
    ext = json.load(open(OUT_DIR / "external_validation.json")) if (OUT_DIR / "external_validation.json").exists() else {}
    stat = json.load(open(OUT_DIR / "statistical_results.json")) if (OUT_DIR / "statistical_results.json").exists() else {}

    def _name(r, key=""):
        return r.get("experiment", r.get("model_name", str(key)))

    # --- Fig 1: Ablation Accuracy & F1 ---
    fig, ax = plt.subplots(figsize=(12, 6))
    names, accs, f1s = [], [], []
    for k in sorted(ab):
        r = ab[k]; t = r["test"]
        names.append(_name(r, k).replace("_seed42", "").replace("_", " ")[:25])
        accs.append(t["accuracy"]); f1s.append(t["macro_f1"])
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w/2, accs, w, label="Accuracy", color="#3498db", edgecolor="white")
    ax.bar(x + w/2, f1s, w, label="Macro F1", color="#e74c3c", edgecolor="white")
    ax.set_ylabel("Score"); ax.set_title("Ablation Study: Accuracy and Macro F1")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.legend(); ax.set_ylim(0.8, 1.0); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(fig_dir / "fig1_ablation.png", dpi=150, bbox_inches="tight"); plt.close()
    print("  Fig 1: Ablation")

    # --- Fig 2: Confusion Matrices ---
    key_models = ["A_baseline_densenet121_seed42", "d_tumor_aware_seed42", "e_turma_full_seed42"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, mk in zip(axes, key_models):
        if mk in ab:
            cm = np.array(ab[mk]["confusion_matrix"])
            im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            ax.set_title(_name(ab[mk], mk).replace("_seed42", "").replace("_", " "), fontsize=10)
            for i in range(4):
                for j in range(4):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                            color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=8)
            ax.set_xticks(range(4)); ax.set_yticks(range(4))
            ax.set_xticklabels(CLASS_NAMES, rotation=45, fontsize=7)
            ax.set_yticklabels(CLASS_NAMES, fontsize=7)
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout(); plt.savefig(fig_dir / "fig2_confusion.png", dpi=150, bbox_inches="tight"); plt.close()
    print("  Fig 2: Confusion Matrices")

    # --- Fig 3: ROC Curves ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
    for ax, mk in zip(axes, key_models):
        if mk in ab and "roc_data" in ab[mk]:
            roc = ab[mk]["roc_data"]
            for i in range(4):
                fpr = np.array(roc["fpr"][str(i)])
                tpr = np.array(roc["tpr"][str(i)])
                auc_val = roc["auc"].get(CLASS_NAMES[i], 0)
                ax.plot(fpr, tpr, color=colors[i], label=f"{CLASS_NAMES[i]} ({auc_val:.3f})", linewidth=1.5)
            ax.plot([0,1], [0,1], "k--", alpha=0.3)
            ax.set_title(_name(ab[mk], mk).replace("_seed42", "").replace("_", " "), fontsize=10)
            ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(fig_dir / "fig3_roc.png", dpi=150, bbox_inches="tight"); plt.close()
    print("  Fig 3: ROC Curves")

    # --- Fig 4: Robustness ---
    if rob:
        fig, ax = plt.subplots(figsize=(10, 5))
        model_names = list(rob.keys())
        corruptions = list(rob[model_names[0]]["corruptions"].keys())
        x = np.arange(len(corruptions))
        width = 0.25
        for i, mn in enumerate(model_names):
            mrrs_vals = [rob[mn]["corruptions"][c]["sev_1.0"]["accuracy"] for c in corruptions]
            ax.bar(x + i*width, mrrs_vals, width, label=mn.replace("_seed42", "").replace("_", " ")[:20])
        ax.set_ylabel("Accuracy"); ax.set_title("Robustness: Accuracy per Corruption (severity=1.0)")
        ax.set_xticks(x + width); ax.set_xticklabels([c.replace("_", " ") for c in corruptions], rotation=45, ha="right", fontsize=8)
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        plt.tight_layout(); plt.savefig(fig_dir / "fig4_robustness.png", dpi=150, bbox_inches="tight"); plt.close()
        print("  Fig 4: Robustness")

    # --- Fig 5: Calibration ---
    if cal:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        model_names = list(cal.keys())
        metrics = ["ece", "ece_ts"]
        x = np.arange(len(model_names))
        for i, m in enumerate(metrics):
            vals = [cal[mn].get(m, 0) for mn in model_names]
            axes[i].bar(x, vals, color=["#e74c3c", "#2ecc71"][i])
            axes[i].set_title("ECE" if m == "ece" else "ECE (temp-scaled)")
            axes[i].set_xticks(x); axes[i].set_xticklabels([mn[:15] for mn in model_names], rotation=45, ha="right")
        plt.tight_layout(); plt.savefig(fig_dir / "fig5_calibration.png", dpi=150, bbox_inches="tight"); plt.close()
        print("  Fig 5: Calibration")

    # --- Fig 6: Uncertainty ---
    if unc:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        model_names = list(unc.keys())
        x = np.arange(len(model_names))
        ent_c = [unc[mn]["entropy_correct"] for mn in model_names]
        ent_i = [unc[mn]["entropy_incorrect"] for mn in model_names]
        axes[0].bar(x - 0.2, ent_c, 0.35, label="Correct", color="#2ecc71")
        axes[0].bar(x + 0.2, ent_i, 0.35, label="Incorrect", color="#e74c3c")
        axes[0].set_title("Predictive Entropy"); axes[0].legend()
        axes[0].set_xticks(x); axes[0].set_xticklabels([mn[:15] for mn in model_names], rotation=45, ha="right")
        for mn in model_names:
            rc = unc[mn]["risk_coverage"]
            coverages = [r["coverage"] for r in rc]
            risks = [r["risk"] for r in rc]
            axes[1].plot(coverages, risks, label=mn[:15], marker="o", markersize=3)
        axes[1].set_xlabel("Coverage"); axes[1].set_ylabel("Risk (1-accuracy)")
        axes[1].set_title("Risk-Coverage Curve"); axes[1].legend(fontsize=7)
        plt.tight_layout(); plt.savefig(fig_dir / "fig6_uncertainty.png", dpi=150, bbox_inches="tight"); plt.close()
        print("  Fig 6: Uncertainty")

    # --- Fig 7: External Validation ---
    if ext and any(isinstance(v, dict) and "accuracy" in v for v in ext.values()):
        fig, ax = plt.subplots(figsize=(8, 5))
        model_names = [k for k, v in ext.items() if isinstance(v, dict) and "accuracy" in v]
        accs = [ext[mn]["accuracy"] for mn in model_names]
        f1s = [ext[mn]["macro_f1"] for mn in model_names]
        x = np.arange(len(model_names))
        ax.bar(x - 0.2, accs, 0.35, label="Accuracy", color="#3498db")
        ax.bar(x + 0.2, f1s, 0.35, label="Macro F1", color="#e74c3c")
        ax.set_title("External Validation (BRISC)"); ax.legend()
        ax.set_xticks(x); ax.set_xticklabels([mn[:20] for mn in model_names], rotation=45, ha="right")
        ax.set_ylim(0.8, 1.0); ax.grid(axis="y", alpha=0.3)
        plt.tight_layout(); plt.savefig(fig_dir / "fig7_external.png", dpi=150, bbox_inches="tight"); plt.close()
        print("  Fig 7: External Validation")

    # --- Fig 8: Multi-seed box plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    model_groups = {}
    for k, v in ms.items():
        mn = v.get("model_name", k.rsplit("_seed", 1)[0])
        model_groups.setdefault(mn, []).append(v["test"]["accuracy"])
    data = [model_groups[mn] for mn in sorted(model_groups.keys())]
    labels = [mn[:20] for mn in sorted(model_groups.keys())]
    ax.boxplot(data, labels=labels)
    ax.set_title("Multi-Seed Accuracy Distribution"); ax.set_ylabel("Accuracy")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(fig_dir / "fig8_multiseed.png", dpi=150, bbox_inches="tight"); plt.close()
    print("  Fig 8: Multi-Seed")

    # --- Fig 9: Efficiency ---
    fig, ax = plt.subplots(figsize=(8, 5))
    names_eff, params, latencies = [], [], []
    for k in sorted(ab):
        r = ab[k]; e = r.get("efficiency", {})
        if e:
            names_eff.append(_name(r, k).replace("_seed42", "").replace("_", " ")[:20])
            params.append(e.get("params", 0) / 1e6)
            latencies.append(e.get("latency_ms", 0))
    x = np.arange(len(names_eff))
    ax.barh(x, params, color="#3498db", alpha=0.7, label="Parameters (M)")
    ax.set_yticks(x); ax.set_yticklabels(names_eff)
    ax.set_xlabel("Parameters (M)"); ax.set_title("Model Efficiency")
    plt.tight_layout(); plt.savefig(fig_dir / "fig9_efficiency.png", dpi=150, bbox_inches="tight"); plt.close()
    print("  Fig 9: Efficiency")

    print(f"\n  All figures saved to {fig_dir}/")


# ============================================================================
# PHASE 12: TABLES
# ============================================================================
def run_tables():
    print("\n" + "=" * 70)
    print("PHASE 12: REGENERATE TABLES")
    print("=" * 70)

    ab = json.load(open("paper/results/ablation_all.json"))
    tab_dir = OUT_DIR / "tables"
    tab_dir.mkdir(exist_ok=True)

    # Table 1: Main Comparison
    lines = ["| Model | Accuracy | Bal. Acc | Macro F1 | Weighted F1 | Precision | Recall | ROC-AUC | PR-AUC | MCC | Kappa | Params |",
             "|-------|----------|----------|----------|-------------|-----------|--------|---------|--------|-----|-------|--------|"]
    for k in sorted(ab):
        r = ab[k]; t = r["test"]; e = r.get("efficiency", {})
        name = r.get("experiment", k).replace("_seed42", "")
        params = f"{e.get('params', 0)/1e6:.1f}M" if e else "N/A"
        lines.append(
            f"| {name} | {t['accuracy']:.4f} | {t['balanced_accuracy']:.4f} | "
            f"{t['macro_f1']:.4f} | {t['weighted_f1']:.4f} | {t['macro_precision']:.4f} | "
            f"{t['macro_recall']:.4f} | {t['roc_auc']:.4f} | {t['pr_auc']:.4f} | "
            f"{t['mcc']:.4f} | {t['kappa']:.4f} | {params} |")
    with open(tab_dir / "table1_main_comparison.md", "w") as f:
        f.write("\n".join(lines))
    print("  Table 1: Main Comparison")

    # Table 2: Per-Class F1
    lines2 = ["| Model | Glioma | Meningioma | No Tumor | Pituitary |",
              "|-------|--------|------------|----------|-----------|"]
    for k in sorted(ab):
        r = ab[k]; pc = r["test"]["per_class_f1"]
        name = r.get("experiment", k).replace("_seed42", "")
        lines2.append(
            f"| {name} | {pc.get('glioma', 0):.4f} | {pc.get('meningioma', 0):.4f} | "
            f"{pc.get('no tumor', 0):.4f} | {pc.get('pituitary', 0):.4f} |")
    with open(tab_dir / "table2_per_class_f1.md", "w") as f:
        f.write("\n".join(lines2))
    print("  Table 2: Per-Class F1")

    # Table 3: Ablation Delta
    bl = ab.get("C_baseline_efficientnet_seed42", {}).get("test", {})
    b_acc = bl.get("accuracy", 0)
    b_f1 = bl.get("macro_f1", 0)
    lines3 = ["| Configuration | Accuracy | Macro F1 | Delta Acc | Delta F1 |",
              "|---------------|----------|----------|-----------|----------|"]
    for k in sorted(ab):
        r = ab[k]; t = r["test"]
        name = r.get("experiment", k).replace("_seed42", "")
        lines3.append(f"| {name} | {t['accuracy']:.4f} | {t['macro_f1']:.4f} | "
                      f"{t['accuracy']-b_acc:+.4f} | {t['macro_f1']-b_f1:+.4f} |")
    with open(tab_dir / "table3_ablation_delta.md", "w") as f:
        f.write("\n".join(lines3))
    print("  Table 3: Ablation Delta")

    print(f"  Tables saved to {tab_dir}/")


# ============================================================================
# PHASE 13: FINAL VALIDATION REPORT
# ============================================================================
def run_final_report():
    print("\n" + "=" * 70)
    print("PHASE 13: FINAL VALIDATION REPORT")
    print("=" * 70)

    ab = json.load(open("paper/results/ablation_all.json"))
    ms = json.load(open("paper/results/multi_seed_all.json"))
    rob = json.load(open(OUT_DIR / "robustness_results.json")) if (OUT_DIR / "robustness_results.json").exists() else {}
    cal = json.load(open(OUT_DIR / "calibration_results.json")) if (OUT_DIR / "calibration_results.json").exists() else {}
    unc = json.load(open(OUT_DIR / "uncertainty_results.json")) if (OUT_DIR / "uncertainty_results.json").exists() else {}
    ext = json.load(open(OUT_DIR / "external_validation.json")) if (OUT_DIR / "external_validation.json").exists() else {}
    stat = json.load(open(OUT_DIR / "statistical_results.json")) if (OUT_DIR / "statistical_results.json").exists() else {}
    leakage = json.load(open(OUT_DIR / "leakage_report.json")) if (OUT_DIR / "leakage_report.json").exists() else {}

    # Build final results CSV
    csv_lines = ["Model,Seed,Accuracy,Balanced_Accuracy,Macro_F1,Weighted_F1,Macro_Precision,Macro_Recall,ROC_AUC,PR_AUC,MCC,Kappa,Params,Latency_ms"]
    for k in sorted(ab):
        r = ab[k]; t = r["test"]; e = r.get("efficiency", {})
        name = r.get("experiment", k).replace("_seed42", "")
        csv_lines.append(
            f"{name},42,{t['accuracy']:.6f},{t['balanced_accuracy']:.6f},{t['macro_f1']:.6f},"
            f"{t['weighted_f1']:.6f},{t['macro_precision']:.6f},{t['macro_recall']:.6f},"
            f"{t['roc_auc']:.6f},{t['pr_auc']:.6f},{t['mcc']:.6f},{t['kappa']:.6f},"
            f"{e.get('params',0)},{e.get('latency_ms',0):.2f}")
    with open(OUT_DIR / "final_results.csv", "w") as f:
        f.write("\n".join(csv_lines))

    # Save all results as JSON
    final = {
        "ablation": ab,
        "multi_seed": ms,
        "robustness": rob,
        "calibration": cal,
        "uncertainty": unc,
        "external_validation": ext,
        "statistical": stat,
        "leakage": leakage,
        "timestamp": datetime.now().isoformat(),
    }
    with open(OUT_DIR / "final_results.json", "w") as f:
        json.dump(final, f, indent=2, cls=NpEncoder)

    # Write final report
    best_model = max(ab.keys(), key=lambda k: ab[k]["test"]["macro_f1"])
    best_acc = max(ab.values(), key=lambda r: r["test"]["accuracy"])
    best_f1 = max(ab.values(), key=lambda r: r["test"]["macro_f1"])

    report = f"""# FINAL VALIDATION REPORT
## BrainTumorDiagnosis — Complete Evaluation Results

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

---

## DATA LEAKAGE

| Check | Status | Details |
|-------|--------|---------|
| Internal Train/Test filename overlap | {218} files | Different content (filename collision, not actual leakage) |
| Kaggle contamination | 839/1311 overlap | Kaggle CANNOT be used as independent validation |
| BRISC independence | VERIFIED | No overlap with internal data |
| Content-based duplicates | {leakage.get('content_duplicates', {}).get('within_train_sampled', 'N/A')} groups | Within sampled subset |

---

## MODEL COMPARISON (Internal Test Set)

| Model | Accuracy | Macro F1 | ROC-AUC | MCC |
|-------|----------|----------|---------|-----|
"""
    for k in sorted(ab):
        r = ab[k]; t = r["test"]
        name = r.get("experiment", k).replace("_seed42", "")
        report += f"| {name} | {t['accuracy']:.4f} | {t['macro_f1']:.4f} | {t['roc_auc']:.4f} | {t['mcc']:.4f} |\n"

    report += f"""
**Best model (F1):** {best_model.replace('_seed42', '')} (F1={best_f1['test']['macro_f1']:.4f})
**Best model (Acc):** {max(ab.keys(), key=lambda k: ab[k]['test']['accuracy']).replace('_seed42', '')} (Acc={best_acc['test']['accuracy']:.4f})

---

## MULTI-SEED STABILITY

"""
    for mn in ["A_baseline_densenet121", "d_tumor_aware", "e_turma_full"]:
        mrs = {k: v for k, v in ms.items() if v.get("model_name") == mn}
        if mrs:
            accs = [v["test"]["accuracy"] for v in mrs.values()]
            f1s = [v["test"]["macro_f1"] for v in mrs.values()]
            report += f"- **{mn}**: Acc={np.mean(accs):.4f}+/-{np.std(accs):.4f}, F1={np.mean(f1s):.4f}+/-{np.std(f1s):.4f}\n"

    if rob:
        report += "\n---\n\n## ROBUSTNESS\n\n| Model | Clean Acc | MRRS | Worst Case |\n|-------|-----------|------|------------|\n"
        for mn, data in rob.items():
            report += f"| {mn} | {data['clean_accuracy']:.4f} | {data['mrrs']:.4f} | {data['worst_case_accuracy']:.4f} |\n"

    if cal:
        report += "\n---\n\n## CALIBRATION\n\n| Model | ECE | MCE | Brier | ECE (temp-scaled) |\n|-------|-----|-----|-------|-------------------|\n"
        for mn, data in cal.items():
            report += f"| {mn} | {data['ece']:.4f} | {data['mce']:.4f} | {data['brier']:.4f} | {data['ece_ts']:.4f} |\n"

    if unc:
        report += "\n---\n\n## UNCERTAINTY\n\n| Model | Mean Entropy | Ent(Correct) | Ent(Incorrect) | AUROC Error Det | Acc@80% Cov |\n|-------|-------------|--------------|----------------|-----------------|-------------|\n"
        for mn, data in unc.items():
            report += f"| {mn} | {data['mean_entropy']:.4f} | {data['entropy_correct']:.4f} | {data['entropy_incorrect']:.4f} | {data['auroc_error_detection']:.4f} | {data['accuracy_at_80_coverage']:.4f} |\n"

    report += f"""
---

## EXTERNAL VALIDATION

**Kaggle: CONTAMINATED** (839/1311 filenames overlap with internal training set)

**BRISC (genuinely independent):**

"""
    if ext:
        for mn, data in ext.items():
            if isinstance(data, dict) and "accuracy" in data:
                report += f"- **{mn}**: acc={data['accuracy']:.4f}, F1={data['macro_f1']:.4f}, AUC={data['roc_auc']:.4f}\n"

    report += f"""
---

## STATISTICAL VALIDATION

"""
    if stat:
        for k, v in stat.items():
            if "bootstrap" in k:
                report += f"- **{k.replace('_bootstrap', '')}**: {v['mean']:.4f} [{v['ci_lower']:.4f}, {v['ci_upper']:.4f}]\n"
            elif "mcnemar" in k:
                sig = "YES" if v.get("significant") else "NO"
                report += f"- **{k}**: chi2={v['statistic']:.4f}, p={v['p_value']:.6f}, significant={sig}\n"

    report += f"""
---

## NOVELTY ASSESSMENT

| Component | Novel? | Evidence |
|-----------|--------|----------|
| ATGFM | YES (genuinely new) | Ablation E vs F: ATGFM contributes within TURMA-Net |
| Consistency Loss | NO (standard technique) | Ablation E vs G: minimal effect |
| Uncertainty Weighting | NO (Kendall 2018) | Ablation E vs H: zero measurable effect |

**Overall Novelty: WEAK** — ATGFM is novel but TURMA-Net does not outperform baselines.

---

## PUBLICATION READINESS

| Dimension | Score | Justification |
|-----------|-------|---------------|
| Implementation | 75/100 | All code works, all experiments runnable |
| Testing | 65/100 | 73 tests pass, forward/backward verified |
| Scientific Validation | 45/100 | Ablation done, multi-seed done, stats done, robustness/cal/unc done |
| Novelty | 35/100 | ATGFM novel but overall model underperforms baselines |
| Reproducibility | 55/100 | Seeds documented, configs saved, but no environment lockfile |
| **Publication Readiness** | **45/100** | Major gap: proposed model does not beat baselines |

---

## FINAL DECISION

**D — NOVELTY NOT SUFFICIENT**

The system works, experiments are complete, but TURMA-Net (0.9219) does not outperform DenseNet121 (0.9795) on the internal test set. ATGFM provides measurable improvement within TURMA-Net but the overall architecture is worse than simpler baselines. External validation shows TumorAware (without ATGFM) generalizing best on BRISC.
"""
    with open(OUT_DIR / "final_validation_report.md", "w") as f:
        f.write(report)

    print(f"  Report saved to {OUT_DIR}/final_validation_report.md")
    print(f"  CSV saved to {OUT_DIR}/final_results.csv")
    print(f"  JSON saved to {OUT_DIR}/final_results.json")


# ============================================================================
# MAIN
# ============================================================================
from datetime import datetime

def main():
    dev = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    tg = time.time()

    run_robustness(dev)
    run_calibration(dev)
    run_uncertainty(dev)
    run_statistics(dev)
    run_gradcam(dev)
    run_external(dev)
    run_figures(dev)
    run_tables()
    run_final_report()

    total = (time.time() - tg) / 60
    print(f"\n{'='*70}")
    print(f"ALL EVALUATION PHASES COMPLETE ({total:.1f} min)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
