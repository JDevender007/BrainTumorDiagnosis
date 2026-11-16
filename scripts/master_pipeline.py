"""
MASTER EXPERIMENT PIPELINE - Fair Comparison, All Models, 3 Seeds, 30 Epochs
================================================================================
Trains: DenseNet121, ResNet50, EfficientNetV2-S, TumorAwareMultiTaskNet, TURMA-Net
Then runs: Ablation, Robustness, Calibration, Uncertainty, Grad-CAM, External Val
"""
import sys, json, time, copy, hashlib, warnings
from pathlib import Path
from datetime import datetime

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

sys.path.insert(0, ".")
from utils.seed import set_seed

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# ============================================================================
# CONFIGURATION
# ============================================================================
BS = 48
ACCUM = 2
EPOCHS = 30
PATIENCE = 7
LR = 1e-4
WEIGHT_DECAY = 1e-4
SEEDS = [42, 123, 456]
CLASS_NAMES = ["glioma", "meningioma", "no tumor", "pituitary"]
NUM_CLASSES = 4
RESULTS_DIR = Path("paper/results/final")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NpEncoder = type("NpEncoder", (json.JSONEncoder,), {
    "default": lambda self, o: o.tolist() if isinstance(o, np.ndarray) else float(o) if isinstance(o, (np.float32, np.float64)) else int(o) if isinstance(o, (np.int32, np.int64)) else super().default(o)
})

# ============================================================================
# DATASET (full-path based, no filename collision issues)
# ============================================================================
CLASS_MAP = {"glioma": 0, "meningioma": 1, "no tumor": 2, "pituitary": 3,
             "notumor": 2, "no_tumor": 2}

def norm_name(n):
    n = n.lower().strip()
    return CLASS_MAP.get(n, -1)

class FairClsDS(Dataset):
    """Uses full paths, handles duplicate filenames correctly."""
    def __init__(self, root, split, tf):
        self.samples = []
        self.labels = []
        d = Path(root) / split
        if not d.exists():
            d = Path(root) / split.capitalize()
        for cd in sorted(d.iterdir()):
            if not cd.is_dir():
                continue
            lb = norm_name(cd.name)
            if lb < 0:
                continue
            for p in sorted(cd.iterdir()):
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
                    self.samples.append(str(p))
                    self.labels.append(lb)
        self.tf = tf

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img = Image.open(self.samples[i]).convert("RGB")
        return self.tf(img), self.labels[i]


def load_data(seed=42, val_ratio=0.15):
    """Load data with stratified split. Uses 15% of train for validation."""
    tf_tr = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])
    tf_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    full_train = FairClsDS("data/raw/classification", "Train", tf_val)
    test_ds = FairClsDS("data/raw/classification", "Test", tf_val)

    # Stratified split
    set_seed(seed)
    labels = np.array(full_train.labels)
    indices = np.arange(len(full_train))
    np.random.seed(seed)
    np.random.shuffle(indices)

    from collections import defaultdict
    class_indices = defaultdict(list)
    for i in indices:
        class_indices[labels[i]].append(i)

    train_idx, val_idx = [], []
    for c, idxs in class_indices.items():
        n_val = max(1, int(len(idxs) * val_ratio))
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])

    train_ds = FairClsDS("data/raw/classification", "Train", tf_tr)
    val_ds = Subset(FairClsDS("data/raw/classification", "Train", tf_val), val_idx)

    kw = dict(num_workers=0, pin_memory=True)
    return (DataLoader(train_ds, BS, shuffle=True, drop_last=True, **kw),
            DataLoader(val_ds, BS * 2, **kw),
            DataLoader(test_ds, BS * 2, **kw),
            len(train_idx), len(val_idx), len(test_ds))


def load_external(root, batch_size=64):
    """Load external dataset."""
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = FairClsDS(root, ".", tf)
    # Try common splits
    if len(ds) == 0:
        for split in ["test", "Test", "Testing", "val", "validation"]:
            ds = FairClsDS(root, split, tf)
            if len(ds) > 0:
                break
    if len(ds) == 0:
        # Flat directory
        samples, labels = [], []
        root_path = Path(root)
        for cd in sorted(root_path.iterdir()):
            if not cd.is_dir():
                continue
            lb = norm_name(cd.name)
            if lb < 0:
                continue
            for p in sorted(cd.iterdir()):
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
                    samples.append(str(p))
                    labels.append(lb)
        class ExtDS(Dataset):
            def __init__(s, sp, lb, t):
                s.sp, s.lb, s.t = sp, lb, t
            def __len__(s):
                return len(s.sp)
            def __getitem__(s, i):
                return s.t(Image.open(s.sp[i]).convert("RGB")), s.lb[i]
        ds = ExtDS(samples, labels, tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)


# ============================================================================
# MODEL BUILDERS
# ============================================================================
def build_classifier(name, seed=42):
    """Build a baseline classifier."""
    set_seed(seed)
    from models.factory import create_backbone_classifier
    bn = {"dense": "densenet121", "resnet": "resnet50", "efficientnet": "efficientnet_v2_s"}[name]
    return create_backbone_classifier(bn, num_classes=NUM_CLASSES, pretrained=True)

def build_tumor_aware(seed=42):
    """Build TumorAwareMultiTaskNet."""
    set_seed(seed)
    from models.tumor_aware import TumorAwareMultiTaskNet, TumorAwareConfig
    return TumorAwareMultiTaskNet(TumorAwareConfig(
        backbone="densenet121", num_classes=NUM_CLASSES,
        segmentation_classes=NUM_CLASSES, dropout=0.3, fusion_dim=256,
        use_segmentation_head=True, use_uncertainty_head=True))

def build_turma(seed=42, use_atgfm=True, use_consistency=True, use_uncertainty=True):
    """Build TURMA-Net with configurable components."""
    set_seed(seed)
    from models.novel.turma_net import TurmaNet, TurmaNetConfig
    return TurmaNet(TurmaNetConfig(
        backbone="densenet121", num_classes=NUM_CLASSES,
        segmentation_classes=NUM_CLASSES, dropout=0.3, fusion_dim=256,
        use_segmentation_head=True, use_uncertainty_head=True,
        use_atgfm=use_atgfm, use_consistency=use_consistency,
        use_uncertainty_weights=use_uncertainty))

def is_turma(model):
    return type(model).__name__ == "TurmaNet"

def is_multitask(model):
    return type(model).__name__ in ("TurmaNet", "TumorAwareMultiTaskNet")


# ============================================================================
# TRAINING
# ============================================================================
def train_epoch(model, loader, criterion, optimizer, scaler, dev, use_mt=False):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    optimizer.zero_grad()
    for bi, (img, lab) in enumerate(loader):
        img = img.to(dev, non_blocking=True)
        lab = lab.to(dev, non_blocking=True)
        with torch.amp.autocast("cuda"):
            if use_mt:
                out = model(img, return_seg=True)
                logits = out["logits"] if isinstance(out, dict) else out
                if isinstance(criterion, tuple):
                    loss = criterion[0](logits, lab)
                else:
                    loss = criterion(logits, lab)
            else:
                logits = model(img)
                if isinstance(logits, dict):
                    logits = logits["logits"]
                loss = criterion(logits, lab)
            loss = loss / ACCUM
        scaler.scale(loss).backward()
        if (bi + 1) % ACCUM == 0 or (bi + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        total_loss += loss.item() * ACCUM
        correct += (logits.argmax(1) == lab).sum().item()
        total += lab.size(0)
    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, dev, use_mt=False):
    model.eval()
    all_probs, all_labels = [], []
    for img, lab in loader:
        img = img.to(dev, non_blocking=True)
        if use_mt:
            out = model(img, return_seg=True)
            logits = out["logits"] if isinstance(out, dict) else out
        else:
            out = model(img)
            logits = out["logits"] if isinstance(out, dict) else out
        probs = F.softmax(logits, dim=1)
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(lab.numpy())
    yt = np.array(all_labels)
    yp = np.array(all_probs)
    pred = yp.argmax(axis=1)
    cm = confusion_matrix(yt, pred, labels=range(NUM_CLASSES))
    try:
        yt_b = label_binarize(yt, classes=range(NUM_CLASSES))
        ra = roc_auc_score(yt_b, yp, multi_class="ovr", average="macro")
        pa = average_precision_score(yt_b, yp, average="macro")
    except:
        ra, pa = 0.0, 0.0
    pc_f1 = {}
    pc_prec = {}
    pc_rec = {}
    for i in range(NUM_CLASSES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-10)
        pc_f1[CLASS_NAMES[i]] = float(f1)
        pc_prec[CLASS_NAMES[i]] = float(p)
        pc_rec[CLASS_NAMES[i]] = float(r)
    roc_fpr, roc_tpr, roc_auc = {}, {}, {}
    pr_prec, pr_rec, pr_auc = {}, {}, {}
    for i in range(NUM_CLASSES):
        fpr_, tpr_, _ = roc_curve(yt_b[:, i], yp[:, i])
        roc_fpr[str(i)] = fpr_
        roc_tpr[str(i)] = tpr_
        roc_auc[CLASS_NAMES[i]] = float(sklearn_auc(fpr_, tpr_))
        prec_, rec_, _ = precision_recall_curve(yt_b[:, i], yp[:, i])
        pr_prec[str(i)] = prec_
        pr_rec[str(i)] = rec_
        pr_auc[CLASS_NAMES[i]] = float(sklearn_auc(rec_, prec_))
    sens = {CLASS_NAMES[i]: float(cm[i, i] / max(cm[i, :].sum(), 1)) for i in range(NUM_CLASSES)}
    spec = {}
    for i in range(NUM_CLASSES):
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        spec[CLASS_NAMES[i]] = float(tn / max(tn + fp, 1))
    return {
        "accuracy": float(accuracy_score(yt, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, pred)),
        "macro_f1": float(f1_score(yt, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(yt, pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(yt, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(yt, pred, average="macro", zero_division=0)),
        "roc_auc": float(ra),
        "pr_auc": float(pa),
        "mcc": float(matthews_corrcoef(yt, pred)),
        "kappa": float(cohen_kappa_score(yt, pred)),
        "per_class_f1": pc_f1,
        "per_class_precision": pc_prec,
        "per_class_recall": pc_rec,
        "sensitivity": sens,
        "specificity": spec,
        "confusion_matrix": cm.tolist(),
        "roc_data": {"fpr": roc_fpr, "tpr": roc_tpr, "auc": roc_auc},
        "pr_data": {"precision": pr_prec, "recall": pr_rec, "auc": pr_auc},
        "y_true": yt.tolist(),
        "y_pred": pred.tolist(),
        "y_probs": yp.tolist(),
    }


def measure_efficiency(model, dev, n=30):
    model.eval()
    d = torch.randn(1, 3, 224, 224, device=dev)
    use_mt = is_multitask(model)
    for _ in range(10):
        with torch.no_grad():
            if use_mt:
                model(d, return_seg=True)
            else:
                model(d)
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            if use_mt:
                model(d, return_seg=True)
            else:
                model(d)
        torch.cuda.synchronize()
        ts.append((time.time() - t0) * 1000)
    params = sum(p.numel() for p in model.parameters())
    sz = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 ** 2
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        if use_mt:
            model(d, return_seg=True)
        else:
            model(d)
    pk = torch.cuda.max_memory_allocated() / 1024 ** 2
    return {
        "params": int(params),
        "size_mb": round(sz, 2),
        "latency_ms": round(np.mean(ts), 2),
        "latency_std": round(np.std(ts), 2),
        "fps": round(1000 / np.mean(ts), 2),
        "peak_vram_mb": round(pk, 2),
    }


def train_model(model, name, tr, va, te, dev, seed, use_mt=False):
    """Full training loop with early stopping."""
    od = RESULTS_DIR / f"{name}_seed{seed}"
    od.mkdir(parents=True, exist_ok=True)

    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda")

    if use_mt and is_turma(model):
        from models.novel.losses import TurmaNetLoss
        criterion = TurmaNetLoss(num_classes=NUM_CLASSES, use_uncertainty_weights=True)
        # For TURMA, we need a simpler cls loss for the basic training loop
        cls_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    elif use_mt:
        cls_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        criterion = cls_criterion
    else:
        cls_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        criterion = cls_criterion

    best_f1, best_acc, best_ep, pat = 0.0, 0.0, 0, 0
    hist = []
    t0 = time.time()

    for ep in range(EPOCHS):
        ep_start = time.time()

        # Training
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        opt.zero_grad()
        for bi, (img, lab) in enumerate(tr):
            img = img.to(dev, non_blocking=True)
            lab = lab.to(dev, non_blocking=True)
            with torch.amp.autocast("cuda"):
                if use_mt:
                    out = model(img, return_seg=True)
                    logits = out["logits"] if isinstance(out, dict) else out
                    if is_turma(model):
                        seg_logits = out.get("seg_logits", None)
                        if seg_logits is not None:
                            total_loss_t, comps = criterion(logits, lab, seg_logits, None)
                            loss = total_loss_t / ACCUM
                        else:
                            loss = cls_criterion(logits, lab) / ACCUM
                    else:
                        loss = cls_criterion(logits, lab) / ACCUM
                else:
                    logits = model(img)
                    if isinstance(logits, dict):
                        logits = logits["logits"]
                    loss = criterion(logits, lab) / ACCUM

            scaler.scale(loss).backward()
            if (bi + 1) % ACCUM == 0 or (bi + 1) == len(tr):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()

            total_loss += loss.item() * ACCUM
            correct += (logits.argmax(1) == lab).sum().item()
            total += lab.size(0)

        sched.step()
        train_loss = total_loss / len(tr)
        train_acc = correct / total

        # Validation
        val_metrics = evaluate(model, va, dev, use_mt=use_mt)
        vf1 = val_metrics["macro_f1"]
        vacc = val_metrics["accuracy"]
        es = time.time() - ep_start
        ts_total = time.time() - t0
        m, s = divmod(int(ts_total), 60)
        h, m = divmod(m, 60)

        mk = ""
        if vf1 > best_f1:
            best_f1, best_acc, best_ep, pat = vf1, vacc, ep + 1, 0
            mk = " *BEST*"
            torch.save({
                "epoch": ep + 1,
                "model_state_dict": model.state_dict(),
                "val_f1": vf1,
                "val_acc": vacc,
                "seed": seed,
                "config": {"name": name, "epochs": EPOCHS, "lr": LR, "bs": BS},
            }, od / "best.pt")
        else:
            pat += 1

        hist.append({
            "epoch": ep + 1, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": 0, "val_f1": vf1, "val_acc": vacc, "time": es,
        })

        print(f"    Ep {ep+1:2d}/{EPOCHS} loss={train_loss:.4f} tr_acc={train_acc:.4f} "
              f"vf1={vf1:.4f} vacc={vacc:.4f} {es:.0f}s {h}h{m:02d}m{s:02d}s{mk}", flush=True)

        if pat >= PATIENCE:
            print(f"    Early stopping at epoch {ep+1}")
            break

    # Save last
    torch.save({"epoch": hist[-1]["epoch"], "model_state_dict": model.state_dict()}, od / "last.pt")
    with open(od / "training_history.json", "w") as f:
        json.dump(hist, f, indent=2)

    # Evaluate on test
    ckpt = torch.load(od / "best.pt", map_location=dev, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = evaluate(model, te, dev, use_mt=use_mt)
    eff = measure_efficiency(model, dev)

    entry = {
        "experiment": name,
        "seed": seed,
        "params": eff["params"],
        "train": {"best_val_f1": best_f1, "best_val_acc": best_acc, "best_epoch": best_ep, "total_time": time.time() - t0},
        "test": {k: test_metrics[k] for k in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
                "macro_precision", "macro_recall", "roc_auc", "pr_auc", "mcc", "kappa",
                "per_class_f1", "per_class_precision", "per_class_recall", "sensitivity", "specificity"]},
        "efficiency": eff,
        "confusion_matrix": test_metrics["confusion_matrix"],
        "roc_data": test_metrics["roc_data"],
        "pr_data": test_metrics["pr_data"],
        "y_true": test_metrics["y_true"],
        "y_pred": test_metrics["y_pred"],
        "y_probs": test_metrics["y_probs"],
        "history": hist,
    }

    with open(od / "result.json", "w") as f:
        json.dump(entry, f, indent=2, cls=NpEncoder)

    total_time = time.time() - t0
    print(f"  TEST: acc={test_metrics['accuracy']:.4f} f1={test_metrics['macro_f1']:.4f} "
          f"auc={test_metrics['roc_auc']:.4f} mcc={test_metrics['mcc']:.4f}")
    print(f"  Time: {total_time/60:.1f} min")
    return entry


# ============================================================================
# MAIN
# ============================================================================
def main():
    dev = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Config: BS={BS} ACCUM={ACCUM} EPOCHS={EPOCHS} PATIENCE={PATIENCE} LR={LR}")
    print(f"Seeds: {SEEDS}")
    tg = time.time()

    all_results = {}

    # ========================================================================
    # PHASE 1: MAIN TRAINING (5 models x 3 seeds)
    # ========================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: MAIN TRAINING - 5 models x 3 seeds x 30 epochs")
    print("=" * 70)

    MODEL_BUILDERS = [
        ("DenseNet121", lambda s: build_classifier("dense", s), False),
        ("ResNet50", lambda s: build_classifier("resnet", s), False),
        ("EfficientNetV2-S", lambda s: build_classifier("efficientnet", s), False),
        ("TumorAwareMultiTaskNet", lambda s: build_tumor_aware(s), True),
        ("TURMA-Net", lambda s: build_turma(s), True),
    ]

    for model_name, builder, use_mt in MODEL_BUILDERS:
        for seed in SEEDS:
            key = f"{model_name}_seed{seed}"
            if key in all_results:
                print(f"  [SKIP] {key}")
                continue

            # Load fresh data for each run
            tr, va, te, n_tr, n_val, n_te = load_data(seed=seed)
            print(f"\n{'='*70}")
            print(f"  {model_name} | seed={seed} | train={n_tr} val={n_val} test={n_te}")
            print(f"{'='*70}")

            model = builder(seed)
            params = sum(p.numel() for p in model.parameters())
            print(f"  Params: {params/1e6:.2f}M")

            entry = train_model(model, key, tr, va, te, dev, seed, use_mt=use_mt)
            all_results[key] = entry

            # Save master results
            with open(RESULTS_DIR / "all_results.json", "w") as f:
                json.dump(all_results, f, indent=2, cls=NpEncoder)

            print(f"  [{key}] DONE - acc={entry['test']['accuracy']:.4f}")

    # ========================================================================
    # PHASE 2: ABLATION (8 configs x 1 seed)
    # ========================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: ABLATION STUDY")
    print("=" * 70)

    ABLATION_CONFIGS = [
        ("A_DenseNet121", lambda s: build_classifier("dense", s), False, {}),
        ("B_ResNet50", lambda s: build_classifier("resnet", s), False, {}),
        ("C_EfficientNetV2-S", lambda s: build_classifier("efficientnet", s), False, {}),
        ("D_TumorAware", lambda s: build_tumor_aware(s), True, {}),
        ("E_TURMA_Full", lambda s: build_turma(s, True, True, True), True, {"atgfm": True, "consistency": True, "uncertainty": True}),
        ("F_TURMA_no_ATGFM", lambda s: build_turma(s, False, True, True), True, {"atgfm": False, "consistency": True, "uncertainty": True}),
        ("G_TURMA_no_Consistency", lambda s: build_turma(s, True, False, True), True, {"atgfm": True, "consistency": False, "uncertainty": True}),
        ("H_TURMA_no_Uncertainty", lambda s: build_turma(s, True, True, False), True, {"atgfm": True, "consistency": True, "uncertainty": False}),
    ]

    ablation_seed = 42
    tr_a, va_a, te_a, _, _, _ = load_data(seed=ablation_seed)

    for abl_name, builder, use_mt, desc in ABLATION_CONFIGS:
        key = f"{abl_name}_seed{ablation_seed}"
        if key in all_results:
            print(f"  [SKIP] {key}")
            continue

        print(f"\n  --- {abl_name} ({desc}) ---")
        model = builder(ablation_seed)
        entry = train_model(model, key, tr_a, va_a, te_a, dev, ablation_seed, use_mt=use_mt)
        all_results[key] = entry
        with open(RESULTS_DIR / "all_results.json", "w") as f:
            json.dump(all_results, f, indent=2, cls=NpEncoder)

    # ========================================================================
    # PHASE 3: EXTERNAL VALIDATION (BRISC only - Kaggle is contaminated)
    # ========================================================================
    print("\n" + "=" * 70)
    print("PHASE 3: EXTERNAL VALIDATION (BRISC only)")
    print("=" * 70)

    ext_results = {}
    brisc_path = "data/external/brisc2025/brisc2025/classification_task/test"
    if Path(brisc_path).exists():
        ext_loader = load_external(brisc_path)
        print(f"  BRISC: {len(ext_loader.dataset)} samples")

        for model_name, builder, use_mt in [("DenseNet121", lambda s: build_classifier("dense", s), False),
                                              ("TumorAwareMultiTaskNet", lambda s: build_tumor_aware(s), True),
                                              ("TURMA-Net", lambda s: build_turma(s), True)]:
            key = f"{model_name}_seed42"
            ck = RESULTS_DIR / key / "best.pt"
            if not ck.exists():
                print(f"  {model_name}: NO CHECKPOINT"); continue
            model = builder(42).to(dev)
            ckpt = torch.load(ck, map_location=dev, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            er = evaluate(model, ext_loader, dev, use_mt=use_mt)
            ext_results[f"{model_name}_BRISC"] = {
                "model": model_name, "dataset": "BRISC",
                "accuracy": er["accuracy"], "macro_f1": er["macro_f1"],
                "roc_auc": er["roc_auc"], "mcc": er["mcc"],
                "per_class_f1": er["per_class_f1"],
                "confusion_matrix": er["confusion_matrix"],
            }
            print(f"  {model_name:30s} BRISC acc={er['accuracy']:.4f} f1={er['macro_f1']:.4f} auc={er['roc_auc']:.4f}")

    with open(RESULTS_DIR / "external_validation.json", "w") as f:
        json.dump(ext_results, f, indent=2, cls=NpEncoder)

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("ALL PHASES COMPLETE")
    print("=" * 70)
    total_min = (time.time() - tg) / 60
    print(f"Total time: {total_min:.1f} min ({total_min/60:.1f} hours)")
    print(f"\n{'Model':45s} {'Acc':>8s} {'F1':>8s} {'AUC':>8s} {'MCC':>8s}")
    print("-" * 81)
    for k in sorted(all_results):
        r = all_results[k]
        t = r["test"]
        print(f"  {k:43s} {t['accuracy']:.4f}  {t['macro_f1']:.4f}  {t['roc_auc']:.4f}  {t['mcc']:.4f}")

    if ext_results:
        print(f"\nExternal Validation:")
        for k, v in ext_results.items():
            print(f"  {k}: acc={v['accuracy']:.4f} f1={v['macro_f1']:.4f}")

    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
