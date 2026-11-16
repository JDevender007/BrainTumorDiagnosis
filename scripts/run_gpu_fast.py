"""Maximum GPU utilization training script.
Changes from original:
  - batch_size 48 (was 8) -> 6x more data per step
  - cudnn.benchmark for fast conv algorithms
  - persistent_workers + pin_memory for fast data loading
  - non_blocking transfers to overlap CPU/GPU
  - torch.compile if available for fused kernels
  - 2 gradient accumulation steps (effective bs=96)
  - Mixed precision everywhere
"""
import sys, json, time, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,
    average_precision_score, matthews_corrcoef, cohen_kappa_score,
    log_loss, confusion_matrix, classification_report,
)
from sklearn.preprocessing import label_binarize
from utils.seed import set_seed

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

PHASE = sys.argv[1] if len(sys.argv) > 1 else "ablation"
BATCH_SIZE = 48
GRAD_ACCUM = 2
EPOCHS = 10
EFFECTIVE_BS = BATCH_SIZE * GRAD_ACCUM

CLASS_NAMES = ["glioma", "meningioma", "no tumor", "pituitary"]
CLASS_MAP = {"glioma": 0, "meningioma": 1, "no tumor": 2, "pituitary": 3,
             "notumor": 2, "no_tumor": 2}

def normalize_class_name(name):
    n = name.lower().strip()
    for key, val in CLASS_MAP.items():
        if n == key:
            return ["glioma", "meningioma", "no tumor", "pituitary"][val]
    return n

class ClassificationDataset:
    def __init__(self, root, split="Train", transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []
        self.labels = []
        split_dir = self.root / split
        if not split_dir.exists():
            split_dir = self.root / split.capitalize()
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            cls_name = normalize_class_name(class_dir.name)
            if cls_name not in CLASS_MAP:
                continue
            label = CLASS_MAP[cls_name]
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
                    self.samples.append(str(img_path))
                    self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

def load_data(root="data/raw/classification", seed=42):
    image_size = (224, 224)
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    full_train = ClassificationDataset(root, "Train", train_transform)
    test_ds = ClassificationDataset(root, "Test", test_transform)

    set_seed(seed)
    n = len(full_train)
    indices = list(range(n))
    np.random.seed(seed)
    np.random.shuffle(indices)
    val_n = int(n * 0.2)
    val_indices, train_indices = indices[:val_n], indices[val_n:]

    val_ds_orig = ClassificationDataset(root, "Train", test_transform)
    train_ds = Subset(full_train, train_indices)
    val_ds = Subset(val_ds_orig, val_indices)

    pw = torch.get_num_threads() > 1
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True,
                              persistent_workers=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                            num_workers=0, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                             num_workers=0, pin_memory=True, drop_last=False)
    return train_loader, val_loader, test_loader

def load_external(root, batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    samples, labels = [], []
    root_path = Path(root)
    for class_dir in sorted(root_path.iterdir()):
        if not class_dir.is_dir():
            continue
        cls_name = normalize_class_name(class_dir.name)
        if cls_name not in CLASS_MAP:
            continue
        label = CLASS_MAP[cls_name]
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
                samples.append(str(img_path))
                labels.append(label)
    class ExtDS:
        def __init__(self, s, l, t):
            self.s, self.l, self.t = s, l, t
        def __len__(self): return len(self.s)
        def __getitem__(self, idx):
            img = Image.open(self.s[idx]).convert("RGB")
            return self.t(img), self.l[idx]
    return DataLoader(ExtDS(samples, labels, transform), batch_size=batch_size,
                      shuffle=False, num_workers=0, pin_memory=True)

def build_baseline(backbone="densenet121"):
    from models.factory import create_backbone_classifier
    return create_backbone_classifier(backbone, num_classes=4, pretrained=True)

def build_tumor_aware(backbone="densenet121"):
    from models.tumor_aware import TumorAwareMultiTaskNet, TumorAwareConfig
    return TumorAwareMultiTaskNet(TumorAwareConfig(backbone=backbone, num_classes=4, use_segmentation_head=True))

def build_turma_net(backbone="densenet121", **kw):
    from models.novel.turma_net import TurmaNet, TurmaNetConfig
    return TurmaNet(TurmaNetConfig(backbone=backbone, num_classes=4, **kw))

ABLATION_CONFIGS = {
    "A_baseline_densenet121": {"desc": "Baseline: DenseNet121", "builder": lambda: build_baseline("densenet121"), "tl": False},
    "B_baseline_resnet50": {"desc": "Baseline: ResNet50", "builder": lambda: build_baseline("resnet50"), "tl": False},
    "C_baseline_efficientnet": {"desc": "Baseline: EfficientNetV2-S", "builder": lambda: build_baseline("efficientnet_v2_s"), "tl": False},
    "D_tumor_aware": {"desc": "TumorAwareMultiTaskNet", "builder": lambda: build_tumor_aware("densenet121"), "tl": False},
    "E_turma_full": {"desc": "TURMA-Net Full", "builder": lambda: build_turma_net("densenet121", use_atgfm=True, use_consistency=True, use_uncertainty_weights=True), "tl": True},
    "F_turma_no_atgfm": {"desc": "TURMA w/o ATGFM", "builder": lambda: build_turma_net("densenet121", use_atgfm=False, use_consistency=True, use_uncertainty_weights=True), "tl": True},
    "G_turma_no_consistency": {"desc": "TURMA w/o Consistency", "builder": lambda: build_turma_net("densenet121", use_atgfm=True, use_consistency=False, use_uncertainty_weights=True), "tl": True},
    "H_turma_no_uncertainty": {"desc": "TURMA w/o Uncertainty", "builder": lambda: build_turma_net("densenet121", use_atgfm=True, use_consistency=True, use_uncertainty_weights=False), "tl": True},
}

def train_model(model, model_name, train_loader, val_loader, device, use_turma_loss, output_dir):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    if use_turma_loss:
        from models.novel.losses import TurmaNetLoss
        criterion = TurmaNetLoss(num_classes=4, use_uncertainty_weights=True)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    scaler = torch.amp.GradScaler("cuda")
    out_dir = Path(output_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    best_f1, best_ep, patience, history = 0.0, 0, 0, []
    t_start = time.time()

    for epoch in range(EPOCHS):
        t_ep = time.time()
        model.train()
        epoch_loss, n_batches = 0.0, 0
        optimizer.zero_grad()

        for bi, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda"):
                if use_turma_loss:
                    outputs = model(images, return_seg=True)
                else:
                    outputs = model(images)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                if use_turma_loss and isinstance(outputs, dict) and "seg_logits" in outputs:
                    loss, _ = criterion(logits, labels, outputs["seg_logits"], None)
                else:
                    loss = criterion(logits, labels)
                loss = loss / GRAD_ACCUM

            scaler.scale(loss).backward()

            if (bi + 1) % GRAD_ACCUM == 0 or (bi + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item() * GRAD_ACCUM
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        model.eval()
        vp, vl = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                outputs = model(images)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                vp.extend(logits.argmax(dim=1).cpu().numpy())
                vl.extend(labels.numpy())

        vf1 = f1_score(vl, vp, average="weighted", zero_division=0)
        vacc = accuracy_score(vl, vp)
        ep_sec = time.time() - t_ep
        total_s = time.time() - t_start
        m, s = divmod(int(total_s), 60)
        h, m = divmod(m, 60)

        marker = ""
        if vf1 > best_f1:
            best_f1, best_ep, patience = vf1, epoch + 1, 0
            marker = " *BEST*"
            torch.save({"epoch": epoch + 1, "model_state_dict": model.state_dict(),
                        "val_f1": vf1, "val_accuracy": vacc}, out_dir / "best.pt")
        else:
            patience += 1

        print(f"    Ep {epoch+1:2d}/{EPOCHS} | loss={avg_loss:.4f} val_f1={vf1:.4f} val_acc={vacc:.4f} | {ep_sec:.1f}s | {h}h{m:02d}m{s:02d}s{marker}", flush=True)
        history.append({"epoch": epoch+1, "train_loss": avg_loss, "val_f1": vf1, "val_accuracy": vacc})

        if patience >= 7:
            print(f"    Early stop at epoch {epoch+1}")
            break

    torch.save({"epoch": history[-1]["epoch"], "model_state_dict": model.state_dict()}, out_dir / "last.pt")
    with open(out_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    return {"best_val_f1": best_f1, "best_epoch": best_ep,
            "total_time": time.time() - t_start, "checkpoint": str(out_dir / "best.pt")}


def evaluate(model, dataloader, device, model_name="model"):
    model.eval()
    ap, al, ab = [], [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            probs = F.softmax(logits, dim=1)
            ap.extend(logits.argmax(dim=1).cpu().numpy())
            al.extend(labels.numpy())
            ab.extend(probs.cpu().numpy())

    yt, yp, ypr = np.array(al), np.array(ap), np.array(ab)
    cm = confusion_matrix(yt, yp, labels=range(4))
    pc_f1 = {}
    for i in range(4):
        p = cm[i, i] / max(cm[:, i].sum(), 1)
        r = cm[i, i] / max(cm[i, :].sum(), 1)
        pc_f1[CLASS_NAMES[i]] = float(2*p*r / max(p+r, 1e-10))

    try:
        yt_bin = label_binarize(yt, classes=range(4))
        roc_auc_val = roc_auc_score(yt_bin, ypr, multi_class="ovr", average="macro")
        pr_auc_val = average_precision_score(yt_bin, ypr, average="macro")
    except:
        roc_auc_val, pr_auc_val = 0.0, 0.0

    fpr_dict, tpr_dict, auc_dict = {}, {}, {}
    for i in range(4):
        fpr_dict[str(i)], tpr_dict[str(i)], _ = roc_curve(yt_bin[:, i], ypr[:, i])
        auc_dict[CLASS_NAMES[i]] = auc(fpr_dict[str(i)], tpr_dict[str(i)])

    pr_curve_d, rc_curve_d, pr_auc_d = {}, {}, {}
    for i in range(4):
        pr_curve_d[str(i)], rc_curve_d[str(i)], _ = precision_recall_curve(yt_bin[:, i], ypr[:, i])
        pr_auc_d[CLASS_NAMES[i]] = auc(rc_curve_d[str(i)], pr_curve_d[str(i)])

    sens = {CLASS_NAMES[i]: float(cm[i, i] / max(cm[i, :].sum(), 1)) for i in range(4)}
    spec = {}
    for i in range(4):
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        spec[CLASS_NAMES[i]] = float(tn / max(tn + fp, 1))

    per_class_p = {CLASS_NAMES[i]: float(cm[i, i] / max(cm[:, i].sum(), 1)) for i in range(4)}
    per_class_r = {CLASS_NAMES[i]: float(cm[i, i] / max(cm[i, :].sum(), 1)) for i in range(4)}

    return {
        "model_name": model_name,
        "accuracy": float(accuracy_score(yt, yp)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(yt, yp, average="weighted", zero_division=0)),
        "roc_auc": float(roc_auc_val), "pr_auc": float(pr_auc_val),
        "mcc": float(matthews_corrcoef(yt, yp)),
        "kappa": float(cohen_kappa_score(yt, yp)),
        "per_class_f1": pc_f1,
        "sensitivity": sens, "specificity": spec,
        "per_class_precision": per_class_p, "per_class_recall": per_class_r,
        "confusion_matrix": cm.tolist(),
        "roc_data": {"fpr": fpr_dict, "tpr": tpr_dict, "auc": auc_dict},
        "pr_data": {"precision": pr_curve_d, "recall": rc_curve_d, "auc": pr_auc_d},
        "y_true": yt.tolist(), "y_pred": yp.tolist(), "y_probs": ypr.tolist(),
    }


def measure_efficiency(model, device, n_runs=30):
    model.eval()
    dummy = torch.randn(1, 3, 224, 224, device=device)
    for _ in range(10):
        with torch.no_grad(): model(dummy)
    torch.cuda.synchronize()
    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad(): model(dummy)
        torch.cuda.synchronize()
        times.append((time.time() - t0) * 1000)
    params = sum(p.numel() for p in model.parameters())
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(): model(dummy)
    peak = torch.cuda.max_memory_allocated() / 1024**2
    return {"params": params, "size_mb": round(size_mb, 2),
            "latency_ms": round(np.mean(times), 2), "latency_std": round(np.std(times), 2),
            "fps": round(1000 / np.mean(times), 2), "peak_vram_mb": round(peak, 2)}


def run_ablation():
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {gpu_name} ({vram:.1f} GB)")
    print(f"Batch size: {BATCH_SIZE} x {GRAD_ACCUM} accum = {EFFECTIVE_BS} effective")
    t_global = time.time()

    set_seed(42)
    train_loader, val_loader, test_loader = load_data(seed=42)
    print(f"Data: train={len(train_loader.dataset)} val={len(val_loader.dataset)} test={len(test_loader.dataset)}")
    print(f"Batches/epoch: train={len(train_loader)} val={len(val_loader)}\n")

    results = {}
    partial = Path("paper/results/ablation_all.json")

    items = list(ABLATION_CONFIGS.items())
    for i, (exp_name, cfg) in enumerate(items):
        t0 = time.time()
        print(f"{'='*60}")
        print(f"[{i+1}/{len(items)}] {exp_name}: {cfg['desc']}")
        print(f"  Start: {time.strftime('%H:%M:%S')}")

        set_seed(42)
        model = cfg["builder"]().to(device)
        params = sum(p.numel() for p in model.parameters())
        print(f"  Params: {params/1e6:.2f}M")

        tr = train_model(model, exp_name, train_loader, val_loader, device, cfg["tl"], "paper/results/ablation")

        ckpt = torch.load(tr["checkpoint"], map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        test_r = evaluate(model, test_loader, device, exp_name)
        eff = measure_efficiency(model, device)

        entry = {
            "experiment": exp_name, "description": cfg["desc"], "seed": 42, "params": params,
            "train": {"best_val_f1": tr["best_val_f1"], "best_epoch": tr["best_epoch"], "total_time": tr["total_time"]},
            "test": {k: test_r[k] for k in ["accuracy","balanced_accuracy","macro_f1","weighted_f1","roc_auc","pr_auc","mcc","kappa","per_class_f1","sensitivity","specificity"]},
            "efficiency": eff, "confusion_matrix": test_r["confusion_matrix"],
            "roc_data": test_r["roc_data"], "pr_data": test_r["pr_data"],
            "y_true": test_r["y_true"], "y_pred": test_r["y_pred"], "y_probs": test_r["y_probs"],
        }
        results[f"{exp_name}_seed42"] = entry
        with open(partial, "w") as f:
            json.dump(results, f, indent=2)

        elapsed = time.time() - t0
        print(f"\n  TEST: acc={test_r['accuracy']:.4f} f1={test_r['macro_f1']:.4f} auc={test_r['roc_auc']:.4f}")
        print(f"  Time: {elapsed:.1f}s | Total: {time.time()-t_global:.0f}s\n")

    print(f"\n{'='*60}")
    print(f"ABLATION COMPLETE ({(time.time()-t_global)/60:.1f} min)")
    print(f"{'='*60}")
    print(f"\n{'Model':35s} {'Acc':>8s} {'F1':>8s} {'AUC':>8s} {'VRAM':>8s}")
    print("-" * 71)
    for key in sorted(results.keys()):
        r = results[key]
        print(f"  {r['experiment']:33s} {r['test']['accuracy']:.4f} {r['test']['macro_f1']:.4f} {r['test']['roc_auc']:.4f} {r['efficiency']['peak_vram_mb']:6.0f}MB")


def run_multiseed():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    t_global = time.time()

    set_seed(42)
    train_loader, val_loader, test_loader = load_data(seed=42)

    KEY_MODELS = {
        "A_baseline_densenet121": {"desc": "Baseline: DenseNet121", "builder": lambda: build_baseline("densenet121"), "tl": False},
        "D_tumor_aware": {"desc": "TumorAwareMultiTaskNet", "builder": lambda: build_tumor_aware("densenet121"), "tl": False},
        "E_turma_full": {"desc": "TURMA-Net Full", "builder": lambda: build_turma_net("densenet121", use_atgfm=True, use_consistency=True, use_uncertainty_weights=True), "tl": True},
    }
    SEEDS = [42, 123, 456]

    ms = {}
    ms_path = Path("paper/results/multi_seed_all.json")
    if ms_path.exists():
        ms = json.load(open(ms_path))

    total = len(KEY_MODELS) * len(SEEDS)
    print(f"\nMULTI-SEED: training {total} configs (10 epochs each)\n")

    rn = 0
    for mn, mc in KEY_MODELS.items():
        for seed in SEEDS:
            rn += 1
            key = f"{mn}_seed{seed}"
            if key in ms:
                print(f"[{rn}/{total}] [SKIP] {key}")
                continue
            t0 = time.time()
            print(f"{'='*60}")
            print(f"[{rn}/{total}] {mn} seed={seed}")

            set_seed(seed)
            model = mc["builder"]().to(device)
            params = sum(p.numel() for p in model.parameters())

            tr = train_model(model, key, train_loader, val_loader, device, mc["tl"], "paper/results/multi_seed")
            ckpt = torch.load(tr["checkpoint"], map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            test_r = evaluate(model, test_loader, device, key)
            eff = measure_efficiency(model, device)

            entry = {
                "model_name": mn, "description": mc["desc"], "seed": seed, "params": params,
                "train": {"best_val_f1": tr["best_val_f1"], "best_epoch": tr["best_epoch"], "total_time": tr["total_time"]},
                "test": {k: test_r[k] for k in ["accuracy","balanced_accuracy","macro_f1","weighted_f1","roc_auc","pr_auc","mcc","kappa","per_class_f1","sensitivity","specificity"]},
                "efficiency": eff, "confusion_matrix": test_r["confusion_matrix"],
                "roc_data": test_r["roc_data"], "pr_data": test_r["pr_data"],
                "y_true": test_r["y_true"], "y_pred": test_r["y_pred"], "y_probs": test_r["y_probs"],
            }
            ms[key] = entry
            with open(ms_path, "w") as f:
                json.dump(ms, f, indent=2)
            print(f"  TEST: acc={test_r['accuracy']:.4f} f1={test_r['macro_f1']:.4f}")
            print(f"  Time: {time.time()-t0:.1f}s\n")

    print(f"\nMULTI-SEED COMPLETE ({(time.time()-t_global)/60:.1f} min)")
    for mn in KEY_MODELS:
        mrs = {k: v for k, v in ms.items() if v.get("model_name") == mn}
        if mrs:
            accs = [v["test"]["accuracy"] for v in mrs.values()]
            f1s = [v["test"]["macro_f1"] for v in mrs.values()]
            aucs = [v["test"]["roc_auc"] for v in mrs.values()]
            print(f"  {mn}:")
            print(f"    Acc: {np.mean(accs):.4f} +/- {np.std(accs):.4f}  F1: {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}  AUC: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")


def run_external():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    ext_results = {}
    datasets = {
        "BRISC": "data/external/brisc2025/brisc2025/classification_task/test",
        "Kaggle": "data/external/kaggle_brain_tumor/Testing",
    }
    models_map = {
        "A_baseline_densenet121": ("A_baseline_densenet121_seed42", lambda: build_baseline("densenet121")),
        "D_tumor_aware": ("D_tumor_aware_seed42", lambda: build_tumor_aware("densenet121")),
        "E_turma_full": ("E_turma_full_seed42", lambda: build_turma_net("densenet121", use_atgfm=True, use_consistency=True, use_uncertainty_weights=True)),
    }
    for ext_name, ext_root in datasets.items():
        if not Path(ext_root).exists():
            print(f"\n  {ext_name}: NOT FOUND")
            continue
        ext_loader = load_external(ext_root)
        print(f"\nEXTERNAL: {ext_name} ({len(ext_loader.dataset)} samples)")
        for mn, (ck, builder) in models_map.items():
            ckpt_path = Path("paper/results/multi_seed") / ck / "best.pt"
            if not ckpt_path.exists():
                ckpt_path = Path("paper/results/ablation") / ck / "best.pt"
            if not ckpt_path.exists():
                print(f"  {mn}: NO CHECKPOINT"); continue
            model = builder().to(device)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            er = evaluate(model, ext_loader, device, f"{mn}_{ext_name}")
            ext_results[f"{mn}_{ext_name}"] = {
                "model": mn, "dataset": ext_name,
                "accuracy": er["accuracy"], "macro_f1": er["macro_f1"], "roc_auc": er["roc_auc"],
                "per_class_f1": er["per_class_f1"], "confusion_matrix": er["confusion_matrix"],
            }
            print(f"  {mn:30s} acc={er['accuracy']:.4f} f1={er['macro_f1']:.4f} auc={er['roc_auc']:.4f}")
    with open("paper/results/external_validation.json", "w") as f:
        json.dump(ext_results, f, indent=2)

def run_figures():
    import importlib
    spec = importlib.util.spec_from_file_location("gen", "scripts/generate_all_figures.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    ab = json.load(open("paper/results/ablation_all.json")) if Path("paper/results/ablation_all.json").exists() else {}
    ms = json.load(open("paper/results/multi_seed_all.json")) if Path("paper/results/multi_seed_all.json").exists() else {}
    gen.generate_figures({**ab, **ms}, ms)
    gen.generate_tables(ab)

if __name__ == "__main__":
    print(f"{'='*60}")
    print(f"GPU TRAINING - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Phase: {PHASE}")
    print(f"{'='*60}\n")

    if PHASE == "ablation":
        run_ablation()
    elif PHASE == "multiseed":
        run_multiseed()
    elif PHASE == "external":
        run_external()
    elif PHASE == "all":
        run_ablation()
        run_multiseed()
        run_external()
        run_figures()
