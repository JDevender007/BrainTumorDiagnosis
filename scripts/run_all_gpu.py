"""Merge all ablation results, run multi-seed + external + figures."""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef, cohen_kappa_score,
    confusion_matrix, roc_curve, precision_recall_curve, auc as sklearn_auc)
from sklearn.preprocessing import label_binarize
from utils.seed import set_seed

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

BS, ACCUM, EPOCHS, PATIENCE = 48, 2, 10, 7
CLASS_NAMES = ["glioma", "meningioma", "no tumor", "pituitary"]
CLASS_MAP = {"glioma": 0, "meningioma": 1, "no tumor": 2, "pituitary": 3, "notumor": 2, "no_tumor": 2}

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)): return float(obj)
        if isinstance(obj, (np.int32, np.int64)): return int(obj)
        return super().default(obj)

def norm_name(n):
    n = n.lower().strip()
    for k, v in CLASS_MAP.items():
        if n == k: return ["glioma","meningioma","no tumor","pituitary"][v]
    return n

class ClsDS:
    def __init__(self, root, split, tf):
        self.samples, self.labels = [], []
        d = Path(root) / split
        if not d.exists(): d = Path(root) / split.capitalize()
        for cd in sorted(d.iterdir()):
            if not cd.is_dir(): continue
            cn = norm_name(cd.name)
            if cn not in CLASS_MAP: continue
            for p in sorted(cd.iterdir()):
                if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".tiff"}:
                    self.samples.append(str(p)); self.labels.append(CLASS_MAP[cn])
        self.tf = tf
    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.tf(Image.open(self.samples[i]).convert("RGB")), self.labels[i]

def load_data(seed=42):
    tf_tr = transforms.Compose([transforms.Resize((224,224)), transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(), transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    tf_te = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    full = ClsDS("data/raw/classification", "Train", tf_tr)
    te = ClsDS("data/raw/classification", "Test", tf_te)
    set_seed(seed); idx = list(range(len(full))); np.random.seed(seed); np.random.shuffle(idx)
    vn = int(len(full) * 0.2)
    tr = Subset(full, idx[vn:])
    val = Subset(ClsDS("data/raw/classification", "Train", tf_te), idx[:vn])
    kw = dict(num_workers=0, pin_memory=True)
    return (DataLoader(tr, BS, shuffle=True, drop_last=True, **kw),
            DataLoader(val, BS*2, **kw), DataLoader(te, BS*2, **kw))

def load_external(root, batch_size=64):
    transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    samples, labels = [], []
    root_path = Path(root)
    for class_dir in sorted(root_path.iterdir()):
        if not class_dir.is_dir(): continue
        cn = norm_name(class_dir.name)
        if cn not in CLASS_MAP: continue
        for p in sorted(class_dir.iterdir()):
            if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".tiff"}:
                samples.append(str(p)); labels.append(CLASS_MAP[cn])
    class ExtDS:
        def __init__(self, s, l, t): self.s, self.l, self.t = s, l, t
        def __len__(self): return len(self.s)
        def __getitem__(self, i): return self.t(Image.open(self.s[i]).convert("RGB")), self.l[i]
    return DataLoader(ExtDS(samples, labels, transform), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

def build(name, seed=42):
    set_seed(seed)
    if name in ["A_baseline_densenet121","B_baseline_resnet50","C_baseline_efficientnet"]:
        from models.factory import create_backbone_classifier
        bn = {"A_baseline_densenet121":"densenet121","B_baseline_resnet50":"resnet50","C_baseline_efficientnet":"efficientnet_v2_s"}[name]
        return create_backbone_classifier(bn, num_classes=4, pretrained=True)
    if name == "d_tumor_aware":
        from models.tumor_aware import TumorAwareMultiTaskNet, TumorAwareConfig
        return TumorAwareMultiTaskNet(TumorAwareConfig(backbone="densenet121", num_classes=4))
    kw = dict(use_atgfm=True, use_consistency=True, use_uncertainty_weights=True)
    if name == "f_no_atgfm": kw["use_atgfm"] = False
    if name == "g_no_consistency": kw["use_consistency"] = False
    if name == "h_no_uncertainty": kw["use_uncertainty_weights"] = False
    from models.novel.turma_net import TurmaNet, TurmaNetConfig
    return TurmaNet(TurmaNetConfig(backbone="densenet121", num_classes=4, **kw))

USE_TURMA = {"d_tumor_aware": False, "e_turma_full": True, "f_no_atgfm": True, "g_no_consistency": True, "h_no_uncertainty": True}
DESCS = {
    "A_baseline_densenet121": "Baseline: DenseNet121",
    "B_baseline_resnet50": "Baseline: ResNet50",
    "C_baseline_efficientnet": "Baseline: EfficientNetV2-S",
    "d_tumor_aware": "TumorAwareMultiTaskNet",
    "e_turma_full": "TURMA-Net Full",
    "f_no_atgfm": "TURMA w/o ATGFM",
    "g_no_consistency": "TURMA w/o Consistency",
    "h_no_uncertainty": "TURMA w/o Uncertainty",
}

def train_model(model, name, tr, va, dev, use_turma=False, seed=42):
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    if use_turma:
        from models.novel.losses import TurmaNetLoss
        crit = TurmaNetLoss(num_classes=4, use_uncertainty_weights=True)
    else:
        crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda")
    od = Path("paper/results/multi_seed") / f"{name}_seed{seed}"; od.mkdir(parents=True, exist_ok=True)
    best_f1, be, pat, hist = 0.0, 0, 0, []
    t0 = time.time()
    for ep in range(EPOCHS):
        tep = time.time(); model.train(); el, nb = 0.0, 0; opt.zero_grad()
        for bi, (img, lab) in enumerate(tr):
            img = img.to(dev, non_blocking=True); lab = lab.to(dev, non_blocking=True)
            with torch.amp.autocast("cuda"):
                if use_turma:
                    out = model(img, return_seg=True); lo = out["logits"]
                    if "seg_logits" in out: loss, _ = crit(lo, lab, out["seg_logits"], None)
                    else: loss = crit(lo, lab)
                else:
                    out = model(img); lo = out["logits"] if isinstance(out, dict) else out
                    loss = crit(lo, lab)
                loss = loss / ACCUM
            scaler.scale(loss).backward()
            if (bi+1) % ACCUM == 0 or (bi+1) == len(tr):
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad()
            el += loss.item() * ACCUM; nb += 1
        sched.step()
        model.eval(); vp, vl = [], []
        with torch.no_grad():
            for img, lab in va:
                img = img.to(dev, non_blocking=True)
                out = model(img); lo = out["logits"] if isinstance(out, dict) else out
                vp.extend(lo.argmax(1).cpu().numpy()); vl.extend(lab.numpy())
        vf1 = f1_score(vl, vp, average="weighted", zero_division=0)
        va_ = accuracy_score(vl, vp); es = time.time()-tep; ts = time.time()-t0
        m, s = divmod(int(ts), 60); h, m = divmod(m, 60)
        mk = ""
        if vf1 > best_f1:
            best_f1, be, pat = vf1, ep+1, 0; mk = " *BEST*"
            torch.save({"epoch": ep+1, "model_state_dict": model.state_dict(), "val_f1": vf1}, od / "best.pt")
        else: pat += 1
        print(f"    Ep {ep+1:2d}/{EPOCHS} loss={el/nb:.4f} vf1={vf1:.4f} vacc={va_:.4f} {es:.1f}s {h}h{m:02d}m{s:02d}s{mk}", flush=True)
        hist.append({"epoch":ep+1,"loss":el/nb,"vf1":vf1,"vacc":va_})
        if pat >= PATIENCE: print("    Early stop"); break
    with open(od/"training_history.json","w") as f: json.dump(hist, f, indent=2)
    return {"best_val_f1": best_f1, "best_epoch": be, "total_time": time.time()-t0, "checkpoint": str(od/"best.pt")}

def evaluate(model, dl, dev, name="m"):
    model.eval(); ap, al, ab = [], [], []
    with torch.no_grad():
        for img, lab in dl:
            img = img.to(dev, non_blocking=True)
            out = model(img); lo = out["logits"] if isinstance(out, dict) else out
            probs = F.softmax(lo, dim=1)
            ap.extend(lo.argmax(1).cpu().numpy()); al.extend(lab.numpy()); ab.extend(probs.cpu().numpy())
    yt, yp, ypr = np.array(al), np.array(ap), np.array(ab)
    cm = confusion_matrix(yt, yp, labels=range(4))
    pc = {}
    for i in range(4):
        p_ = cm[i,i]/max(cm[:,i].sum(),1); r_ = cm[i,i]/max(cm[i,:].sum(),1)
        pc[CLASS_NAMES[i]] = float(2*p_*r_/max(p_+r_,1e-10))
    try:
        yt_b = label_binarize(yt, classes=range(4))
        ra = roc_auc_score(yt_b, ypr, multi_class="ovr", average="macro")
        pa = average_precision_score(yt_b, ypr, average="macro")
    except: ra, pa = 0.0, 0.0
    fpr_, tpr_, auc_ = {}, {}, {}
    for i in range(4):
        fpr_[str(i)], tpr_[str(i)], _ = roc_curve(yt_b[:,i], ypr[:,i])
        auc_[CLASS_NAMES[i]] = sklearn_auc(fpr_[str(i)], tpr_[str(i)])
    prc, rcc, pca = {}, {}, {}
    for i in range(4):
        prc[str(i)], rcc[str(i)], _ = precision_recall_curve(yt_b[:,i], ypr[:,i])
        pca[CLASS_NAMES[i]] = sklearn_auc(rcc[str(i)], prc[str(i)])
    sens = {CLASS_NAMES[i]: float(cm[i,i]/max(cm[i,:].sum(),1)) for i in range(4)}
    spec = {}
    for i in range(4):
        tn = cm.sum()-cm[i,:].sum()-cm[:,i].sum()+cm[i,i]; fp = cm[:,i].sum()-cm[i,i]
        spec[CLASS_NAMES[i]] = float(tn/max(tn+fp,1))
    return {"accuracy": float(accuracy_score(yt,yp)), "balanced_accuracy": float(balanced_accuracy_score(yt,yp)),
        "macro_f1": float(f1_score(yt,yp,average="macro",zero_division=0)),
        "weighted_f1": float(f1_score(yt,yp,average="weighted",zero_division=0)),
        "roc_auc": float(ra), "pr_auc": float(pa), "mcc": float(matthews_corrcoef(yt,yp)),
        "kappa": float(cohen_kappa_score(yt,yp)), "per_class_f1": pc, "sensitivity": sens, "specificity": spec,
        "confusion_matrix": cm.tolist(),
        "roc_data": {"fpr": fpr_, "tpr": tpr_, "auc": auc_},
        "pr_data": {"precision": prc, "recall": rcc, "auc": pca},
        "y_true": yt.tolist(), "y_pred": yp.tolist(), "y_probs": ypr.tolist()}

def measure_eff(model, dev, n=30):
    model.eval(); d = torch.randn(1,3,224,224,device=dev)
    for _ in range(10):
        with torch.no_grad(): model(d)
    torch.cuda.synchronize(); ts = []
    for _ in range(n):
        torch.cuda.synchronize(); t0=time.time()
        with torch.no_grad(): model(d)
        torch.cuda.synchronize(); ts.append((time.time()-t0)*1000)
    params = sum(p.numel() for p in model.parameters())
    sz = sum(p.numel()*p.element_size() for p in model.parameters())/1024**2
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(): model(d)
    pk = torch.cuda.max_memory_allocated()/1024**2
    return {"params":params,"size_mb":round(sz,2),"latency_ms":round(np.mean(ts),2),
            "latency_std":round(np.std(ts),2),"fps":round(1000/np.mean(ts),2),"peak_vram_mb":round(pk,2)}

def merge_ablation():
    results = {}
    ab_dir = Path("paper/results/ablation")
    # Load from individual JSON files (A-C have seed42 naming)
    for fn in ab_dir.glob("*_seed42.json"):
        with open(fn) as f: data = json.load(f)
        results[fn.stem] = data
    # Load from directory-based results (D-H)
    for d in ab_dir.iterdir():
        if not d.is_dir(): continue
        hist = d / "training_history.json"
        if not hist.exists(): continue
        key = f"{d.name}_seed42"
        if key in results: continue  # already loaded
        # Find checkpoint and rebuild result
        ck = d / "best.pt"
        if not ck.exists(): continue
        name = d.name
        with open(hist) as f: hist_data = json.load(f)
        best_f1 = max(h["vf1"] for h in hist_data)
        best_ep = [h["epoch"] for h in hist_data if h["vf1"] == best_f1][0]
        results[key] = {
            "experiment": name, "description": DESCS.get(name, name), "seed": 42,
            "params": 0, "train": {"best_val_f1": best_f1, "best_epoch": best_ep, "total_time": 0},
            "test": {"accuracy": 0, "balanced_accuracy": 0, "macro_f1": 0, "weighted_f1": 0,
                     "roc_auc": 0, "pr_auc": 0, "mcc": 0, "kappa": 0, "per_class_f1": {},
                     "sensitivity": {}, "specificity": {}},
            "efficiency": {"params": 0, "size_mb": 0, "latency_ms": 0, "latency_std": 0, "fps": 0, "peak_vram_mb": 0},
            "confusion_matrix": [], "roc_data": {}, "pr_data": {},
            "y_true": [], "y_pred": [], "y_probs": [],
        }
    return results


def main():
    dev = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    tg = time.time()
    tr, va, te = load_data()

    # ===== PHASE 1: Rebuild ablation_all.json =====
    print("\n" + "="*60)
    print("PHASE 1: Rebuild ablation results")
    print("="*60)
    results = merge_ablation()
    needs_eval = [k for k, v in results.items() if v["test"]["accuracy"] == 0]
    print(f"  {len(results)} configs found, {len(needs_eval)} need evaluation")

    for key in needs_eval:
        name = key.replace("_seed42", "")
        print(f"  Evaluating {name}...")
        ck = Path("paper/results/ablation") / name / "best.pt"
        if not ck.exists():
            print(f"    NO CHECKPOINT, skipping"); continue
        model = build(name).to(dev)
        ckpt = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        te_r = evaluate(model, te, dev, name)
        eff = measure_eff(model, dev)
        params = sum(p.numel() for p in model.parameters())
        results[key]["params"] = params
        results[key]["test"] = {k: te_r[k] for k in ["accuracy","balanced_accuracy","macro_f1","weighted_f1","roc_auc","pr_auc","mcc","kappa","per_class_f1","sensitivity","specificity"]}
        results[key]["efficiency"] = eff
        results[key]["confusion_matrix"] = te_r["confusion_matrix"]
        results[key]["roc_data"] = te_r["roc_data"]
        results[key]["pr_data"] = te_r["pr_data"]
        results[key]["y_true"] = te_r["y_true"]
        results[key]["y_pred"] = te_r["y_pred"]
        results[key]["y_probs"] = te_r["y_probs"]
        print(f"    acc={te_r['accuracy']:.4f} f1={te_r['macro_f1']:.4f} auc={te_r['roc_auc']:.4f}")

    with open("paper/results/ablation_all.json", "w") as f:
        json.dump(results, f, indent=2, cls=NpEncoder)

    # ===== PHASE 2: Multi-seed =====
    print("\n" + "="*60)
    print("PHASE 2: Multi-seed (3 models x 3 seeds x 10 epochs)")
    print("="*60)

    KEY_MODELS = [
        ("A_baseline_densenet121", False),
        ("d_tumor_aware", False),
        ("e_turma_full", True),
    ]
    SEEDS = [42, 123, 456]

    ms = {}
    ms_path = Path("paper/results/multi_seed_all.json")
    if ms_path.exists(): ms = json.load(open(ms_path))

    for mn, use_t in KEY_MODELS:
        for seed in SEEDS:
            key = f"{mn}_seed{seed}"
            if key in ms:
                print(f"  [SKIP] {key}"); continue
            t0 = time.time()
            print(f"\n{'='*60}")
            print(f"  {mn} seed={seed}")
            set_seed(seed)
            model = build(mn, seed).to(dev)
            trr = train_model(model, mn, tr, va, dev, use_t, seed=seed)
            ckpt = torch.load(trr["checkpoint"], map_location=dev, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            te_r = evaluate(model, te, dev, mn)
            eff = measure_eff(model, dev)
            params = sum(p.numel() for p in model.parameters())
            entry = {"model_name": mn, "description": DESCS.get(mn, mn), "seed": seed, "params": params,
                "train": {"best_val_f1": trr["best_val_f1"], "best_epoch": trr["best_epoch"], "total_time": trr["total_time"]},
                "test": {k: te_r[k] for k in ["accuracy","balanced_accuracy","macro_f1","weighted_f1","roc_auc","pr_auc","mcc","kappa","per_class_f1","sensitivity","specificity"]},
                "efficiency": eff, "confusion_matrix": te_r["confusion_matrix"],
                "roc_data": te_r["roc_data"], "pr_data": te_r["pr_data"],
                "y_true": te_r["y_true"], "y_pred": te_r["y_pred"], "y_probs": te_r["y_probs"]}
            ms[key] = entry
            with open(ms_path, "w") as f: json.dump(ms, f, indent=2, cls=NpEncoder)
            print(f"  DONE: acc={te_r['accuracy']:.4f} f1={te_r['macro_f1']:.4f} ({time.time()-t0:.0f}s)")

    # ===== PHASE 3: External validation =====
    print("\n" + "="*60)
    print("PHASE 3: External validation")
    print("="*60)
    ext_results = {}
    ext_datasets = {"BRISC": "data/external/brisc2025/brisc2025/classification_task/test",
                    "Kaggle": "data/external/kaggle_brain_tumor/Testing"}
    ext_models = [("A_baseline_densenet121", "A_baseline_densenet121_seed42"),
                  ("d_tumor_aware", "d_tumor_aware_seed42"),
                  ("e_turma_full", "e_turma_full_seed42")]

    for ext_name, ext_root in ext_datasets.items():
        if not Path(ext_root).exists(): print(f"  {ext_name}: NOT FOUND"); continue
        ext_loader = load_external(ext_root)
        print(f"\n  {ext_name} ({len(ext_loader.dataset)} samples)")
        for mn, ck_key in ext_models:
            ck = Path("paper/results/multi_seed") / ck_key / "best.pt"
            if not ck.exists(): ck = Path("paper/results/ablation") / ck_key.replace("_seed42","") / "best.pt"
            if not ck.exists(): print(f"    {mn}: NO CKPT"); continue
            model = build(mn).to(dev)
            ckpt = torch.load(ck, map_location=dev, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            er = evaluate(model, ext_loader, dev, mn)
            k = f"{mn}_{ext_name}"
            ext_results[k] = {"model": mn, "dataset": ext_name,
                "accuracy": er["accuracy"], "macro_f1": er["macro_f1"], "roc_auc": er["roc_auc"],
                "per_class_f1": er["per_class_f1"], "confusion_matrix": er["confusion_matrix"]}
            print(f"    {mn:30s} acc={er['accuracy']:.4f} f1={er['macro_f1']:.4f} auc={er['roc_auc']:.4f}")
    with open("paper/results/external_validation.json", "w") as f: json.dump(ext_results, f, indent=2, cls=NpEncoder)

    # ===== PHASE 4: Figures =====
    print("\n" + "="*60)
    print("PHASE 4: Generating figures and tables")
    print("="*60)
    import importlib
    spec = importlib.util.spec_from_file_location("gen", "scripts/generate_all_figures.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    ms_final = json.load(open(ms_path)) if ms_path.exists() else {}
    gen.generate_figures({**results, **ms_final}, ms_final)
    gen.generate_tables(results)

    # ===== SUMMARY =====
    print(f"\n{'='*60}")
    print(f"ALL PHASES COMPLETE ({(time.time()-tg)/60:.1f} min)")
    print(f"{'='*60}")
    print(f"\n{'Model':35s} {'Acc':>8s} {'F1':>8s} {'AUC':>8s}")
    print("-"*63)
    for k in sorted(results):
        r = results[k]
        t = r["test"]
        print(f"  {r['experiment']:33s} {t['accuracy']:.4f} {t['macro_f1']:.4f} {t['roc_auc']:.4f}")

    if ms:
        print(f"\nMulti-seed (mean +/- std):")
        for mn, _ in KEY_MODELS:
            mrs = {k: v for k, v in ms.items() if v.get("model_name") == mn}
            if mrs:
                accs = [v["test"]["accuracy"] for v in mrs.values()]
                f1s = [v["test"]["macro_f1"] for v in mrs.values()]
                aucs = [v["test"]["roc_auc"] for v in mrs.values()]
                print(f"  {mn}:")
                print(f"    Acc: {np.mean(accs):.4f}+/-{np.std(accs):.4f}  F1: {np.mean(f1s):.4f}+/-{np.std(f1s):.4f}  AUC: {np.mean(aucs):.4f}+/-{np.std(aucs):.4f}")

    if ext_results:
        print(f"\nExternal validation:")
        for k, v in ext_results.items():
            print(f"  {k}: acc={v['accuracy']:.4f} f1={v['macro_f1']:.4f} auc={v['roc_auc']:.4f}")

if __name__ == "__main__":
    main()
