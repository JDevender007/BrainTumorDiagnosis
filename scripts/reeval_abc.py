"""Re-evaluate A-C from new 10-epoch checkpoints."""
import json, torch, torch.nn.functional as F, numpy as np, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef, cohen_kappa_score,
    confusion_matrix, roc_curve, precision_recall_curve, auc as sklearn_auc)
from sklearn.preprocessing import label_binarize
from torchvision import transforms
from PIL import Image
from utils.seed import set_seed

torch.backends.cudnn.benchmark = True
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

def load_te():
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    from torch.utils.data import DataLoader
    return DataLoader(ClsDS("data/raw/classification", "Test", tf), 64, shuffle=False, num_workers=0, pin_memory=True)

def evaluate(model, dl, dev):
    model.eval(); ap, al, ab = [], [], []
    with torch.no_grad():
        for img, lab in dl:
            img = img.to(dev, non_blocking=True)
            lo = model(img); probs = F.softmax(lo, dim=1)
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
        torch.cuda.synchronize(); t0=__import__("time").time()
        with torch.no_grad(): model(d)
        torch.cuda.synchronize(); ts.append((__import__("time").time()-t0)*1000)
    params = sum(p.numel() for p in model.parameters())
    sz = sum(p.numel()*p.element_size() for p in model.parameters())/1024**2
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(): model(d)
    pk = torch.cuda.max_memory_allocated()/1024**2
    return {"params":int(params),"size_mb":round(sz,2),"latency_ms":round(np.mean(ts),2),
            "latency_std":round(np.std(ts),2),"fps":round(1000/np.mean(ts),2),"peak_vram_mb":round(pk,2)}

def build(name):
    from models.factory import create_backbone_classifier
    bn = {"A_baseline_densenet121":"densenet121","B_baseline_resnet50":"resnet50","C_baseline_efficientnet":"efficientnet_v2_s"}[name]
    return create_backbone_classifier(bn, num_classes=4, pretrained=True)

def main():
    dev = torch.device("cuda")
    te = load_te()
    ab = json.load(open("paper/results/ablation_all.json"))

    for name in ["A_baseline_densenet121", "B_baseline_resnet50", "C_baseline_efficientnet"]:
        key = f"{name}_seed42"
        ck = Path("paper/results/ablation") / name / "best.pt"
        if not ck.exists():
            print(f"  {name}: NO CHECKPOINT"); continue
        print(f"  Evaluating {name} from {ck}...")
        model = build(name).to(dev)
        ckpt = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        te_r = evaluate(model, te, dev)
        eff = measure_eff(model, dev)
        params = sum(p.numel() for p in model.parameters())
        # Read training info from history
        hist = Path("paper/results/ablation") / name / "training_history.json"
        train_info = {"best_val_f1": 0, "best_epoch": 0, "total_time": 0}
        if hist.exists():
            h = json.load(open(hist))
            best_f1 = max(h, key=lambda x: x["vf1"])
            train_info = {"best_val_f1": best_f1["vf1"], "best_epoch": best_f1["epoch"], "total_time": 0}
        desc = {"A_baseline_densenet121":"Baseline: DenseNet121","B_baseline_resnet50":"Baseline: ResNet50","C_baseline_efficientnet":"Baseline: EfficientNetV2-S"}[name]
        entry = {"experiment": name, "description": desc, "seed": 42, "params": params,
            "train": train_info,
            "test": {k: te_r[k] for k in ["accuracy","balanced_accuracy","macro_f1","weighted_f1","roc_auc","pr_auc","mcc","kappa","per_class_f1","sensitivity","specificity"]},
            "efficiency": eff, "confusion_matrix": te_r["confusion_matrix"],
            "roc_data": te_r["roc_data"], "pr_data": te_r["pr_data"],
            "y_true": te_r["y_true"], "y_pred": te_r["y_pred"], "y_probs": te_r["y_probs"]}
        ab[key] = entry
        print(f"    acc={te_r['accuracy']:.4f} f1={te_r['macro_f1']:.4f} auc={te_r['roc_auc']:.4f}")

    with open("paper/results/ablation_all.json", "w") as f:
        json.dump(ab, f, indent=2, cls=NpEncoder)
    print("  Updated ablation_all.json")

if __name__ == "__main__":
    main()
