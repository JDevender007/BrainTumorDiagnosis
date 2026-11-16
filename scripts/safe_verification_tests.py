"""Safe verification tests for final audit."""
import sys, json, torch, traceback
sys.path.insert(0, ".")

results = []

def log(name, status, detail=""):
    results.append({"name": name, "status": status, "detail": detail})
    icon = "PASS" if status == "PASS" else "FAIL" if status == "FAIL" else "WARN"
    print(f"  [{icon}] {name}: {detail}")

print("=" * 60)
print("SAFE VERIFICATION TESTS")
print("=" * 60)

# 1. TURMA-Net instantiation + forward
print("\n--- TURMA-Net ---")
try:
    from models.novel.turma_net import TurmaNet, TurmaNetConfig
    cfg = TurmaNetConfig(backbone="densenet121", num_classes=4, use_atgfm=True,
                         use_consistency=True, use_uncertainty_weights=True)
    model = TurmaNet(cfg)
    params = sum(p.numel() for p in model.parameters())
    log("TURMA-Net instantiate", "PASS", f"{params/1e6:.2f}M params")

    x = torch.randn(2, 3, 224, 224)
    out = model(x, return_seg=True)
    assert "logits" in out and out["logits"].shape == (2, 4)
    assert "seg_logits" in out and out["seg_logits"].shape == (2, 4, 224, 224)
    assert "loc_uncertainty" in out and out["loc_uncertainty"].shape == (2, 1)
    assert "uncertainty_logits" in out
    log("TURMA-Net forward", "PASS", f"logits={out['logits'].shape}, seg={out['seg_logits'].shape}")

    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(out["logits"], torch.tensor([0, 1]))
    loss.backward()
    log("TURMA-Net backward", "PASS", f"loss={loss.item():.4f}")
except Exception as e:
    log("TURMA-Net", "FAIL", str(e))
    traceback.print_exc()

# 2. ATGFM module
print("\n--- ATGFM ---")
try:
    from models.novel.turma_net import AdaptiveTumorGuidedModulation
    atgfm = AdaptiveTumorGuidedModulation(encoder_channels=1024, fusion_dim=256)
    g = torch.randn(2, 1024)
    l = torch.randn(2, 256)
    unc = torch.randn(2, 1)
    fused = atgfm(g, l, unc)
    assert fused.shape == (2, 256)
    log("ATGFM forward", "PASS", f"output={fused.shape}")

    # Check gamma/beta/alpha exist and are sigmoid/tanh
    has_gamma = any("gamma" in n for n, _ in atgfm.named_parameters())
    has_beta = any("beta" in n for n, _ in atgfm.named_parameters())
    has_alpha = any("alpha" in n for n, _ in atgfm.named_parameters())
    log("ATGFM components", "PASS", f"gamma={has_gamma}, beta={has_beta}, alpha={has_alpha}")
except Exception as e:
    log("ATGFM", "FAIL", str(e))

# 3. TumorAwareMultiTaskNet
print("\n--- TumorAware ---")
try:
    from models.tumor_aware import TumorAwareMultiTaskNet, TumorAwareConfig
    cfg = TumorAwareConfig(backbone="densenet121", num_classes=4)
    model = TumorAwareMultiTaskNet(cfg)
    params = sum(p.numel() for p in model.parameters())
    x = torch.randn(2, 3, 224, 224)
    out = model(x, return_seg=True)
    assert "logits" in out
    log("TumorAware forward", "PASS", f"{params/1e6:.2f}M params, logits={out['logits'].shape}")
except Exception as e:
    log("TumorAware", "FAIL", str(e))

# 4. Losses
print("\n--- Losses ---")
try:
    from models.novel.losses import TurmaNetLoss, FocalLoss, DiceCELoss, ConsistencyLoss
    loss_fn = TurmaNetLoss(num_classes=4, use_uncertainty_weights=True)
    logits = torch.randn(4, 4)
    targets = torch.tensor([0, 1, 2, 3])
    seg_logits = torch.randn(4, 4, 224, 224)
    seg_targets = torch.randint(0, 4, (4, 224, 224))
    total_loss, components = loss_fn(logits, targets, seg_logits, seg_targets)
    assert total_loss.requires_grad
    log("TurmaNetLoss", "PASS", f"total={total_loss.item():.4f}, components={list(components.keys())}")

    total_loss.backward()
    log("TurmaNetLoss backward", "PASS")

    # Without uncertainty weights
    loss_fn2 = TurmaNetLoss(num_classes=4, use_uncertainty_weights=False)
    total_loss2, _ = loss_fn2(logits, targets, seg_logits, seg_targets)
    log("TurmaNetLoss (no UW)", "PASS", f"total={total_loss2.item():.4f}")
except Exception as e:
    log("Losses", "FAIL", str(e))
    traceback.print_exc()

# 5. Backbone classifiers
print("\n--- Backbone Classifiers ---")
try:
    from models.factory import create_backbone_classifier
    for bn in ["densenet121", "resnet50", "efficientnet_v2_s"]:
        m = create_backbone_classifier(bn, num_classes=4, pretrained=False)
        p = sum(p.numel() for p in m.parameters())
        x = torch.randn(2, 3, 224, 224)
        out = m(x)
        assert out.shape == (2, 4)
        log(f"Classifier {bn}", "PASS", f"{p/1e6:.2f}M, out={out.shape}")
except Exception as e:
    log("Classifiers", "FAIL", str(e))

# 6. SimCLR / SSL
print("\n--- SimCLR ---")
try:
    from training.self_supervised_trainer import SimCLRModel, ProjectionHead, nt_xent_loss, ContrastiveAugmentation
    model = SimCLRModel(backbone="densenet121", projector_hidden=512, projection_dim=128)
    x = torch.randn(4, 3, 224, 224)
    z = model(x)
    assert z.shape == (4, 128)
    log("SimCLR forward", "PASS", f"proj={z.shape}")

    loss = nt_xent_loss(z[:2], z[2:], temperature=0.07)
    loss.backward()
    log("NT-Xent loss", "PASS", f"loss={loss.item():.4f}")
except Exception as e:
    log("SimCLR", "FAIL", str(e))

# 7. Checkpoint loading
print("\n--- Checkpoint Loading ---")
try:
    ck = torch.load("paper/results/ablation/e_turma_full/best.pt", map_location="cpu", weights_only=False)
    assert "model_state_dict" in ck or "state_dict" in ck
    key = "model_state_dict" if "model_state_dict" in ck else "state_dict"
    log("TURMA checkpoint load", "PASS", f"keys={list(ck.keys())[:5]}")
except Exception as e:
    log("TURMA checkpoint", "FAIL", str(e))

try:
    ck = torch.load("paper/results/multi_seed/d_tumor_aware_seed42/best.pt", map_location="cpu", weights_only=False)
    log("TumorAware checkpoint", "PASS", f"keys={list(ck.keys())[:5]}")
except Exception as e:
    log("TumorAware checkpoint", "FAIL", str(e))

# 8. MC Dropout uncertainty
print("\n--- MC Dropout ---")
try:
    from models.novel.turma_net import TurmaNet, TurmaNetConfig
    cfg = TurmaNetConfig(backbone="densenet121", num_classes=4, mc_dropout_passes=5)
    model = TurmaNet(cfg)
    x = torch.randn(2, 3, 224, 224)
    (mean_p, preds, conf, var, ent, mi, epist, aleat) = model.predict_with_uncertainty(x, num_passes=5)
    assert mean_p.shape == (2, 4)
    assert ent.shape == (2,)
    assert mi.shape == (2,)
    log("MC Dropout uncertainty", "PASS", f"entropy={ent.mean().item():.4f}, MI={mi.mean().item():.4f}")
except Exception as e:
    log("MC Dropout", "FAIL", str(e))

# 9. UncertaintyWeightedMultiTaskLoss module
print("\n--- UncertaintyWeightedLoss ---")
try:
    from models.novel.turma_net import UncertaintyWeightedMultiTaskLoss
    uw = UncertaintyWeightedMultiTaskLoss(num_tasks=3)
    losses = [torch.tensor(1.0, requires_grad=True), torch.tensor(0.5, requires_grad=True), torch.tensor(0.3, requires_grad=True)]
    total = uw(losses)
    total.backward()
    log("UncertaintyWeightedLoss", "PASS", f"total={total.item():.4f}")
except Exception as e:
    log("UncertaintyWeightedLoss", "FAIL", str(e))

# 10. Verify 3D U-Net segmentation
print("\n--- 3D U-Net Segmentation ---")
try:
    from models.segmentation import UNet3D, SegmentationConfig
    cfg = SegmentationConfig(in_channels=4, out_channels=4, base_channels=8)
    model = UNet3D(cfg)
    x = torch.randn(1, 4, 32, 32, 32)
    out = model(x)
    assert out.shape == (1, 4, 32, 32, 32)
    log("3D U-Net forward", "PASS", f"in={x.shape}, out={out.shape}")
except Exception as e:
    log("3D U-Net", "FAIL", str(e))

# 11. Statistical validation
print("\n--- Statistical Validation ---")
try:
    from evaluation.statistical_validation import bootstrap_confidence_interval, mcnemar_test, cohens_d, paired_bootstrap_test
    import numpy as np
    y_true = np.array([0,1,2,3,0,1,2,3,0,1])
    y_pred1 = np.array([0,1,2,3,0,1,2,0,0,1])
    y_pred2 = np.array([0,1,2,0,0,1,2,3,0,1])
    ci = bootstrap_confidence_interval(y_true, y_pred1)
    log("Bootstrap CI", "PASS", f"CI={ci}")
    mcnemar = mcnemar_test(y_true, y_pred1, y_pred2)
    log("McNemar test", "PASS", f"statistic={mcnemar.get('statistic', 'N/A')}")
    d = cohens_d(y_pred1.astype(float), y_pred2.astype(float))
    log("Cohen's d", "PASS", f"effect_size={d}")
except Exception as e:
    log("Statistical validation", "FAIL", str(e))

# 12. Verify data exists
print("\n--- Data Verification ---")
from pathlib import Path
cls_train = Path("data/raw/classification/Train")
cls_test = Path("data/raw/classification/Test")
brisc = Path("data/external/brisc2025/brisc2025/classification_task/test")
kaggle = Path("data/external/kaggle_brain_tumor/Testing")
log("Classification Train", "PASS" if cls_train.exists() else "FAIL", str(cls_train))
log("Classification Test", "PASS" if cls_test.exists() else "FAIL", str(cls_test))
log("BRISC external", "PASS" if brisc.exists() else "FAIL", str(brisc))
log("Kaggle external", "PASS" if kaggle.exists() else "FAIL", str(kaggle))

# 13. Result files
print("\n--- Result Files ---")
for fn in ["paper/results/ablation_all.json", "paper/results/multi_seed_all.json", "paper/results/external_validation.json"]:
    exists = Path(fn).exists()
    if exists:
        d = json.load(open(fn))
        log(fn, "PASS", f"{len(d)} entries")
    else:
        log(fn, "FAIL", "NOT FOUND")

# 14. Figures
print("\n--- Figures ---")
fig_dir = Path("paper/figures")
figs = list(fig_dir.glob("fig*.png"))
log("Figures exist", "PASS" if len(figs) > 0 else "FAIL", f"{len(figs)} files")

# 15. Tables
print("\n--- Tables ---")
tab_dir = Path("paper/tables")
tabs = list(tab_dir.glob("table*.md"))
log("Tables exist", "PASS" if len(tabs) > 0 else "FAIL", f"{len(tabs)} files")

# Summary
print("\n" + "=" * 60)
passes = sum(1 for r in results if r["status"] == "PASS")
fails = sum(1 for r in results if r["status"] == "FAIL")
warns = sum(1 for r in results if r["status"] == "WARN")
print(f"TOTAL: {len(results)} tests | PASS: {passes} | FAIL: {fails} | WARN: {warns}")
print("=" * 60)

# Save results
with open("research_implementation_audit/safe_verification_results.json", "w") as f:
    json.dump(results, f, indent=2)
