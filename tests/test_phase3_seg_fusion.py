"""Phase 3A ablation wiring: λ_seg (real masks) + t-only fusion.

Covers the 2026-11-21 additions:

1. ``ClassificationSample.mask_path`` -> dataset yields ``(image, label,
   mask)`` triples with a binarized, aligned (224, 224) mask.
2. ``train_multitask_model`` consumes 3-tuple batches and ACTIVATES
   ``L_seg`` (finding D2: historical runs never passed seg_targets), while
   plain 2-tuple batches keep the historical ``seg_loss == 0`` behaviour.
3. ``fusion_mode="t"`` (t-only vs g+t ablation): builds, forwards, and
   ``fuse(g, t)`` returns ``t`` unchanged; invalid modes raise.
4. ``load_brisc_split`` pairs classification labels with aligned masks
   (skipped when the BRISC download is absent).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from PIL import Image

from datasets.classification import (
    CLASS_NAMES,
    BrainTumorClassificationDataset,
    ClassificationSample,
)
from datasets.dataloaders import make_dataloader
from models.tumor_aware import TumorAwareConfig, TumorAwareMultiTaskNet
from training.multitask_trainer import (
    MultiTaskTrainingConfig,
    train_multitask_model,
)

NUM_SAMPLES = 32
BATCH = 8


# --------------------------------------------------------------------------
# 1. masked dataset
# --------------------------------------------------------------------------
def _write_sample(tmp_path, mask_values) -> ClassificationSample:
    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (64, 64), (10, 20, 30)).save(img_path)
    mask_path = tmp_path / "mask.png"
    import numpy as np

    Image.fromarray(np.array(mask_values, dtype=np.uint8)).save(mask_path)
    return ClassificationSample(
        image_path=img_path,
        class_name="Glioma",
        class_index=0,
        split="train",
        mask_path=mask_path,
    )


def test_masked_sample_yields_aligned_binary_triple(tmp_path) -> None:
    import numpy as np

    raw = np.zeros((32, 32), dtype=np.uint8)
    raw[8:24, 8:24] = 255  # anti-aliased-style binary mask
    sample = _write_sample(tmp_path, raw)

    ds = BrainTumorClassificationDataset([sample])
    image, label, mask = ds[0]

    assert image.shape == (3, 224, 224)
    assert label.dtype == torch.long and int(label) == 0
    assert mask.shape == (224, 224)
    assert mask.dtype == torch.int64
    assert set(torch.unique(mask).tolist()) <= {0, 1}
    # NEAREST resize of a centered square keeps tumor pixels present
    assert mask.sum() > 0


def test_antialiased_mask_binarized_at_127(tmp_path) -> None:
    import numpy as np

    raw = np.zeros((16, 16), dtype=np.uint8)
    raw[4:12, 4:12] = 200  # >= 127 -> tumor
    raw[0:2, 0:2] = 100  # < 127 -> background
    sample = _write_sample(tmp_path, raw)

    _, _, mask = BrainTumorClassificationDataset([sample])[0]
    assert set(torch.unique(mask).tolist()) <= {0, 1}
    assert mask.sum() > 0


def test_unmasked_sample_stays_pair(tmp_path) -> None:
    img_path = tmp_path / "plain.jpg"
    Image.new("RGB", (64, 64)).save(img_path)
    sample = ClassificationSample(
        image_path=img_path, class_name="Glioma", class_index=0, split="train"
    )
    item = BrainTumorClassificationDataset([sample])[0]
    assert len(item) == 2  # historical contract unchanged


# --------------------------------------------------------------------------
# 2. trainer: 3-tuple batches activate L_seg
# --------------------------------------------------------------------------
class TinySegModel(nn.Module):
    """Dict-contract model that also emits (B, 2, H, W) seg_logits."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3 * 8 * 8, 4)
        self.seg = nn.Conv2d(3, 2, kernel_size=1)

    def forward(self, x: torch.Tensor, return_seg: bool = False) -> dict:
        out = {"logits": self.fc(torch.flatten(x, 1))}
        if return_seg:
            out["seg_logits"] = self.seg(x)
        return out


def _seg_data(with_masks: bool):
    g = torch.Generator().manual_seed(0)
    images = torch.randn(NUM_SAMPLES, 3, 8, 8, generator=g)
    labels = torch.randint(0, 4, (NUM_SAMPLES,), generator=g)
    masks = (torch.rand(NUM_SAMPLES, 8, 8, generator=g) > 0.5).long()
    if with_masks:
        return torch.utils.data.TensorDataset(images, labels, masks)
    return torch.utils.data.TensorDataset(images, labels)


def _seg_config(out_dir) -> MultiTaskTrainingConfig:
    return MultiTaskTrainingConfig(
        backbone="tiny",
        pretrained=False,
        epochs=1,
        batch_size=BATCH,
        lr=1e-3,
        patience=7,
        seed=42,
        mixed_precision=False,
        gradient_accumulation_steps=1,
        output_dir=str(out_dir),
        seg_loss_weight=0.5,
    )


def test_three_tuple_batches_activate_seg_loss(tmp_path) -> None:
    ds = _seg_data(with_masks=True)
    train = make_dataloader(ds, batch_size=BATCH, shuffle=True, seed=42)
    val = make_dataloader(ds, batch_size=BATCH, shuffle=False)

    torch.manual_seed(0)
    result = train_multitask_model(
        _seg_config(tmp_path / "seg"),
        train,
        val,
        torch.device("cpu"),
        model=TinySegModel(),
    )
    seg_losses = [h["train_seg_loss"] for h in result.training_history]
    assert len(seg_losses) == 1
    assert seg_losses[0] > 0.0, "L_seg must be active when masks are supplied"


def test_two_tuple_batches_keep_historical_zero_seg_loss(tmp_path) -> None:
    ds = _seg_data(with_masks=False)
    train = make_dataloader(ds, batch_size=BATCH, shuffle=True, seed=42)
    val = make_dataloader(ds, batch_size=BATCH, shuffle=False)

    torch.manual_seed(0)
    with pytest.warns(RuntimeWarning, match="seg_targets"):
        result = train_multitask_model(
            _seg_config(tmp_path / "noseg"),
            train,
            val,
            torch.device("cpu"),
            model=TinySegModel(),
        )
    # D2: without masks the aux branch gets no gradient (historical runs)
    assert result.training_history[0]["train_seg_loss"] == 0.0


# --------------------------------------------------------------------------
# 3. t-only fusion
# --------------------------------------------------------------------------
def test_t_only_fusion_forwards_and_returns_t() -> None:
    cfg = TumorAwareConfig(pretrained=False, fusion_mode="t")
    model = TumorAwareMultiTaskNet(cfg)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    assert out["logits"].shape == (2, 4)
    assert out["fused_features"].shape == (2, cfg.fusion_dim)

    g = torch.randn(2, 1024)
    t = torch.randn(2, cfg.fusion_dim)
    assert torch.equal(model.fuse(g, t), t)  # z = t exactly


def test_default_fusion_mode_is_g_plus_t() -> None:
    cfg = TumorAwareConfig(pretrained=False)
    assert cfg.fusion_mode == "g+t"
    model = TumorAwareMultiTaskNet(cfg)
    g = torch.randn(2, 1024)
    t = torch.randn(2, cfg.fusion_dim)
    assert not torch.equal(model.fuse(g, t), t)  # goes through Phi


def test_invalid_fusion_mode_raises() -> None:
    with pytest.raises(ValueError, match="fusion_mode"):
        TumorAwareMultiTaskNet(TumorAwareConfig(pretrained=False, fusion_mode="x"))


# --------------------------------------------------------------------------
# 4. BRISC paired split (needs the local download)
# --------------------------------------------------------------------------
def test_brisc_split_pairs_masks() -> None:
    from scripts.phase3_train import load_brisc_split

    try:
        splits = load_brisc_split()
    except SystemExit:
        pytest.skip("BRISC data not present")

    assert set(splits) == {"train", "val", "test"}
    # full official splits participate: tumor slices carry their real mask,
    # no_tumor slices the generated all-zero mask (BRISC masks only tumors)
    assert len(splits["train"]) + len(splits["val"]) == 5000
    assert len(splits["test"]) == 1000
    for bucket in splits.values():
        assert {s.class_name for s in bucket} == set(CLASS_NAMES)
        for s in bucket:
            assert s.mask_path is not None and s.mask_path.exists()
            assert s.class_name in CLASS_NAMES
    # stratified split keeps every class in train and val
    assert {s.class_name for s in splits["train"]} == set(CLASS_NAMES)
    assert {s.class_name for s in splits["val"]} == set(CLASS_NAMES)
