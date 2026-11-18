"""Phase 2 code↔paper tests for models/tumor_aware.py.

Covers:
  * M_c (channel attention) is now instantiable behind
    ``use_channel_attention`` while the default keeps the historical model;
  * ``seg_decoder="multiscale"`` U-Net-style decoder (densenet121 only) vs the
    historical 7x7 no-skip "lightweight" decoder;
  * ``fuse()`` documents/embodies the exact Φ(g, t) op;
  * MC dropout fix: BatchNorm must stay in eval during sampling and the
    previous train/eval mode must be restored (Phase 0 finding #5).

All models are built with ``pretrained=False`` (no network access in tests).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from models.tumor_aware import (
    MultiScaleSegmentationHead,
    SegmentationHead,
    TumorAwareConfig,
    TumorAwareMultiTaskNet,
)


def _config(**overrides) -> TumorAwareConfig:
    defaults = dict(backbone="densenet121", pretrained=False)
    defaults.update(overrides)
    return TumorAwareConfig(**defaults)


def _batch(batch: int = 2) -> torch.Tensor:
    return torch.randn(batch, 3, 224, 224)


class TestChannelAttentionFlag:
    def test_default_has_no_channel_attention(self) -> None:
        """Default config must reproduce the historical (M_c-less) model."""
        model = TumorAwareMultiTaskNet(_config())
        assert model.tumor_pool.channel_attention is None

    def test_flag_wires_channel_attention(self) -> None:
        model = TumorAwareMultiTaskNet(_config(use_channel_attention=True))
        assert model.tumor_pool.channel_attention is not None
        model.eval()
        with torch.no_grad():
            out = model(_batch())
        assert out["logits"].shape == (2, 4)

    def test_flag_changes_forward(self) -> None:
        """M_c must actually affect the computation when enabled."""
        torch.manual_seed(0)
        plain = TumorAwareMultiTaskNet(_config(use_channel_attention=False))
        torch.manual_seed(0)
        with_c = TumorAwareMultiTaskNet(_config(use_channel_attention=True))
        with_c.load_state_dict(
            {
                k: v
                for k, v in plain.state_dict().items()
                if k in with_c.state_dict() and with_c.state_dict()[k].shape == v.shape
            },
            strict=False,
        )
        plain.eval()
        with_c.eval()
        x = _batch()
        with torch.no_grad():
            logits_plain = plain(x)["logits"]
            logits_c = with_c(x)["logits"]
        # same weights everywhere they overlap; M_c adds a sigmoid gate -> differs
        assert not torch.allclose(logits_plain, logits_c)


class TestSegDecoderVariants:
    def test_default_is_lightweight_no_skip_decoder(self) -> None:
        model = TumorAwareMultiTaskNet(_config())
        assert isinstance(model.seg_head, SegmentationHead)

    def test_multiscale_decoder_forward(self) -> None:
        model = TumorAwareMultiTaskNet(_config(seg_decoder="multiscale"))
        assert isinstance(model.seg_head, MultiScaleSegmentationHead)
        model.eval()
        with torch.no_grad():
            out = model(_batch(), return_seg=True)
        assert out["seg_logits"].shape == (2, 4, 224, 224)
        assert out["logits"].shape == (2, 4)

    def test_multiscale_rejects_non_densenet_backbone(self) -> None:
        with pytest.raises(ValueError, match="densenet121"):
            TumorAwareMultiTaskNet(
                _config(backbone="resnet50", seg_decoder="multiscale")
            )

    def test_invalid_seg_decoder_value(self) -> None:
        with pytest.raises(ValueError, match="seg_decoder"):
            TumorAwareMultiTaskNet(_config(seg_decoder="foo"))

    def test_taps_reset_between_forwards(self) -> None:
        model = TumorAwareMultiTaskNet(_config(seg_decoder="multiscale"))
        model.eval()
        with torch.no_grad():
            model(_batch(1), return_seg=True)
            assert len(model._scale_taps) == 3
            model(_batch(1), return_seg=True)
            assert len(model._scale_taps) == 3  # not accumulating


class TestFuse:
    def test_fuse_matches_attention_path(self) -> None:
        model = TumorAwareMultiTaskNet(_config())
        g = torch.randn(2, 1024)
        t = torch.randn(2, 256)
        assert torch.allclose(model.fuse(g, t), model.fusion(g, t))

    def test_fuse_concat_path(self) -> None:
        model = TumorAwareMultiTaskNet(_config(use_attention_fusion=False))
        g = torch.randn(2, 1024)
        t = torch.randn(2, 256)
        assert torch.equal(model.fuse(g, t), torch.cat([g, t], dim=1))

    def test_forward_uses_fuse(self) -> None:
        model = TumorAwareMultiTaskNet(_config())
        model.eval()
        with torch.no_grad():
            out = model(_batch())
        assert out["fused_features"].shape == (2, 256)


class TestMcDropoutFix:
    def test_batchnorm_stays_eval_and_mode_restored(self) -> None:
        model = TumorAwareMultiTaskNet(_config())
        model.eval()  # inference mode, as in evaluation paths
        bn_modules = [
            m for m in model.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))
        ]
        assert bn_modules, "expected BatchNorm layers in the model"
        before = {id(m): m.running_mean.clone() for m in bn_modules}

        result = model.predict_with_uncertainty(_batch(), num_passes=3)

        for m in bn_modules:
            assert not m.training, "BatchNorm must stay in eval during MC passes"
            assert torch.equal(m.running_mean, before[id(m)]), (
                "MC passes must not update BatchNorm running stats"
            )
        assert not model.training, "previous (eval) mode must be restored"
        assert result["mean_probabilities"].shape == (2, 4)
        assert result["mutual_information"].shape == (2,)

    def test_sampling_is_stochastic(self) -> None:
        model = TumorAwareMultiTaskNet(_config())
        model.eval()
        x = _batch()
        with torch.no_grad():
            run_a = model.predict_with_uncertainty(x, num_passes=4)
            run_b = model.predict_with_uncertainty(x, num_passes=4)
        assert not torch.allclose(
            run_a["mean_probabilities"], run_b["mean_probabilities"]
        ), "dropout must produce different MC means across runs"

    def test_train_mode_restored_after_training_call(self) -> None:
        model = TumorAwareMultiTaskNet(_config())
        model.train()
        model.predict_with_uncertainty(_batch(2), num_passes=2)
        assert model.training, "train mode must be restored for training loops"


class TestMultiTaskLossSegSupervision:
    """Phase 2 finding D2: L_seg was 0 in every historical run."""

    def _outputs(self) -> dict:
        return {
            "logits": torch.randn(2, 4),
            "seg_logits": torch.randn(2, 4, 16, 16),
        }

    def test_no_targets_gives_zero_seg_loss_and_warns_once(self) -> None:
        import warnings

        from training.multitask_trainer import MultiTaskLoss

        loss_fn = MultiTaskLoss()
        outputs = self._outputs()
        labels = torch.tensor([0, 1])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            first = loss_fn(outputs, labels)
            second = loss_fn(outputs, labels)  # second batch: no repeat warning

        seg_warnings = [
            w
            for w in caught
            if issubclass(w.category, RuntimeWarning)
            and "seg_targets" in str(w.message)
        ]
        assert len(seg_warnings) == 1, "warn once per training run, not per batch"
        assert first["seg_loss"].item() == 0.0
        assert second["seg_loss"].item() == 0.0
        assert torch.allclose(first["total_loss"], first["cls_loss"])

    def test_targets_activate_2d_dice(self) -> None:
        import warnings

        from training.multitask_trainer import MultiTaskLoss

        loss_fn = MultiTaskLoss()
        outputs = self._outputs()
        labels = torch.tensor([0, 1])
        seg_targets = torch.randint(0, 4, (2, 16, 16))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = loss_fn(outputs, labels, seg_targets)

        assert out["seg_loss"].item() > 0.0
        expected = out["cls_loss"] + 0.5 * out["seg_loss"]
        assert torch.allclose(out["total_loss"], expected)

    def test_5d_volume_path_still_works(self) -> None:
        import warnings

        from training.multitask_trainer import MultiTaskLoss

        loss_fn = MultiTaskLoss()
        outputs = {
            "logits": torch.randn(2, 4),
            "seg_logits": torch.randn(2, 5, 4, 8, 8),
        }
        labels = torch.tensor([0, 1])
        seg_targets = torch.randint(0, 5, (2, 4, 8, 8))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = loss_fn(outputs, labels, seg_targets)
        assert out["seg_loss"].item() > 0.0

    def test_dice2d_rejects_wrong_rank(self) -> None:
        from losses.segmentation import DiceLoss2D

        loss = DiceLoss2D()
        with pytest.raises(ValueError, match="\\(B, C, H, W\\)"):
            loss(torch.randn(2, 4, 2, 16, 16), torch.randint(0, 4, (2, 16, 16)))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
