"""Phase 3 shutdown-safety: per-epoch checkpoint + resume.

The multitask trainer must save a full resumable state
(``checkpoint_last.pt``) after EVERY epoch, and ``resume=True`` must
continue from the last completed epoch with model / optimizer /
scheduler / best-state / patience / history / RNG / shuffle order all
restored — i.e. a system shutdown mid-run loses at most the in-flight
epoch, never the run.
"""

from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn

from datasets.dataloaders import make_dataloader
from training.multitask_trainer import (
    MultiTaskTrainingConfig,
    train_multitask_model,
)

NUM_SAMPLES = 64
BATCH = 8
EPOCHS = 3
SEED = 42


class TinyDictModel(nn.Module):
    """Minimal model honouring the train_multitask_model dict contract.

    Dropout is included deliberately: matching the reference run proves
    the RNG state (dropout masks) is restored on resume.
    """

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()
        self.fc = nn.Linear(3 * 8 * 8, 4)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_seg: bool = False) -> dict:
        return {"logits": self.dropout(self.fc(torch.flatten(x, 1)))}


class CrashOnEpoch2Model(TinyDictModel):
    """Raises on the first forward of epoch 2 (simulated shutdown).

    Epoch 1 = 8 train batches + 8 val batches = 16 forwards; forward #17
    is epoch 2's first train batch, i.e. the checkpoint for epoch 1 is
    already on disk when the "crash" happens.
    """

    def __init__(self, crash_at: int = 17) -> None:
        super().__init__()
        self.crash_at = crash_at
        self.calls = 0

    def forward(self, x: torch.Tensor, return_seg: bool = False) -> dict:
        self.calls += 1
        if self.calls >= self.crash_at:
            raise RuntimeError("simulated system shutdown")
        return super().forward(x, return_seg)


def _make_data() -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    images = torch.randn(NUM_SAMPLES, 3, 8, 8, generator=g)
    labels = torch.randint(0, 4, (NUM_SAMPLES,), generator=g)
    return images, labels


def _make_loaders():
    images, labels = _make_data()
    ds = torch.utils.data.TensorDataset(images, labels)
    train = make_dataloader(ds, batch_size=BATCH, shuffle=True, seed=SEED)
    val = make_dataloader(ds, batch_size=BATCH, shuffle=False)
    return train, val


def _config(out_dir, epochs: int = EPOCHS) -> MultiTaskTrainingConfig:
    return MultiTaskTrainingConfig(
        backbone="tiny",
        pretrained=False,
        epochs=epochs,
        batch_size=BATCH,
        lr=1e-3,
        patience=7,  # never early-stops in these tiny runs
        seed=SEED,
        mixed_precision=False,  # CPU test
        gradient_accumulation_steps=1,
        output_dir=str(out_dir),
    )


def _fresh_model() -> TinyDictModel:
    # Deterministic init: the trainer re-seeds internally *after* the model
    # is built, so the test must fix the ambient RNG for identical weights.
    torch.manual_seed(123)
    return TinyDictModel()


def test_uninterrupted_reference_run(tmp_path) -> None:
    out = tmp_path / "ref"
    train, val = _make_loaders()
    result = train_multitask_model(
        _config(out), train, val, torch.device("cpu"), model=_fresh_model()
    )

    assert [h["epoch"] for h in result.training_history] == [1, 2, 3]
    assert (out / "checkpoint_last.pt").exists()
    assert (out / "best.pt").exists()
    assert (out / "progress.json").exists()

    ckpt = torch.load(out / "checkpoint_last.pt", weights_only=False)
    assert ckpt["epoch"] == 3  # completed epochs
    for key in (
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "best_val_f1",
        "best_epoch",
        "patience_counter",
        "history",
        "rng_state",
        "train_generator_state",
    ):
        assert key in ckpt, f"missing resume field: {key}"


def test_crash_then_resume_continues_from_saved_epoch(tmp_path) -> None:
    # Reference: uninterrupted 3-epoch run.
    ref_out = tmp_path / "ref"
    ref_train, ref_val = _make_loaders()
    ref = train_multitask_model(
        _config(ref_out), ref_train, ref_val, torch.device("cpu"),
        model=_fresh_model(),
    )

    # Attempt 1: crashes at the start of epoch 2 (checkpoint of epoch 1
    # is already saved when the "shutdown" hits).
    out = tmp_path / "resumed"
    train1, val1 = _make_loaders()
    torch.manual_seed(123)  # same init as _fresh_model() -> epoch 1 matches ref
    with pytest.raises(RuntimeError, match="simulated system shutdown"):
        train_multitask_model(
            _config(out), train1, val1, torch.device("cpu"),
            model=CrashOnEpoch2Model(),
        )
    assert (out / "checkpoint_last.pt").exists(), "crash lost the checkpoint"
    crashed = torch.load(out / "checkpoint_last.pt", weights_only=False)
    assert crashed["epoch"] == 1, "checkpoint must hold the last COMPLETED epoch"

    # Attempt 2: fresh process equivalent (fresh loaders/model), resume=True.
    train2, val2 = _make_loaders()
    result = train_multitask_model(
        _config(out), train2, val2, torch.device("cpu"),
        model=_fresh_model(), resume=True,
    )

    # Epoch numbering continues — no restart from epoch 1.
    assert [h["epoch"] for h in result.training_history] == [1, 2, 3]

    # The resumed epochs must match the uninterrupted reference: identical
    # losses prove optimizer / scheduler / RNG / shuffle order restored.
    for ref_h, res_h in zip(ref.training_history, result.training_history):
        assert res_h["train_loss"] == pytest.approx(
            ref_h["train_loss"], rel=1e-3, abs=1e-5
        )
        assert res_h["val_f1"] == pytest.approx(ref_h["val_f1"], rel=1e-4)
    assert result.best_val_f1 == pytest.approx(ref.best_val_f1, rel=1e-4)
    assert result.best_epoch == ref.best_epoch


def test_resume_without_checkpoint_starts_fresh(tmp_path) -> None:
    out = tmp_path / "fresh"
    train, val = _make_loaders()
    result = train_multitask_model(
        _config(out, epochs=1), train, val, torch.device("cpu"),
        model=_fresh_model(), resume=True,
    )
    assert [h["epoch"] for h in result.training_history] == [1]


def test_progress_json_matches_checkpoint_on_disk(tmp_path) -> None:
    out = tmp_path / "prog"
    train, val = _make_loaders()
    train_multitask_model(
        _config(out), train, val, torch.device("cpu"), model=_fresh_model()
    )
    progress = json.loads((out / "progress.json").read_text(encoding="utf-8"))
    ckpt = torch.load(out / "checkpoint_last.pt", weights_only=False)

    # progress.json must never lag behind the checkpoint on disk.
    assert progress["best_val_f1"] == pytest.approx(ckpt["best_val_f1"])
    assert progress["best_epoch"] == ckpt["best_epoch"]
    assert progress["epoch"] == ckpt["epoch"]
