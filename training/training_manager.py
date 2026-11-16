"""
Persistent training manager for BrainTumorDiagnosis.

Tracks project-level and per-backbone training progress in a JSON
state file so training can be stopped, resumed, and survives PC
shutdowns (the trainer saves ``last.pt`` + history every epoch).

Workflow:

    # initialize the state file (detects existing checkpoints)
    python -m scripts.manage_training init

    # start all unfinished baselines (resumes where it left off)
    python -m scripts.manage_training start --all

    # start a single backbone
    python -m scripts.manage_training start --backbone resnet50

    # graceful stop (trainer exits at the next epoch boundary)
    python -m scripts.manage_training stop

    # print status once
    python -m scripts.manage_training status

    # print status every N minutes while training runs
    python -m scripts.manage_training watch --interval 10

After a PC shutdown, simply run ``start --all`` again; any
interrupted backbone resumes from its last checkpoint.
"""

from __future__ import annotations

import datetime
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from configs import load_merged_config
from training.classification_trainer import (
    ClassificationTrainingConfig,
    train_classification_model,
)


LOGGER = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

STATE_FILE = PROJECT_ROOT / "outputs" / "training_state.json"

STOP_DIRECTORY = PROJECT_ROOT / "outputs" / "training_state"

CONFIG_DIRECTORY = PROJECT_ROOT / "configs"

STATE_VERSION = 1

BASELINE_BACKBONES = (
    "densenet121",
    "resnet50",
    "efficientnet_v2_s",
    "convnext_tiny",
    "swin_t",
    "vit_b_16",
)

DEFAULT_TOTAL_EPOCHS = 30

# Weighting of the project completion milestones (sums to 1.0).
PROJECT_WEIGHTS = {
    "segmentation": 0.15,
    "baselines": 0.60,
    "ssl": 0.10,
    "evaluations": 0.10,
    "xai": 0.05,
}


@dataclass
class BackboneProgress:
    backbone: str
    status: str = "pending"  # pending | running | done | stopped | failed
    total_epochs: int = DEFAULT_TOTAL_EPOCHS
    epochs_done: int = 0
    best_accuracy: float | None = None
    started_at: str | None = None
    finished_at: str | None = None
    last_update: str | None = None


def now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def load_state(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path or STATE_FILE)

    if not path.exists():
        return _fresh_state()

    with path.open("r", encoding="utf-8") as file:
        state = json.load(file)

    if state.get("version") != STATE_VERSION:
        state = _fresh_state()

    _ensure_state_shape(state)

    return state


def save_state(
    state: dict[str, Any],
    path: str | Path | None = None,
) -> None:
    path = Path(path or STATE_FILE)

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(state, file, indent=2)


def _fresh_state() -> dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "created_at": now_iso(),
        "project": {
            "segmentation_trained": False,
            "ssl_pretrained": False,
            "evals_done": [],
            "xai_done": False,
        },
        "backbones": {
            name: asdict(BackboneProgress(backbone=name))
            for name in BASELINE_BACKBONES
        },
    }


def _ensure_state_shape(state: dict[str, Any]) -> None:
    state.setdefault("project", {})
    state.setdefault("backbones", {})

    for name in BASELINE_BACKBONES:
        if name not in state["backbones"]:
            state["backbones"][name] = asdict(
                BackboneProgress(backbone=name)
            )

    for key in (
        "segmentation_trained",
        "ssl_pretrained",
        "xai_done",
        "evals_done",
    ):
        state["project"].setdefault(
            key,
            [] if key == "evals_done" else False,
        )


def progress_file_for(backbone: str) -> Path:
    return (
        PROJECT_ROOT
        / "outputs"
        / f"{backbone}_training"
        / "progress.json"
    )


def stop_flag_for(backbone: str) -> Path:
    return STOP_DIRECTORY / f"{backbone}.stop"


def config_file_for(backbone: str) -> Path:
    return CONFIG_DIRECTORY / f"{backbone}.yaml"


def history_file_for(backbone: str) -> Path:
    return (
        PROJECT_ROOT
        / "outputs"
        / f"{backbone}_training"
        / "training_history.json"
    )


def detection_epochs_done(backbone: str) -> int:
    """Infer completed epochs from the saved training history."""

    history_file = history_file_for(backbone)

    if not history_file.exists():
        return 0

    try:
        with history_file.open("r", encoding="utf-8") as file:
            history = json.load(file)

        return len(history)
    except (json.JSONDecodeError, OSError):
        return 0


def is_backbone_done(state: dict[str, Any], backbone: str) -> bool:
    entry = state["backbones"][backbone]

    return (
        entry.get("status") == "done"
        or entry.get("epochs_done", 0) >= entry.get("total_epochs", 1)
    )


def _best_accuracy_from_history(backbone: str) -> float | None:
    history_file = history_file_for(backbone)

    if not history_file.exists():
        return None

    try:
        with history_file.open("r", encoding="utf-8") as file:
            history = json.load(file)

        if not history:
            return None

        return max(
            float(entry.get("validation_accuracy", 0.0))
            for entry in history
        )
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return None


def init_state() -> dict[str, Any]:
    """Create the state file, detecting existing checkpoints."""

    state = load_state()

    for backbone in BASELINE_BACKBONES:
        entry = state["backbones"][backbone]

        epochs_done = detection_epochs_done(backbone)

        entry["epochs_done"] = epochs_done

        entry["best_accuracy"] = _best_accuracy_from_history(backbone)

        total_epochs = int(entry.get("total_epochs", DEFAULT_TOTAL_EPOCHS))

        if epochs_done >= total_epochs:
            entry["status"] = "done"
        elif epochs_done > 0:
            entry["status"] = "stopped"
        else:
            entry["status"] = "pending"

    evals_done = []

    for backbone in BASELINE_BACKBONES:
        metrics_file = (
            PROJECT_ROOT / "outputs" / f"{backbone}_evaluation" / "evaluation_metrics.json"
        )

        if metrics_file.exists():
            evals_done.append(backbone)

    state["project"]["evals_done"] = evals_done

    state["project"]["segmentation_trained"] = (
        PROJECT_ROOT / "checkpoints" / "segmentation_unet3d" / "best.pt"
    ).exists()

    ssl_checkpoints = list(
        (PROJECT_ROOT / "checkpoints" / "self_supervised").glob("*/best.pt")
    )

    state["project"]["ssl_pretrained"] = len(ssl_checkpoints) > 0

    save_state(state)

    return state


def compute_backbone_percentage(
    state: dict[str, Any],
    backbone: str,
) -> float:
    """Fraction (0..1) of the target epochs completed."""

    entry = state["backbones"].get(backbone, {})

    total = int(entry.get("total_epochs", 1))

    done = int(entry.get("epochs_done", 0))

    return min(done / total, 1.0)


def compute_project_percentage(state: dict[str, Any]) -> dict[str, Any]:
    """Project completion by weighted milestones (sums to ~100%)."""

    project = state["project"]

    segmentation = (
        PROJECT_WEIGHTS["segmentation"] if project.get("segmentation_trained") else 0.0
    )

    baseline_weight = PROJECT_WEIGHTS["baselines"] / len(BASELINE_BACKBONES)

    baselines = sum(
        baseline_weight * compute_backbone_percentage(state, name)
        for name in BASELINE_BACKBONES
    )

    ssl = PROJECT_WEIGHTS["ssl"] if project.get("ssl_pretrained") else 0.0

    evaluations = (
        PROJECT_WEIGHTS["evaluations"]
        * len(project.get("evals_done", []))
        / len(BASELINE_BACKBONES)
    )

    xai = PROJECT_WEIGHTS["xai"] if project.get("xai_done") else 0.0

    total = segmentation + baselines + ssl + evaluations + xai

    return {
        "total": total,
        "segmentation": segmentation,
        "baselines": baselines,
        "ssl": ssl,
        "evaluations": evaluations,
        "xai": xai,
    }


def read_live_progress(backbone: str) -> dict[str, Any] | None:
    """Read the per-epoch progress file written by the trainer."""

    progress_file = progress_file_for(backbone)

    if not progress_file.exists():
        return None

    try:
        with progress_file.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return None


def _progress_is_fresh(progress: dict[str, Any]) -> bool:
    """A progress file is live only if updated recently."""

    updated_at = progress.get("updated_at")

    if not updated_at:
        return False

    try:
        timestamp = datetime.datetime.fromisoformat(str(updated_at))

        return (
            datetime.datetime.now() - timestamp
        ) < datetime.timedelta(minutes=30)
    except ValueError:
        return False


def sync_state_from_disk(state: dict[str, Any]) -> dict[str, Any]:
    """Refresh per-backbone state from history + progress files.

    Called while training runs so ``watch`` sees fresh numbers.
    """

    for backbone in BASELINE_BACKBONES:
        entry = state["backbones"][backbone]

        live = read_live_progress(backbone)

        if live is not None and _progress_is_fresh(live):
            entry["epochs_done"] = int(live.get("epoch", 0))
            entry["total_epochs"] = int(live.get("total_epochs", DEFAULT_TOTAL_EPOCHS))
            entry["best_accuracy"] = live.get("best_accuracy")
            entry["last_update"] = live.get("updated_at")

            if entry["status"] not in ("done", "stopped"):
                entry["status"] = "running"

            stop_flag = stop_flag_for(backbone)

            if stop_flag.exists():
                entry["status"] = "stopped"
        else:
            history_done = detection_epochs_done(backbone)

            if history_done > entry.get("epochs_done", 0):
                entry["epochs_done"] = history_done

            if entry.get("status") == "running":
                entry["status"] = "stopped"

    return state


def run_backbone(
    backbone: str,
    state: dict[str, Any],
    state_path: str | Path,
) -> None:
    """Train one backbone, resuming from its last checkpoint."""

    entry = state["backbones"][backbone]

    if is_backbone_done(state, backbone):
        LOGGER.info("%s already complete; skipping.", backbone)
        return

    config_path = config_file_for(backbone)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration missing for {backbone}: {config_path}"
        )

    config = load_merged_config(
        ClassificationTrainingConfig(),
        config_path,
        sections=("model", "data", "training", "paths"),
    )

    last_checkpoint = (
        PROJECT_ROOT / "checkpoints" / "classification" / backbone / "last.pt"
    )
    best_checkpoint = (
        PROJECT_ROOT / "checkpoints" / "classification" / backbone / "best.pt"
    )

    if last_checkpoint.exists():
        config.resume_from = str(last_checkpoint)
    elif best_checkpoint.exists():
        config.resume_from = str(best_checkpoint)
    else:
        config.resume_from = None

    config.stop_flag_path = str(stop_flag_for(backbone))

    config.progress_file = str(progress_file_for(backbone))

    entry["status"] = "running"
    entry["started_at"] = entry.get("started_at") or now_iso()

    save_state(state, state_path)

    LOGGER.info("=" * 78)
    LOGGER.info("Training: %s", backbone)
    LOGGER.info("Resume checkpoint: %s", config.resume_from)
    LOGGER.info("Stop flag: %s", config.stop_flag_path)
    LOGGER.info("=" * 78)

    try:
        train_classification_model(config)
    except KeyboardInterrupt:
        entry["status"] = "stopped"
        entry["finished_at"] = now_iso()
        save_state(state, state_path)
        LOGGER.info("%s interrupted; state saved.", backbone)
        raise
    except Exception:
        entry["status"] = "failed"
        entry["finished_at"] = now_iso()
        save_state(state, state_path)
        raise

    if stop_flag_for(backbone).exists():
        entry["status"] = "stopped"
    else:
        entry["status"] = "done"

    entry["epochs_done"] = detection_epochs_done(backbone)
    # Read best accuracy from training history if available
    history_path = Path(f"outputs/{backbone}_training/training_history.json")
    if history_path.exists():
        try:
            with open(history_path) as f:
                history = json.load(f)
            if history:
                entry["best_accuracy"] = max(h.get("validation_accuracy", 0) for h in history)
        except Exception:
            pass
    entry["finished_at"] = now_iso()

    save_state(state, state_path)


def run_all(
    state_path: str | Path | None = None,
    backbones: tuple[str, ...] = BASELINE_BACKBONES,
) -> None:
    """Train every incomplete backbone sequentially."""

    state_path = Path(state_path or STATE_FILE)

    state = load_state(state_path)

    for backbone in backbones:
        if backbone not in BASELINE_BACKBONES:
            raise ValueError(
                f"Unknown backbone: {backbone}. "
                f"Supported: {BASELINE_BACKBONES}"
            )

        run_backbone(backbone, state, state_path)

    LOGGER.info("All requested backbone trainings finished.")

    print_status(state)


def stop_all(state_path: str | Path | None = None) -> None:
    """Request a graceful stop of every running training.

    The trainer checks its stop flag after each epoch and exits
    cleanly, saving ``last.pt`` + history.
    """

    state_path = Path(state_path or STATE_FILE)

    STOP_DIRECTORY.mkdir(parents=True, exist_ok=True)

    state = load_state(state_path)

    for backbone in BASELINE_BACKBONES:
        entry = state["backbones"][backbone]

        if entry.get("status") == "running":
            stop_flag_for(backbone).touch()

            LOGGER.info("Stop requested for %s.", backbone)

            entry["status"] = "stopping"
        else:
            LOGGER.info("%s is not running; nothing to stop.", backbone)

    save_state(state, state_path)

    LOGGER.info(
        "Stop flags written to %s. Running trainers exit at the "
        "next epoch boundary and save their state.",
        STOP_DIRECTORY,
    )


def clear_stop_flags(state_path: str | Path | None = None) -> None:
    """Remove stop flags so paused trainings can resume."""

    state_path = Path(state_path or STATE_FILE)

    state = load_state(state_path)

    for backbone in BASELINE_BACKBONES:
        stop_flag = stop_flag_for(backbone)

        if stop_flag.exists():
            stop_flag.unlink()

            entry = state["backbones"][backbone]

            if entry.get("status") in ("stopped", "stopping"):
                entry["status"] = "pending"

    save_state(state, state_path)

    LOGGER.info("Stop flags cleared.")


def _format_line(
    backbone: str,
    entry: dict[str, Any],
) -> str:
    total = int(entry.get("total_epochs", 1))
    done = int(entry.get("epochs_done", 0))

    status = str(entry.get("status", "pending")).upper()

    accuracy = entry.get("best_accuracy")

    accuracy_text = f"{accuracy * 100:.2f}%" if accuracy is not None else "-"

    percent = 100.0 * min(done / total, 1.0)

    return (
        f"  {backbone:<20} {status:<10} {done:>3}/{total:<3} "
        f"({percent:5.1f}%)  best {accuracy_text}"
    )


def print_status(state: dict[str, Any]) -> None:
    """Print the project + per-backbone progress table."""

    state = sync_state_from_disk(state)

    project = compute_project_percentage(state)

    print("=" * 78)
    print("BRAIN TUMOR DIAGNOSIS - TRAINING STATUS")
    print("=" * 78)
    print(f"  Project completion : {project['total'] * 100:5.1f}%")
    print("-" * 78)
    print("  Milestones")
    print(f"    Segmentation training : {project['segmentation'] * 100:5.1f}%")
    print(f"    Classification baselines: {project['baselines'] * 100:5.1f}%")
    print(f"    Self-supervised pretrain: {project['ssl'] * 100:5.1f}%")
    print(f"    Test-set evaluations   : {project['evaluations'] * 100:5.1f}%")
    print(f"    Grad-CAM figures       : {project['xai'] * 100:5.1f}%")
    print("-" * 78)
    print("  Backbone training progress")

    for backbone in BASELINE_BACKBONES:
        print(_format_line(backbone, state["backbones"][backbone]))

    print("=" * 78)


def watch(
    interval_minutes: float = 10.0,
    state_path: str | Path | None = None,
    iterations: int | None = None,
) -> None:
    """Print the status table every ``interval_minutes`` minutes."""

    state_path = Path(state_path or STATE_FILE)

    interval_seconds = max(interval_minutes * 60.0, 30.0)

    LOGGER.info(
        "Watching training progress every %.1f minutes (Ctrl+C to stop).",
        interval_minutes,
    )

    iteration = 0

    try:
        while iterations is None or iteration < iterations:
            state = load_state(state_path)

            print_status(state)

            iteration += 1

            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        LOGGER.info("Watch stopped.")
        sys.exit(0)