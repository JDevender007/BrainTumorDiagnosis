"""
Experiment manifest infrastructure for reproducible research.

Creates and manages experiment manifests that record:
- Random seed
- Python/PyTorch/CUDA versions
- GPU information
- Dataset version/hash
- Config file hash
- Git commit
- Training duration
- Final metrics
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch


@dataclass
class ExperimentManifest:
    """Manifest for a single experiment."""

    experiment_id: str
    experiment_name: str
    description: str = ""

    # Environment
    python_version: str = ""
    pytorch_version: str = ""
    cuda_version: str = ""
    gpu_name: str = ""
    platform: str = ""

    # Reproducibility
    seed: int = 42
    git_commit: str = ""
    config_hash: str = ""

    # Dataset
    dataset_name: str = ""
    dataset_hash: str = ""
    train_samples: int = 0
    val_samples: int = 0
    test_samples: int = 0

    # Training
    start_time: str = ""
    end_time: str = ""
    duration_seconds: float = 0.0
    epochs_completed: int = 0
    best_epoch: int = 0

    # Architecture
    model_name: str = ""
    backbone: str = ""
    num_parameters: int = 0
    model_size_mb: float = 0.0

    # Metrics
    metrics: dict[str, Any] = field(default_factory=dict)

    # Custom fields
    custom: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path | str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: Path | str) -> "ExperimentManifest":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


def get_environment_info() -> dict[str, str]:
    """Collect environment information."""
    info = {
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "N/A",
        "platform": platform.platform(),
    }

    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
    else:
        info["gpu_name"] = "CPU"

    # Try to get git commit
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            timeout=5,
        )
        info["git_commit"] = result.stdout.strip()
    except Exception:
        info["git_commit"] = "unknown"

    return info


def hash_file(path: Path) -> str:
    """Compute MD5 hash of a file."""
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def hash_directory(directory: Path, extensions: set[str] | None = None) -> str:
    """Compute MD5 hash of all files in a directory."""
    if extensions is None:
        extensions = {".yaml", ".yml", ".json", ".py", ".txt"}

    hasher = hashlib.md5()
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in extensions:
            hasher.update(path.read_bytes())
    return hasher.hexdigest()


def create_experiment_manifest(
    experiment_id: str,
    experiment_name: str,
    config_path: Path | str | None = None,
    dataset_path: Path | str | None = None,
    **kwargs,
) -> ExperimentManifest:
    """Create a new experiment manifest with environment info."""
    env = get_environment_info()

    manifest = ExperimentManifest(
        experiment_id=experiment_id,
        experiment_name=experiment_name,
        start_time=datetime.now(timezone.utc).isoformat(),
        **env,
        **kwargs,
    )

    if config_path is not None:
        config_path = Path(config_path)
        if config_path.exists():
            manifest.config_hash = hash_file(config_path)

    return manifest


class ExperimentRegistry:
    """Registry for managing multiple experiment manifests."""

    def __init__(self, registry_path: Path | str = "experiments/registry.json"):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self):
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                self.experiments = json.load(f)
        else:
            self.experiments = {}

    def _save(self):
        with open(self.registry_path, "w") as f:
            json.dump(self.experiments, f, indent=2, default=str)

    def register(self, manifest: ExperimentManifest):
        self.experiments[manifest.experiment_id] = manifest.to_dict()
        self._save()

    def get(self, experiment_id: str) -> dict[str, Any] | None:
        return self.experiments.get(experiment_id)

    def list_experiments(self) -> list[str]:
        return list(self.experiments.keys())

    def get_comparison_table(self, experiment_ids: list[str] | None = None) -> dict[str, Any]:
        """Get a comparison table of experiments."""
        if experiment_ids is None:
            experiment_ids = list(self.experiments.keys())

        table = {}
        for eid in experiment_ids:
            if eid in self.experiments:
                exp = self.experiments[eid]
                table[eid] = {
                    "name": exp.get("experiment_name", ""),
                    "backbone": exp.get("backbone", ""),
                    "seed": exp.get("seed", 0),
                    "metrics": exp.get("metrics", {}),
                }
        return table
