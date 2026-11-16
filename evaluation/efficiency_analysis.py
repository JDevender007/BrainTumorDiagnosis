"""
Efficiency analysis for brain tumor classification models.

Measures:
- Parameter count
- Model size (MB)
- FLOPs/MACs
- GPU memory usage
- Inference latency
- FPS
- Accuracy vs Efficiency tradeoff
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class EfficiencyResult:
    model_name: str
    total_params: int
    trainable_params: int
    model_size_mb: float
    flops: float
    macs: float
    gpu_memory_mb: float
    latency_ms: float
    fps: float
    accuracy: float
    f1: float
    throughput_samples_per_sec: float


def count_parameters(model: nn.Module) -> tuple[int, int, float]:
    """Count parameters and model size."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    size_mb = size_bytes / (1024 * 1024)
    return total, trainable, size_mb


def estimate_flops(model: nn.Module, input_size: tuple = (1, 3, 224, 224)) -> tuple[float, float]:
    """Estimate FLOPs and MACs using hook-based counting."""
    flops_dict = {}
    macs_dict = {}

    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            batch_size = output.shape[0]
            out_features = np.prod(output.shape[1:])
            if hasattr(module, 'weight'):
                in_features = np.prod(module.weight.shape[1:])
                kernel_ops = np.prod(module.weight.shape[2:]) if len(module.weight.shape) > 2 else 1
                flops_dict[module] = 2 * batch_size * out_features * kernel_ops
                macs_dict[module] = batch_size * out_features * kernel_ops

    hooks = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.Conv1d)):
            hooks.append(module.register_forward_hook(hook_fn))

    dummy = torch.randn(*input_size)
    device = next(model.parameters()).device
    dummy = dummy.to(device)

    with torch.no_grad():
        model(dummy)

    for h in hooks:
        h.remove()

    total_flops = sum(flops_dict.values())
    total_macs = sum(macs_dict.values())

    return float(total_flops), float(total_macs)


def measure_latency(
    model: nn.Module,
    device: torch.device,
    num_runs: int = 100,
    input_size: tuple = (1, 3, 224, 224),
    warmup: int = 10,
) -> tuple[float, float]:
    """Measure inference latency."""
    model.eval()
    dummy = torch.randn(*input_size).to(device)

    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)

    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            model(dummy)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

    return float(np.mean(latencies)), float(np.std(latencies))


def measure_gpu_memory(
    model: nn.Module,
    device: torch.device,
    input_size: tuple = (1, 3, 224, 224),
) -> float:
    """Measure GPU memory usage."""
    if device.type != "cuda":
        return 0.0

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    model.eval()
    dummy = torch.randn(*input_size).to(device)

    with torch.no_grad():
        model(dummy)

    memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    return float(memory_mb)


def measure_throughput(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_batches: int = 50,
) -> float:
    """Measure throughput in samples/sec."""
    model.eval()
    total_samples = 0
    start_time = time.perf_counter()

    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= num_batches:
                break
            images = images.to(device)
            model(images)
            total_samples += images.size(0)

    elapsed = time.perf_counter() - start_time
    return total_samples / elapsed if elapsed > 0 else 0.0


def analyze_efficiency(
    model: nn.Module,
    model_name: str,
    dataloader: DataLoader,
    device: torch.device,
    accuracy: float = 0.0,
    f1: float = 0.0,
    input_size: tuple = (1, 3, 224, 224),
) -> EfficiencyResult:
    """Full efficiency analysis for a model."""
    total_params, trainable_params, size_mb = count_parameters(model)
    flops, macs = estimate_flops(model, input_size)
    latency_mean, latency_std = measure_latency(model, device, input_size=input_size)
    gpu_memory = measure_gpu_memory(model, device, input_size)
    throughput = measure_throughput(model, dataloader, device)
    fps = 1000.0 / latency_mean if latency_mean > 0 else 0.0

    return EfficiencyResult(
        model_name=model_name,
        total_params=total_params,
        trainable_params=trainable_params,
        model_size_mb=size_mb,
        flops=flops,
        macs=macs,
        gpu_memory_mb=gpu_memory,
        latency_ms=latency_mean,
        fps=fps,
        accuracy=accuracy,
        f1=f1,
        throughput_samples_per_sec=throughput,
    )


def generate_efficiency_comparison(
    results: list[EfficiencyResult],
    output_dir: Path,
):
    """Generate efficiency comparison table and plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    header = "| Model | Params (M) | Size (MB) | FLOPs (G) | Latency (ms) | FPS | GPU Mem (MB) | Accuracy |"
    separator = "|-------|------------|-----------|-----------|--------------|-----|--------------|----------|"
    rows = []
    for r in results:
        rows.append(
            f"| {r.model_name} | {r.total_params / 1e6:.2f} | {r.model_size_mb:.2f} | "
            f"{r.flops / 1e9:.2f} | {r.latency_ms:.2f} | {r.fps:.1f} | "
            f"{r.gpu_memory_mb:.1f} | {r.accuracy:.4f} |"
        )

    table = "\n".join([header, separator] + rows)
    with open(output_dir / "efficiency_table.md", "w") as f:
        f.write(table)

    with open(output_dir / "efficiency_results.json", "w") as f:
        json.dump([
            {
                "model": r.model_name,
                "total_params": r.total_params,
                "model_size_mb": r.model_size_mb,
                "flops": r.flops,
                "latency_ms": r.latency_ms,
                "fps": r.fps,
                "gpu_memory_mb": r.gpu_memory_mb,
                "accuracy": r.accuracy,
                "f1": r.f1,
            }
            for r in results
        ], f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        names = [r.model_name for r in results]
        accuracies = [r.accuracy for r in results]
        latencies = [r.latency_ms for r in results]
        sizes = [r.model_size_mb for r in results]

        axes[0].scatter(latencies, accuracies, s=100, c="steelblue", edgecolors="black")
        for i, name in enumerate(names):
            axes[0].annotate(name, (latencies[i], accuracies[i]), fontsize=8, ha="center", va="bottom")
        axes[0].set_xlabel("Latency (ms)")
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title("Accuracy vs Latency")
        axes[0].grid(True, alpha=0.3)

        axes[1].scatter(sizes, accuracies, s=100, c="coral", edgecolors="black")
        for i, name in enumerate(names):
            axes[1].annotate(name, (sizes[i], accuracies[i]), fontsize=8, ha="center", va="bottom")
        axes[1].set_xlabel("Model Size (MB)")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Accuracy vs Model Size")
        axes[1].grid(True, alpha=0.3)

        params = [r.total_params / 1e6 for r in results]
        axes[2].scatter(params, accuracies, s=100, c="forestgreen", edgecolors="black")
        for i, name in enumerate(names):
            axes[2].annotate(name, (params[i], accuracies[i]), fontsize=8, ha="center", va="bottom")
        axes[2].set_xlabel("Parameters (M)")
        axes[2].set_ylabel("Accuracy")
        axes[2].set_title("Accuracy vs Parameters")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "efficiency_analysis.png", dpi=150, bbox_inches="tight")
        plt.close()
    except ImportError:
        pass
