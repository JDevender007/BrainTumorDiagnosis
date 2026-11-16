from __future__ import annotations

import logging

import torch


logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """
    Select the best available computation device.

    Returns:
        torch.device: CUDA device when available, otherwise CPU.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA device: %s", torch.cuda.get_device_name(0))
        return device

    device = torch.device("cpu")
    logger.info("CUDA unavailable. Using CPU.")
    return device


def get_device_name(device: torch.device) -> str:
    """
    Return a human-readable device name.
    """
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)

    return "CPU"


def configure_cuda() -> None:
    """
    Configure CUDA runtime settings when CUDA is available.

    Benchmark mode stays disabled by default so it never conflicts
    with deterministic training (see utils.seed.set_seed).
    """
    if not torch.cuda.is_available():
        return

    logger.info(
        "CUDA configured. GPU: %s",
        torch.cuda.get_device_name(0),
    )


def get_cuda_memory() -> dict[str, float]:
    """
    Return current CUDA memory usage in MB.
    """
    if not torch.cuda.is_available():
        return {
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
        }

    return {
        "allocated_mb": torch.cuda.memory_allocated() / (1024**2),
        "reserved_mb": torch.cuda.memory_reserved() / (1024**2),
    }