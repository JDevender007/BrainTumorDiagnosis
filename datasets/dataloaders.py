"""
BrainTumorDiagnosis
Shared DataLoader utilities.

Central place for constructing torch DataLoader objects so the
trainers and inference code use identical worker / memory /
reproducibility settings.
"""

from __future__ import annotations

import random
from typing import Sequence

import torch
from torch.utils.data import DataLoader, Dataset


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    drop_last: bool = False,
    persistent_workers: bool = False,
    seed: int | None = None,
) -> DataLoader:
    """
    Create a DataLoader with the project-wide defaults.

    Parameters:
        pin_memory:
            When None, defaults to ``torch.cuda.is_available()``.

        seed:
            When provided (and the loader is shuffled), the loader
            is constructed with a deterministic generator so
            reshuffling across runs is reproducible.
    """

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive: {batch_size}")

    if num_workers < 0:
        raise ValueError(f"num_workers must be non-negative: {num_workers}")

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    generator = None

    if shuffle and seed is not None:
        generator = torch.Generator()

        generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=(persistent_workers and num_workers > 0),
        generator=generator,
    )


def make_dataloaders(
    datasets: Sequence[Dataset] | dict[str, Dataset],
    batch_size: int,
    shuffle: Sequence[bool] | None = None,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    drop_last: bool = False,
    persistent_workers: bool = False,
    seed: int | None = None,
) -> list[DataLoader] | dict[str, DataLoader]:
    """
    Create loaders for several datasets with one configuration.

    Mirrors the structure of ``datasets``: lists become lists,
    dictionaries become dictionaries.

    ``shuffle`` may be a single bool (applied to all loaders) or
    a sequence matching the datasets.
    """

    items: Sequence[tuple[str, Dataset]] | list[Dataset]

    if isinstance(datasets, dict):
        names = list(datasets.keys())

        items = [(name, datasets[name]) for name in names]
    else:
        items = list(datasets)

    if shuffle is None:
        shuffle_flags: Sequence[bool] = [False] * len(items)
    elif isinstance(shuffle, bool):
        shuffle_flags = [shuffle] * len(items)
    else:
        shuffle_flags = list(shuffle)

        if len(shuffle_flags) != len(items):
            raise ValueError(
                "shuffle must match the number of datasets: "
                f"{len(shuffle_flags)} vs {len(items)}"
            )

    loaders = [
        make_dataloader(
            item[1] if isinstance(item, tuple) else item,
            batch_size=batch_size,
            shuffle=shuffle_flags[index],
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=persistent_workers,
            seed=seed,
        )
        for index, item in enumerate(items)
    ]

    if isinstance(datasets, dict):
        return {name: loader for (name, _), loader in zip(items, loaders)}

    return loaders


def worker_seed(worker_id: int) -> None:
    """
    Seed function for DataLoader workers.

    Ensures each worker uses a deterministic, distinct seed based
    on the global torch seed.
    """

    base_seed = torch.initial_seed() % (2**32)

    random.seed(base_seed + worker_id)

    import numpy as np

    np.random.seed(base_seed + worker_id)


def _test() -> None:
    from torch.utils.data import TensorDataset

    print("=" * 78)
    print("BrainTumorDiagnosis DataLoader Utilities Test")
    print("=" * 78)

    dataset = TensorDataset(
        torch.arange(32),
        torch.arange(32),
    )

    loader = make_dataloader(
        dataset,
        batch_size=8,
        shuffle=True,
        seed=42,
    )

    batches = list(loader)

    assert len(batches) == 4

    first = batches[0][0]

    assert first.shape == (8,)

    loaders = make_dataloaders(
        {
            "train": dataset,
            "val": dataset,
        },
        batch_size=8,
        shuffle=[True, False],
        seed=42,
    )

    assert isinstance(loaders, dict)

    assert set(loaders.keys()) == {"train", "val"}

    assert len(list(loaders["val"])) == 4

    print("Single loader batches : 4")
    print("Dict loaders          : train, val")

    print("=" * 78)
    print("DataLoader utilities test passed.")
    print("=" * 78)


if __name__ == "__main__":
    _test()
