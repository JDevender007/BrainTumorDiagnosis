"""
==============================================================================
BraTS 2024 Dataset Loader

Author  : Devender J.
Project : Hybrid Brain Tumor Diagnosis

Loads BraTS 2024 multimodal MRI data.

Modalities:
    T1 native
    T1 contrast enhanced
    T2 weighted
    T2 FLAIR

Segmentation:
    BraTS segmentation mask
==============================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

LOGGER = logging.getLogger(__name__)


# ==============================================================================
# Configuration
# ==============================================================================


@dataclass(slots=True)
class DatasetConfig:
    dataset_root: str

    modalities: Tuple[str, ...] = (
        "t1n",
        "t1c",
        "t2w",
        "t2f",
    )

    segmentation_suffix: str = "seg"

    image_dtype: np.dtype = np.float32

    mask_dtype: np.dtype = np.uint8


# ==============================================================================
# BraTS Dataset
# ==============================================================================


class BraTSDataset(Dataset):
    """
    PyTorch Dataset for BraTS 2024.

    Expected directory structure:

    data/
        raw/
            BraTS2024/
                BraTS-GLI-xxxxx-xxx/
                    *_t1n.nii
                    *_t1c.nii
                    *_t2w.nii
                    *_t2f.nii
                    *_seg.nii
    """

    def __init__(
        self,
        config: DatasetConfig,
        transform: Optional[Callable] = None,
    ) -> None:

        self.config = config
        self.transform = transform

        self.dataset_root = Path(config.dataset_root)

        if not self.dataset_root.exists():
            raise FileNotFoundError(
                f"Dataset directory not found:\n{self.dataset_root}"
            )

        if not self.dataset_root.is_dir():
            raise NotADirectoryError(
                f"Dataset root is not a directory:\n{self.dataset_root}"
            )

        LOGGER.info("Loading BraTS Dataset...")

        self.patient_dirs = self._discover_patients()

        if not self.patient_dirs:
            raise RuntimeError(f"No patient directories found in:\n{self.dataset_root}")

        LOGGER.info("BraTS Dataset Loaded Successfully")
        LOGGER.info("Patients Found : %d", len(self.patient_dirs))

    # ==========================================================================
    # Discover Patients
    # ==========================================================================

    def _discover_patients(self) -> List[Path]:

        patients = []

        for path in sorted(self.dataset_root.iterdir()):

            if not path.is_dir():
                continue

            if path.name.startswith("."):
                continue

            patients.append(path)

        return patients

    # ==========================================================================
    # Find File
    # ==========================================================================

    @staticmethod
    def _find_file(
        patient_dir: Path,
        suffix: str,
    ) -> Path:

        patterns = [
            f"*_{suffix}.nii",
            f"*_{suffix}.nii.gz",
            f"*-{suffix}.nii",
            f"*-{suffix}.nii.gz",
            f"*{suffix}.nii",
            f"*{suffix}.nii.gz",
        ]

        matches: List[Path] = []

        for pattern in patterns:
            matches.extend(patient_dir.glob(pattern))

        unique_matches = sorted(set(matches))

        if not unique_matches:
            raise FileNotFoundError(f"{suffix} not found in\n{patient_dir}")

        return unique_matches[0]

    # ==========================================================================
    # Validate Patient
    # ==========================================================================

    def _validate_patient(
        self,
        patient_dir: Path,
    ) -> None:

        for modality in self.config.modalities:

            self._find_file(
                patient_dir,
                modality,
            )

        self._find_file(
            patient_dir,
            self.config.segmentation_suffix,
        )

    # ==========================================================================
    # Load NIfTI
    # ==========================================================================

    @staticmethod
    def _load_nifti(
        file_path: Path,
    ) -> np.ndarray:

        image = nib.load(str(file_path))

        data = image.get_fdata(dtype=np.float32)

        return np.asarray(
            data,
            dtype=np.float32,
        )

    # ==========================================================================
    # Load Patient
    # ==========================================================================

    def _load_patient(
        self,
        patient_dir: Path,
    ) -> Tuple[np.ndarray, np.ndarray]:

        self._validate_patient(patient_dir)

        modalities = []

        for modality in self.config.modalities:

            file_path = self._find_file(
                patient_dir,
                modality,
            )

            volume = self._load_nifti(file_path)

            modalities.append(volume)

        image = np.stack(
            modalities,
            axis=0,
        ).astype(
            np.float32,
            copy=False,
        )

        segmentation_path = self._find_file(
            patient_dir,
            self.config.segmentation_suffix,
        )

        mask = self._load_nifti(segmentation_path)

        mask = np.asarray(
            mask,
            dtype=np.uint8,
        )

        return image, mask

    # ==========================================================================
    # Convert To Tensor
    # ==========================================================================

    @staticmethod
    def _to_tensor(
        image,
        mask,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if isinstance(image, torch.Tensor):

            image = image.float()

        else:

            image = torch.as_tensor(
                np.asarray(image),
                dtype=torch.float32,
            )

        if isinstance(mask, torch.Tensor):

            mask = mask.long()

        else:

            mask = torch.as_tensor(
                np.asarray(mask),
                dtype=torch.long,
            )

        return image, mask

    # ==========================================================================
    # Normalize Mask
    # ==========================================================================

    @staticmethod
    def _normalize_mask(
        mask: torch.Tensor,
    ) -> torch.Tensor:

        mask = mask.long()

        return mask

    # ==========================================================================
    # Get Item
    # ==========================================================================

    def __getitem__(
        self,
        index: int,
    ) -> Dict[str, torch.Tensor]:

        patient_dir = self.patient_dirs[index]

        try:

            image, mask = self._load_patient(patient_dir)

            if self.transform is not None:

                sample = self.transform(
                    image,
                    mask,
                )

                if isinstance(sample, dict):

                    image = sample["image"]
                    mask = sample["mask"]

                else:

                    raise TypeError(
                        "Transform must return a dictionary "
                        "containing 'image' and 'mask'."
                    )

            image, mask = self._to_tensor(
                image,
                mask,
            )

            mask = self._normalize_mask(mask)

            return {
                "image": image,
                "mask": mask,
                "patient_id": patient_dir.name,
            }

        except Exception as exc:

            LOGGER.error(
                "Error loading %s",
                patient_dir.name,
            )

            LOGGER.error(
                "%s",
                exc,
            )

            raise

    # ==========================================================================
    # Length
    # ==========================================================================

    def __len__(self) -> int:

        return len(self.patient_dirs)

    # ==========================================================================
    # Dataset Summary
    # ==========================================================================

    def summary(self) -> None:

        LOGGER.info("=" * 60)
        LOGGER.info("BraTS Dataset Summary")
        LOGGER.info("=" * 60)

        LOGGER.info(
            "Dataset Root : %s",
            self.dataset_root,
        )

        LOGGER.info(
            "Patients     : %d",
            len(self.patient_dirs),
        )

        LOGGER.info(
            "Modalities   : %s",
            ", ".join(self.config.modalities),
        )

        LOGGER.info(
            "Segmentation : %s",
            self.config.segmentation_suffix,
        )

        LOGGER.info("=" * 60)


# ==============================================================================
# Factory
# ==============================================================================


class BraTSDatasetFactory:

    @staticmethod
    def create(
        dataset_root: str,
        transform: Optional[Callable] = None,
    ) -> BraTSDataset:

        config = DatasetConfig(dataset_root=dataset_root)

        return BraTSDataset(
            config=config,
            transform=transform,
        )


# ==============================================================================
# Test
# ==============================================================================


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    dataset_root = Path(__file__).resolve().parents[2] / "data" / "raw" / "BraTS2024"

    dataset = BraTSDatasetFactory.create(dataset_root=str(dataset_root))

    dataset.summary()

    print()
    print("Dataset length:", len(dataset))

    sample = dataset[0]

    print()
    print("Patient ID:")
    print(sample["patient_id"])

    print()
    print("Image shape:")
    print(sample["image"].shape)

    print()
    print("Mask shape:")
    print(sample["mask"].shape)

    print()
    print("Image dtype:")
    print(sample["image"].dtype)

    print()
    print("Mask dtype:")
    print(sample["mask"].dtype)
