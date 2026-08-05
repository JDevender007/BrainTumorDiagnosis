"""
==============================================================================
BraTS 2024 Dataset Loader

Author: Devender J.
Project: Brain Tumor Diagnosis using Hybrid Vision Transformer
==============================================================================

Supports
--------
• BraTS 2024
• Multi-modal MRI
• .nii
• .nii.gz
• PyTorch Dataset
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

# ==============================================================================
# Logger
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

LOGGER = logging.getLogger(__name__)

# ==============================================================================
# BraTS Modalities
# ==============================================================================

MODALITIES = (
    "t1n",
    "t1c",
    "t2w",
    "t2f",
)

SEGMENTATION = "seg"

SUPPORTED_EXTENSIONS = (
    ".nii",
    ".nii.gz",
)

# ==============================================================================
# Dataset Configuration
# ==============================================================================


@dataclass(slots=True)
class DatasetConfig:
    """
    Dataset configuration.
    """

    dataset_root: Path

    transform: Optional[Callable] = None

    use_segmentation: bool = True

    image_dtype: np.dtype = np.float32

    mask_dtype: np.dtype = np.uint8

    verbose: bool = True


# ==============================================================================
# BraTS Dataset
# ==============================================================================


class BraTSDataset(Dataset):
    """
    BraTS 2024 Dataset Loader.

    Expected Folder Structure

    data/
        raw/
            BraTS2024/
                BraTS-GLI-00001-000/
                    BraTS-GLI-00001-000-t1n.nii
                    BraTS-GLI-00001-000-t1c.nii
                    BraTS-GLI-00001-000-t2w.nii
                    BraTS-GLI-00001-000-t2f.nii
                    BraTS-GLI-00001-000-seg.nii
    """

    def __init__(
        self,
        config: DatasetConfig,
    ) -> None:

        super().__init__()

        self.config = config

        self.root = Path(config.dataset_root)

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset folder not found:\n{self.root}")

        self.patient_dirs = self._discover_patients()

        if len(self.patient_dirs) == 0:
            raise RuntimeError("No patient folders found.")

        if self.config.verbose:
            LOGGER.info("BraTS Dataset Loaded Successfully")
            LOGGER.info(
                "Patients Found : %d",
                len(self.patient_dirs),
            )

    # ==========================================================================
    # Dataset Length
    # ==========================================================================

    def __len__(self) -> int:
        return len(self.patient_dirs)

    # ==========================================================================
    # Discover Patients
    # ==========================================================================

    def _discover_patients(
        self,
    ) -> List[Path]:

        patients = sorted([folder for folder in self.root.iterdir() if folder.is_dir()])

        return patients

    # ==========================================================================
    # Locate MRI File
    # ==========================================================================

    @staticmethod
    def _find_file(
        patient_dir: Path,
        suffix: str,
    ) -> Path:
        """
        Finds either .nii or .nii.gz file.
        """

        nii_file = list(patient_dir.glob(f"*{suffix}.nii"))

        if len(nii_file) == 1:
            return nii_file[0]

        nii_gz_file = list(patient_dir.glob(f"*{suffix}.nii.gz"))

        if len(nii_gz_file) == 1:
            return nii_gz_file[0]

        raise FileNotFoundError(f"{suffix} not found in\n{patient_dir}")

    # ==========================================================================
    # Validate Patient
    # ==========================================================================

    def _validate_patient(
        self,
        patient_dir: Path,
    ) -> None:
        """
        Ensures every MRI modality exists.
        """

        for modality in MODALITIES:

            self._find_file(
                patient_dir,
                modality,
            )

        if self.config.use_segmentation:

            self._find_file(
                patient_dir,
                SEGMENTATION,
            )

    # ==========================================================================
    # Validate Entire Dataset
    # ==========================================================================

    def validate_dataset(self) -> None:
        """
        Validate every patient folder.
        """

        LOGGER.info("Validating dataset...")

        valid = 0

        for patient in self.patient_dirs:

            try:

                self._validate_patient(patient)

                valid += 1

            except Exception as error:

                LOGGER.warning(
                    "%s\n%s",
                    patient.name,
                    error,
                )

        LOGGER.info(
            "Valid Patients : %d / %d",
            valid,
            len(self.patient_dirs),
        )

    # ==========================================================================
    # Load NIfTI File
    # ==========================================================================

    def _load_nifti(
        self,
        file_path: Path,
    ) -> np.ndarray:
        """
        Load a NIfTI volume.
        """

        image = nib.load(str(file_path))

        volume = image.get_fdata()

        volume = volume.astype(self.config.image_dtype)

        return volume

    # ==========================================================================
    # Z-Score Normalization
    # ==========================================================================

    @staticmethod
    def _normalize(
        volume: np.ndarray,
    ) -> np.ndarray:
        """
        Z-score normalization.
        """

        mask = volume > 0

        if np.sum(mask) == 0:
            return volume

        mean = volume[mask].mean()

        std = volume[mask].std()

        if std < 1e-8:
            std = 1.0

        volume[mask] = (volume[mask] - mean) / std

        return volume

    # ==========================================================================
    # Load Single MRI Modality
    # ==========================================================================

    def _load_modality(
        self,
        patient_dir: Path,
        modality: str,
    ) -> np.ndarray:

        file_path = self._find_file(
            patient_dir,
            modality,
        )

        volume = self._load_nifti(file_path)

        volume = self._normalize(volume)

        return volume

    # ==========================================================================
    # Load Segmentation Mask
    # ==========================================================================

    def _load_segmentation(
        self,
        patient_dir: Path,
    ) -> np.ndarray:

        file_path = self._find_file(
            patient_dir,
            SEGMENTATION,
        )

        mask = nib.load(str(file_path)).get_fdata()

        mask = mask.astype(self.config.mask_dtype)

        return mask

    # ==========================================================================
    # Load All MRI Modalities
    # ==========================================================================

    def _load_modalities(
        self,
        patient_dir: Path,
    ) -> Dict[str, np.ndarray]:

        images = {}

        for modality in MODALITIES:

            images[modality] = self._load_modality(
                patient_dir,
                modality,
            )

        return images

    # ==========================================================================
    # Stack MRI Channels
    # ==========================================================================

    @staticmethod
    def _stack_modalities(
        images: Dict[str, np.ndarray],
    ) -> np.ndarray:

        stacked = np.stack(
            [
                images["t1n"],
                images["t1c"],
                images["t2w"],
                images["t2f"],
            ],
            axis=0,
        )

        return stacked

    # ==========================================================================
    # Validate Shapes
    # ==========================================================================

    @staticmethod
    def _validate_shapes(
        image: np.ndarray,
        mask: Optional[np.ndarray],
    ) -> None:

        if image.ndim != 4:

            raise RuntimeError(f"Expected 4D image. Got {image.shape}")

        if mask is not None:

            if image.shape[1:] != mask.shape:

                raise RuntimeError(
                    "Image and mask shape mismatch.\n"
                    f"Image : {image.shape}\n"
                    f"Mask  : {mask.shape}"
                )

    # ==========================================================================
    # Patient Information
    # ==========================================================================

    @staticmethod
    def _patient_id(
        patient_dir: Path,
    ) -> str:

        return patient_dir.name

    # ==========================================================================
    # Load Complete Patient
    # ==========================================================================

    def _load_patient(
        self,
        patient_dir: Path,
    ) -> Dict:

        self._validate_patient(patient_dir)

        images = self._load_modalities(patient_dir)

        image = self._stack_modalities(images)

        mask = None

        if self.config.use_segmentation:

            mask = self._load_segmentation(patient_dir)

        self._validate_shapes(
            image,
            mask,
        )

        return {
            "patient_id": self._patient_id(patient_dir),
            "image": image,
            "mask": mask,
        }

    # ==========================================================================
    # Apply Transform
    # ==========================================================================

    def _apply_transform(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray],
    ):

        if self.config.transform is None:
            return image, mask

        transformed = self.config.transform(
            image=image,
            mask=mask,
        )

        image = transformed["image"]

        if mask is not None:
            mask = transformed["mask"]

        return image, mask

    # ==========================================================================
    # Convert to Tensor
    # ==========================================================================

    @staticmethod
    def _to_tensor(
        image: np.ndarray,
        mask: Optional[np.ndarray],
    ):

        image = torch.from_numpy(image).float()

        if mask is not None:

            mask = torch.from_numpy(mask).long()

        return image, mask

    # ==========================================================================
    # Get Single Sample
    # ==========================================================================

    def __getitem__(
        self,
        index: int,
    ) -> Dict[str, torch.Tensor]:

        patient_dir = self.patient_dirs[index]

        try:

            patient = self._load_patient(patient_dir)

            image = patient["image"]

            mask = patient["mask"]

            image, mask = self._apply_transform(
                image,
                mask,
            )

            image, mask = self._to_tensor(
                image,
                mask,
            )

            sample = {
                "patient_id": patient["patient_id"],
                "image": image,
                "mask": mask,
            }

            return sample

        except Exception as error:

            LOGGER.error(
                "Error loading %s",
                patient_dir.name,
            )

            LOGGER.exception(error)

            raise

    # ==========================================================================
    # Dataset Summary
    # ==========================================================================

    def summary(self) -> None:

        LOGGER.info("===================================")

        LOGGER.info("BraTS 2024 Dataset Summary")

        LOGGER.info("===================================")

        LOGGER.info(
            "Dataset Root : %s",
            self.root,
        )

        LOGGER.info(
            "Patients     : %d",
            len(self),
        )

        LOGGER.info(
            "Modalities   : %s",
            ", ".join(MODALITIES),
        )

        LOGGER.info(
            "Segmentation : %s",
            self.config.use_segmentation,
        )

        LOGGER.info("===================================")

    # ==========================================================================
    # Print Dataset
    # ==========================================================================

    def __repr__(self):

        return f"BraTSDataset(" f"patients={len(self)}, " f"root='{self.root}')"
