from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


logger = logging.getLogger(__name__)


MRI_MODALITIES = ("t1n", "t1c", "t2w", "t2f")

_PATIENT_ID_SUFFIX = re.compile(
    r"^(?P<patient_id>.+)-\d+$",
)


def extract_patient_id(
    case_id: str,
) -> str:
    """
    Extract the base patient identifier from a BraTS case folder name.

    BraTS folder names follow the pattern ``<PATIENT>-<EXAM>``,
    for example ``BraTS-GLI-02062-100``.

    Multiple exams of the same patient share the same base identifier,
    so grouping by this value is required for patient-level splitting.

    When the name does not match the expected pattern, the full
    case identifier is returned unchanged.
    """
    match = _PATIENT_ID_SUFFIX.match(case_id)

    if match is None:
        return case_id

    return match.group("patient_id")


@dataclass(frozen=True)
class BraTSSample:
    case_id: str
    case_dir: Path
    t1n: Path
    t1c: Path
    t2w: Path
    t2f: Path
    segmentation: Path | None = None


@dataclass
class DataLoaderConfig:
    batch_size: int = 1
    num_workers: int = 0
    pin_memory: bool = False
    shuffle_train: bool = True
    drop_last: bool = False


def _find_file(case_dir: Path, modality: str) -> Path | None:
    case_id = case_dir.name

    candidates = [
        case_dir / f"{case_id}-{modality}.nii",
        case_dir / f"{case_id}-{modality}.nii.gz",
    ]

    for path in candidates:
        if path.is_file():
            return path

    matches = list(case_dir.glob(f"*{modality}.nii"))
    matches += list(case_dir.glob(f"*{modality}.nii.gz"))

    if matches:
        return matches[0]

    return None


def _find_segmentation(case_dir: Path) -> Path | None:
    case_id = case_dir.name

    candidates = [
        case_dir / f"{case_id}-seg.nii",
        case_dir / f"{case_id}-seg.nii.gz",
    ]

    for path in candidates:
        if path.is_file():
            return path

    matches = list(case_dir.glob("*-seg.nii"))
    matches += list(case_dir.glob("*-seg.nii.gz"))

    if matches:
        return matches[0]

    return None


def discover_brats_cases(
    root_dir: str | Path,
    require_segmentation: bool = True,
) -> list[BraTSSample]:
    root = Path(root_dir)

    if not root.exists():
        raise FileNotFoundError(f"BraTS directory does not exist: {root}")

    if not root.is_dir():
        raise NotADirectoryError(f"BraTS path is not a directory: {root}")

    logger.info("Searching BraTS cases in: %s", root)

    case_dirs = sorted(path for path in root.iterdir() if path.is_dir())

    logger.info(
        "Candidate case directories: %d",
        len(case_dirs),
    )

    cases: list[BraTSSample] = []

    for case_dir in case_dirs:
        modality_paths: dict[str, Path] = {}

        valid = True

        for modality in MRI_MODALITIES:
            path = _find_file(case_dir, modality)

            if path is None:
                logger.warning(
                    "Missing %s modality in case %s",
                    modality,
                    case_dir.name,
                )
                valid = False
                break

            modality_paths[modality] = path

        if not valid:
            continue

        segmentation = _find_segmentation(case_dir)

        if require_segmentation and segmentation is None:
            logger.warning(
                "Missing segmentation for case %s",
                case_dir.name,
            )
            continue

        cases.append(
            BraTSSample(
                case_id=case_dir.name,
                case_dir=case_dir,
                t1n=modality_paths["t1n"],
                t1c=modality_paths["t1c"],
                t2w=modality_paths["t2w"],
                t2f=modality_paths["t2f"],
                segmentation=segmentation,
            )
        )

    logger.info(
        "Valid BraTS cases discovered: %d",
        len(cases),
    )

    return cases


def load_nifti(
    path: str | Path,
    dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"NIfTI file does not exist: {path}")

    image = nib.load(str(path))
    data = image.get_fdata(dtype=dtype)

    return np.asarray(data)


def normalize_volume(
    volume: np.ndarray,
    clip_percentiles: tuple[float, float] | None = None,
) -> np.ndarray:
    """
    Normalize a BraTS MRI volume.

    Delegates to the canonical implementation in
    datasets.preprocessing.normalize_volume.

    Unlike the default, background voxels participate in the
    statistics (all finite voxels), preserving the original
    BraTS preprocessor behavior.
    """

    from datasets.preprocessing import normalize_volume as _canonical

    return _canonical(
        volume,
        clip_percentiles=clip_percentiles,
        mask_background=False,
    )


def preprocess_case(
    case: BraTSSample,
    return_segmentation: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    volumes = []

    for modality in MRI_MODALITIES:
        path = getattr(case, modality)

        volume = load_nifti(path)

        volume = normalize_volume(volume)

        volumes.append(volume)

    image = np.stack(volumes, axis=0)

    image_tensor = torch.from_numpy(image.astype(np.float32))

    if not return_segmentation:
        return image_tensor

    if case.segmentation is None:
        raise ValueError(f"Segmentation file missing for case {case.case_id}")

    segmentation = load_nifti(
        case.segmentation,
        dtype=np.float32,
    )

    segmentation_tensor = torch.from_numpy(segmentation.astype(np.int64))

    return image_tensor, segmentation_tensor


def preprocess_segmentation(
    segmentation: torch.Tensor,
    num_classes: int = 4,
) -> torch.Tensor:
    segmentation = segmentation.long()

    segmentation = torch.clamp(
        segmentation,
        min=0,
        max=num_classes - 1,
    )

    return segmentation


class BraTSDataset(Dataset):
    def __init__(
        self,
        cases: Sequence[BraTSSample],
        return_segmentation: bool = False,
    ) -> None:
        self.cases = list(cases)
        self.return_segmentation = return_segmentation

        if not self.cases:
            raise ValueError("BraTSDataset received zero cases.")

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, object]:
        case = self.cases[index]

        if self.return_segmentation:
            image, segmentation = preprocess_case(
                case,
                return_segmentation=True,
            )

            segmentation = preprocess_segmentation(segmentation)

            return {
                "image": image,
                "segmentation": segmentation,
                "case_id": case.case_id,
            }

        image = preprocess_case(
            case,
            return_segmentation=False,
        )

        return {
            "image": image,
            "case_id": case.case_id,
        }

    def __iter__(self) -> Iterator[dict[str, object]]:
        for index in range(len(self)):
            yield self[index]


def group_cases_by_patient(
    cases: Sequence[BraTSSample],
) -> dict[str, list[BraTSSample]]:
    """
    Group BraTS cases by base patient identifier.

    Returns:
        Mapping from patient_id to the list of their case folders.
    """
    groups: dict[str, list[BraTSSample]] = {}

    for case in cases:
        patient_id = extract_patient_id(case.case_id)
        groups.setdefault(patient_id, []).append(case)

    return groups


def split_brats_cases_by_patient(
    cases: Sequence[BraTSSample],
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[
    list[BraTSSample],
    list[BraTSSample],
    list[BraTSSample],
]:
    """
    Split BraTS cases at patient level.

    Every case folder of one patient is assigned to exactly one
    split, which prevents the same brain appearing in both the
    training and the test set.

    Raises:
        ValueError: when ratios do not sum to one or are non-positive.
    """
    if not cases:
        raise ValueError("Cannot split an empty BraTS case list.")

    total_ratio = train_ratio + validation_ratio + test_ratio

    if not np.isclose(total_ratio, 1.0):
        raise ValueError(
            "train_ratio + validation_ratio + test_ratio must equal 1.0"
        )

    if any(
        ratio <= 0
        for ratio in (
            train_ratio,
            validation_ratio,
            test_ratio,
        )
    ):
        raise ValueError("All split ratios must be greater than zero.")

    groups = group_cases_by_patient(cases)

    patient_ids = sorted(groups.keys())

    random_generator = random.Random(seed)
    random_generator.shuffle(patient_ids)

    total_patients = len(patient_ids)

    train_patient_count = int(total_patients * train_ratio)
    validation_patient_count = int(total_patients * validation_ratio)

    if train_patient_count == 0:
        train_patient_count = 1

    if validation_patient_count == 0:
        validation_patient_count = 1

    test_patient_count = total_patients - train_patient_count - validation_patient_count

    if test_patient_count <= 0:
        test_patient_count = 1
        validation_patient_count -= 1

    train_patients = patient_ids[:train_patient_count]

    validation_start = train_patient_count
    validation_end = validation_start + validation_patient_count

    validation_patients = patient_ids[validation_start:validation_end]

    test_patients = patient_ids[validation_end:]

    train_cases: list[BraTSSample] = []
    validation_cases: list[BraTSSample] = []
    test_cases: list[BraTSSample] = []

    for patient_id in train_patients:
        train_cases.extend(groups[patient_id])

    for patient_id in validation_patients:
        validation_cases.extend(groups[patient_id])

    for patient_id in test_patients:
        test_cases.extend(groups[patient_id])

    logger.info("BraTS dataset split (patient level):")
    logger.info("Train patients     : %d (%d cases)", len(train_patients), len(train_cases))
    logger.info(
        "Validation patients: %d (%d cases)",
        len(validation_patients),
        len(validation_cases),
    )
    logger.info("Test patients      : %d (%d cases)", len(test_patients), len(test_cases))

    return (
        train_cases,
        validation_cases,
        test_cases,
    )


def split_brats_cases(
    cases: Sequence[BraTSSample],
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    patient_level: bool = True,
) -> tuple[
    list[BraTSSample],
    list[BraTSSample],
    list[BraTSSample],
]:
    """
    Split BraTS cases into train, validation and test folds.

    Patient-level splitting is the default: all exam folders of one
    patient are kept together. Splitting by individual case folder
    can leak the same patient into multiple folds because BraTS
    patients frequently have several exam folders.

    Args:
        patient_level:
            Group by base patient identifier (default True).
            When False, an exam-folder split is performed.
    """
    if patient_level:
        return split_brats_cases_by_patient(
            cases,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )

    return _split_brats_cases_by_folder(
        cases,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )


def _split_brats_cases_by_folder(
    cases: Sequence[BraTSSample],
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[
    list[BraTSSample],
    list[BraTSSample],
    list[BraTSSample],
]:
    if not cases:
        raise ValueError("Cannot split an empty BraTS case list.")

    total_ratio = train_ratio + validation_ratio + test_ratio

    if not np.isclose(total_ratio, 1.0):
        raise ValueError(
            "train_ratio + validation_ratio + " "test_ratio must equal 1.0"
        )

    if any(
        ratio <= 0
        for ratio in (
            train_ratio,
            validation_ratio,
            test_ratio,
        )
    ):
        raise ValueError("All split ratios must be greater than zero.")

    shuffled = list(cases)

    random_generator = random.Random(seed)
    random_generator.shuffle(shuffled)

    total = len(shuffled)

    train_count = int(total * train_ratio)
    validation_count = int(total * validation_ratio)

    if train_count == 0:
        train_count = 1

    if validation_count == 0:
        validation_count = 1

    test_count = total - train_count - validation_count

    if test_count == 0:
        test_count = 1
        validation_count -= 1

    train_cases = shuffled[:train_count]

    validation_start = train_count
    validation_end = validation_start + validation_count

    validation_cases = shuffled[validation_start:validation_end]

    test_cases = shuffled[validation_end:]

    logger.info("BraTS dataset split:")
    logger.info(
        "Train cases      : %d",
        len(train_cases),
    )
    logger.info(
        "Validation cases : %d",
        len(validation_cases),
    )
    logger.info(
        "Test cases       : %d",
        len(test_cases),
    )

    return (
        train_cases,
        validation_cases,
        test_cases,
    )


def create_dataloaders(
    train_cases: Sequence[BraTSSample],
    validation_cases: Sequence[BraTSSample],
    test_cases: Sequence[BraTSSample],
    config: DataLoaderConfig | None = None,
    return_segmentation: bool = False,
) -> tuple[
    DataLoader,
    DataLoader,
    DataLoader,
]:
    if config is None:
        config = DataLoaderConfig()

    train_dataset = BraTSDataset(
        train_cases,
        return_segmentation=return_segmentation,
    )

    validation_dataset = BraTSDataset(
        validation_cases,
        return_segmentation=return_segmentation,
    )

    test_dataset = BraTSDataset(
        test_cases,
        return_segmentation=return_segmentation,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle_train,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
    )

    logger.info("DataLoaders created successfully.")

    logger.info(
        "Train samples      : %d",
        len(train_dataset),
    )

    logger.info(
        "Validation samples : %d",
        len(validation_dataset),
    )

    logger.info(
        "Test samples       : %d",
        len(test_dataset),
    )

    logger.info(
        "Batch size         : %d",
        config.batch_size,
    )

    logger.info(
        "Workers            : %d",
        config.num_workers,
    )

    return (
        train_loader,
        validation_loader,
        test_loader,
    )


def inspect_case(
    case: BraTSSample,
) -> dict[str, object]:
    result: dict[str, object] = {
        "case_id": case.case_id,
        "case_dir": str(case.case_dir),
    }

    for modality in MRI_MODALITIES:
        path = getattr(case, modality)

        image = nib.load(str(path))

        result[modality] = {
            "path": str(path),
            "shape": tuple(image.shape),
            "dtype": str(image.get_data_dtype()),
            "spacing": tuple(float(value) for value in image.header.get_zooms()[:3]),
        }

    if case.segmentation is not None:
        segmentation = nib.load(str(case.segmentation))

        result["segmentation"] = {
            "path": str(case.segmentation),
            "shape": tuple(segmentation.shape),
            "dtype": str(segmentation.get_data_dtype()),
        }

    return result


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=("%(asctime)s | %(levelname)s | " "%(name)s | %(message)s"),
    )

    logger.info("=" * 78)
    logger.info("BrainTumorDiagnosis BraTS Dataset Test")
    logger.info("=" * 78)

    root = Path("data/raw/BraTS2024")

    cases = discover_brats_cases(root)

    if not cases:
        raise RuntimeError("No valid BraTS cases were discovered.")

    logger.info(
        "Cases discovered: %d",
        len(cases),
    )

    first_case = cases[0]

    logger.info(
        "First case: %s",
        first_case.case_id,
    )

    logger.info(
        "Case directory: %s",
        first_case.case_dir,
    )

    for modality in MRI_MODALITIES:
        logger.info(
            "%s: %s",
            modality,
            getattr(first_case, modality),
        )

    inspection = inspect_case(first_case)

    logger.info("Case inspection:")

    for modality in MRI_MODALITIES:
        info = inspection[modality]

        logger.info(
            "%s | Shape: %s | Dtype: %s",
            modality,
            info["shape"],
            info["dtype"],
        )

    image = preprocess_case(first_case)

    logger.info(
        "Preprocessed image shape: %s",
        tuple(image.shape),
    )

    logger.info(
        "Preprocessed image dtype: %s",
        image.dtype,
    )

    logger.info(
        "Image minimum: %.6f",
        float(image.min()),
    )

    logger.info(
        "Image maximum: %.6f",
        float(image.max()),
    )

    logger.info(
        "Image mean: %.6f",
        float(image.mean()),
    )

    logger.info(
        "Image standard deviation: %.6f",
        float(image.std()),
    )

    (
        train_cases,
        validation_cases,
        test_cases,
    ) = split_brats_cases(
        cases,
        train_ratio=0.70,
        validation_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    config = DataLoaderConfig(
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        shuffle_train=True,
    )

    (
        train_loader,
        validation_loader,
        test_loader,
    ) = create_dataloaders(
        train_cases,
        validation_cases,
        test_cases,
        config=config,
        return_segmentation=False,
    )

    logger.info(
        "Train batches: %d",
        len(train_loader),
    )

    logger.info(
        "Validation batches: %d",
        len(validation_loader),
    )

    logger.info(
        "Test batches: %d",
        len(test_loader),
    )

    first_batch = next(iter(train_loader))

    logger.info(
        "First train batch image shape: %s",
        tuple(first_batch["image"].shape),
    )

    logger.info(
        "First train batch case IDs: %s",
        first_batch["case_id"],
    )

    logger.info("=" * 78)
    logger.info("BraTS Dataset Test Passed")
    logger.info("=" * 78)


if __name__ == "__main__":
    main()
