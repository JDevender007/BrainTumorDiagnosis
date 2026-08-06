import logging
from pathlib import Path

from datasets.dataloader import BraTSDataModule

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s : %(message)s",
)

DATASET_ROOT = r"D:\Projects\BrainTumorDiagnosis\data\raw"

if not Path(DATASET_ROOT).exists():
    raise FileNotFoundError(DATASET_ROOT)

print("Dataset Found:", DATASET_ROOT)

datamodule = BraTSDataModule(
    dataset_root=DATASET_ROOT,
    batch_size=1,
    num_workers=0,
)

datamodule.setup()

datamodule.summary()

train_loader = datamodule.train_dataloader()

print("Number of Training Batches:", len(train_loader))

batch = next(iter(train_loader))

print("Image Shape:", batch["image"].shape)
print("Mask Shape:", batch["mask"].shape)
print("Image Dtype:", batch["image"].dtype)
print("Mask Dtype:", batch["mask"].dtype)

print("DataLoader Test Passed Successfully.")
