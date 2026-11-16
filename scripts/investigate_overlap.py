"""Investigate the 218 train/test overlap and duplicate filenames."""
import json
from pathlib import Path
from collections import Counter

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

def collect_images_with_class(root):
    """Collect images with their class labels."""
    root = Path(root)
    images = {}
    for cd in sorted(root.iterdir()):
        if not cd.is_dir(): continue
        for p in cd.iterdir():
            if p.is_file() and p.suffix.lower() in IMG_EXT:
                if p.name not in images:
                    images[p.name] = {"path": str(p), "class": cd.name, "dir": str(cd)}
                else:
                    images[p.name]["duplicate_in"] = cd.name
    return images

print("=" * 70)
print("INVESTIGATING TRAIN/TEST OVERLAP AND DUPLICATES")
print("=" * 70)

# Collect with class info
train = collect_images_with_class("data/raw/classification/Train")
test = collect_images_with_class("data/raw/classification/Test")

print(f"\nTrain: {len(train)} unique filenames")
print(f"Test:  {len(test)} unique filenames")

# Check for duplicates within train
train_dupes = {n: info for n, info in train.items() if "duplicate_in" in info}
print(f"\nTrain duplicates (same filename, different class dirs): {len(train_dupes)}")
for n, info in list(train_dupes.items())[:10]:
    print(f"  {n}: {info['class']} and {info['duplicate_in']}")

# Check for duplicates within test
test_dupes = {n: info for n, info in test.items() if "duplicate_in" in info}
print(f"\nTest duplicates (same filename, different class dirs): {len(test_dupes)}")
for n, info in list(test_dupes.items())[:10]:
    print(f"  {n}: {info['class']} and {info['duplicate_in']}")

# Check train/test overlap
overlap = set(train.keys()) & set(test.keys())
print(f"\nTrain/Test filename overlap: {len(overlap)}")
for n in sorted(overlap)[:10]:
    print(f"  {n}: train={train[n]['class']}, test={test[n]['class']}")

# Count actual files per class (including duplicates)
print("\n--- Actual file counts (all files, including duplicate names) ---")
for split_name, root in [("Train", "data/raw/classification/Train"), ("Test", "data/raw/classification/Test")]:
    d = Path(root)
    total = 0
    for cd in sorted(d.iterdir()):
        if cd.is_dir():
            count = len([p for p in cd.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT])
            total += count
            print(f"  {split_name}/{cd.name}: {count} files")
    print(f"  {split_name} TOTAL: {count} files")
    unique = len(train if split_name == "Train" else test)
    print(f"  {split_name} UNIQUE filenames: {unique}")
    print(f"  {split_name} DUPLICATE names: {total - unique}")
    print()
