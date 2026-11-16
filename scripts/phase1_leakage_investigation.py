"""Phase 1: Data Leakage Investigation."""
import sys, json, hashlib, os
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, ".")

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

def collect_images(root):
    """Collect all image paths and their basenames."""
    root = Path(root)
    images = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            images[p.name] = p
    return images

def file_hash(path, block=65536):
    """Compute MD5 hash of file contents."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(block)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

print("=" * 70)
print("PHASE 1: DATA LEAKAGE INVESTIGATION")
print("=" * 70)

# 1. Collect all images
print("\n--- Collecting image paths ---")
internal_train = collect_images("data/raw/classification/Train")
internal_test = collect_images("data/raw/classification/Test")
kaggle_test = collect_images("data/external/kaggle_brain_tumor/Testing")
kaggle_train = collect_images("data/external/kaggle_brain_tumor/Training")
brisc = collect_images("data/external/brisc2025/brisc2025/classification_task/test")

print(f"  Internal Train: {len(internal_train)} images")
print(f"  Internal Test:  {len(internal_test)} images")
print(f"  Kaggle Testing: {len(kaggle_test)} images")
print(f"  Kaggle Training: {len(kaggle_train)} images")
print(f"  BRISC:          {len(brisc)} images")

# 2. Filename overlap analysis
print("\n--- Filename Overlap Analysis ---")
internal_all_names = set(internal_train.keys()) | set(internal_test.keys())
kaggle_all_names = set(kaggle_test.keys()) | set(kaggle_train.keys())

overlap_internal_kaggle = internal_all_names & kaggle_all_names
print(f"  Internal ALL vs Kaggle ALL: {len(overlap_internal_kaggle)} filename overlaps")
print(f"    Internal: {len(internal_all_names)} unique filenames")
print(f"    Kaggle:   {len(kaggle_all_names)} unique filenames")

# Specific overlaps
overlap_train_ktest = set(internal_train.keys()) & set(kaggle_test.keys())
overlap_test_ktest = set(internal_test.keys()) & set(kaggle_test.keys())
overlap_train_ktrain = set(internal_train.keys()) & set(kaggle_train.keys())
overlap_test_ktrain = set(internal_test.keys()) & set(kaggle_train.keys())

print(f"\n  Internal Train vs Kaggle Test:  {len(overlap_train_ktest)} overlaps")
print(f"  Internal Test vs Kaggle Test:   {len(overlap_test_ktest)} overlaps")
print(f"  Internal Train vs Kaggle Train: {len(overlap_train_ktrain)} overlaps")
print(f"  Internal Test vs Kaggle Train:  {len(overlap_test_ktrain)} overlaps")

# Kaggle Testing overlaps only - this is the contamination
print(f"\n  *** CONTAMINATION: {len(overlap_train_ktest)} train images appear in Kaggle test set ***")
print(f"  *** CONTAMINATION: {len(overlap_test_ktest)} test images appear in Kaggle test set ***")

# Show some examples
if overlap_train_ktest:
    print(f"\n  Sample overlapping filenames (Train -> Kaggle Test):")
    for name in sorted(overlap_train_ktest)[:10]:
        print(f"    {name}")

# 3. Internal train/test overlap
print("\n--- Internal Train/Test Overlap ---")
overlap_train_test = set(internal_train.keys()) & set(internal_test.keys())
print(f"  Internal Train vs Internal Test: {len(overlap_train_test)} filename overlaps")
if overlap_train_test:
    print(f"  CRITICAL: {len(overlap_train_test)} images appear in BOTH train and test!")
    for name in sorted(overlap_train_test)[:5]:
        print(f"    {name}")

# 4. Class distribution
print("\n--- Class Distribution ---")
for split_name, images_dir in [
    ("Internal Train", "data/raw/classification/Train"),
    ("Internal Test", "data/raw/classification/Test"),
    ("Kaggle Test", "data/external/kaggle_brain_tumor/Testing"),
    ("Kaggle Train", "data/external/kaggle_brain_tumor/Training"),
    ("BRISC", "data/external/brisc2025/brisc2025/classification_task/test"),
]:
    d = Path(images_dir)
    if not d.exists():
        print(f"  {split_name}: NOT FOUND"); continue
    classes = {}
    for cd in sorted(d.iterdir()):
        if cd.is_dir():
            count = len([p for p in cd.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT])
            classes[cd.name] = count
    total = sum(classes.values())
    print(f"  {split_name} ({total}): {classes}")

# 5. Content-based duplicate detection (sample-based for speed)
print("\n--- Content-Based Duplicate Detection (MD5 hash) ---")
print("  Computing hashes for internal train (sampling up to 500)...")
sample_train = dict(list(internal_train.items())[:500])
train_hashes = {}
for name, path in sample_train.items():
    try:
        h = file_hash(path)
        train_hashes.setdefault(h, []).append(name)
    except:
        pass

dupes = {h: names for h, names in train_hashes.items() if len(names) > 1}
print(f"  Hash duplicates in sampled train: {len(dupes)} groups")
for h, names in list(dupes.items())[:5]:
    print(f"    {names}")

# Check if any train hashes match test hashes
print("  Computing hashes for internal test...")
test_hashes = {}
for name, path in internal_test.items():
    try:
        h = file_hash(path)
        test_hashes.setdefault(h, []).append(name)
    except:
        pass

# Cross-set hash overlap
hash_overlap = set(train_hashes.keys()) & set(test_hashes.keys())
print(f"  Content duplicates between Train and Test (from sampled train): {len(hash_overlap)}")
if hash_overlap:
    print("  CRITICAL: Content-identical images found in both train and test!")
    for h in list(hash_overlap)[:5]:
        print(f"    Train: {train_hashes[h]}, Test: {test_hashes[h]}")

# 6. Save full report
print("\n--- Saving Report ---")
report = {
    "counts": {
        "internal_train": len(internal_train),
        "internal_test": len(internal_test),
        "kaggle_test": len(kaggle_test),
        "kaggle_train": len(kaggle_train),
        "brisc": len(brisc),
    },
    "filename_overlaps": {
        "internal_all_vs_kaggle_all": len(overlap_internal_kaggle),
        "internal_train_vs_kaggle_test": len(overlap_train_ktest),
        "internal_test_vs_kaggle_test": len(overlap_test_ktest),
        "internal_train_vs_kaggle_train": len(overlap_train_ktrain),
        "internal_test_vs_kaggle_train": len(overlap_test_ktrain),
        "internal_train_vs_internal_test": len(overlap_train_test),
    },
    "content_duplicates": {
        "within_train_sampled": len(dupes),
        "between_train_and_test": len(hash_overlap),
    },
    "contamination": {
        "kaggle_test_contaminated": len(overlap_train_ktest) + len(overlap_test_ktest),
        "kaggle_test_clean": len(kaggle_test) - len(overlap_train_ktest) - len(overlap_test_ktest),
    },
    "overlap_filenames_kaggle_test": sorted(list(overlap_train_ktest | overlap_test_ktest)),
}
with open("research_final_validation/leakage_report.json", "w") as f:
    json.dump(report, f, indent=2)
print(f"  Saved to research_final_validation/leakage_report.json")

print("\n" + "=" * 70)
print("LEAKAGE VERDICT")
print("=" * 70)
if overlap_train_test:
    print("  INTERNAL TRAIN/TEST OVERLAP: FAIL - images leak between splits")
else:
    print("  INTERNAL TRAIN/TEST OVERLAP: PASS - no filename overlap")

if overlap_train_ktest or overlap_test_ktest:
    print(f"  KAGGLE CONTAMINATION: FAIL - {len(overlap_train_ktest | overlap_test_ktest)} images overlap")
    print(f"  -> Kaggle CANNOT be used as independent external validation")
else:
    print("  KAGGLE CONTAMINATION: PASS - no overlap")

if hash_overlap:
    print("  CONTENT DUPLICATES: FAIL - identical images in train and test")
else:
    print("  CONTENT DUPLICATES: PASS (within sampled subset)")
