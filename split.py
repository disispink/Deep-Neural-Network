"""
split.py  —  Organise the Stanford Dogs Dataset into train / val / test splits.

Expected input layout (what Kaggle gives you):
    images/
        Images/
            n02085620-Chihuahua/
                n02085620_1001.jpg
                ...
            n02085782-Japanese_spaniel/
                ...
    annotations/
        Annotation/
            n02085620-Chihuahua/
                n02085620_1001   ← XML bounding-box files (optional)

Output layout created by this script:
    dataset/
        train/  <class_name>/  *.jpg
        val/    <class_name>/  *.jpg
        test/   <class_name>/  *.jpg

Ratios: 70 % train | 15 % val | 15 % test  (stratified per breed)
"""

import os
import shutil
import random
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────
RAW_IMAGES_DIR = Path("C:\machine learning\Image Classification NN\Dog Breed Classifier\dog_breed_classifier\images\Images")      # adjust if your path differs
OUTPUT_DIR     = Path("dataset")
TRAIN_RATIO    = 0.70
VAL_RATIO      = 0.15
# TEST_RATIO is implicitly 1 - TRAIN - VAL = 0.15
SEED           = 42
# ─────────────────────────────────────────────────────────────────────────────


def clean_class_name(folder_name: str) -> str:
    """
    Convert 'n02085620-Chihuahua' → 'Chihuahua'
    Makes folder names human-readable and Keras-friendly.
    """
    parts = folder_name.split("-", 1)
    return parts[1].replace("_", " ") if len(parts) > 1 else folder_name


def split_dataset():
    random.seed(SEED)

    if not RAW_IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"Could not find '{RAW_IMAGES_DIR}'. "
            "Check that 'images/Images' exists in your project folder."
        )

    breed_folders = sorted([d for d in RAW_IMAGES_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(breed_folders)} breed folders.\n")

    stats = {"train": 0, "val": 0, "test": 0}

    for breed_folder in breed_folders:
        class_name = clean_class_name(breed_folder.name)

        # Collect all JPEG images for this breed
        images = list(breed_folder.glob("*.jpg")) + list(breed_folder.glob("*.JPEG"))
        if not images:
            print(f"  ⚠  No images found in {breed_folder.name}, skipping.")
            continue

        random.shuffle(images)
        n       = len(images)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)

        splits = {
            "train": images[:n_train],
            "val":   images[n_train : n_train + n_val],
            "test":  images[n_train + n_val :],
        }

        for split_name, split_images in splits.items():
            dest_dir = OUTPUT_DIR / split_name / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)

            for img_path in split_images:
                shutil.copy2(img_path, dest_dir / img_path.name)

            stats[split_name] += len(split_images)

        print(f"  ✔  {class_name:<30} "
              f"train={len(splits['train']):<4} "
              f"val={len(splits['val']):<4} "
              f"test={len(splits['test']):<4}")

    print("\n" + "="*55)
    print(f"Dataset split complete!")
    print(f"  Total train : {stats['train']}")
    print(f"  Total val   : {stats['val']}")
    print(f"  Total test  : {stats['test']}")
    print(f"  Grand total : {sum(stats.values())}")
    print(f"\nOutput saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    split_dataset()
