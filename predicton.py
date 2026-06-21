"""
Dog Breed Prediction — Test from URL or local file
Usage:
    python predict_from_url.py --url "https://example.com/dog.jpg"
    python predict_from_url.py --file "my_dog.jpg"
"""

import argparse
import json
import numpy as np
import tensorflow as tf
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# CONFIG — match your training settings
# ─────────────────────────────────────────────
IMG_SIZE    = 300
MODEL_PATH  = r"C:\machine learning\Image Classification NN\Dog Breed Classifier\best_model_phase2.keras"
CLASS_JSON  = r"C:\machine learning\Image Classification NN\Dog Breed Classifier\class_indices.json"
TOP_K       = 5


# ─────────────────────────────────────────────
# LOAD MODEL + CLASS MAP
# ─────────────────────────────────────────────
def load_model_and_classes():
    print(f"Loading model from {MODEL_PATH} ...")
    model = tf.keras.models.load_model(MODEL_PATH)  # ← do not touch this

    with open(CLASS_JSON, "r") as f:
        class_indices = json.load(f)

    idx_to_class = {v: k for k, v in class_indices.items()}
    print(f"✅ Model loaded | {len(idx_to_class)} breeds")
    return model, idx_to_class


# ─────────────────────────────────────────────
# IMAGE LOADING
# ─────────────────────────────────────────────
def load_image_from_url(url: str) -> Image.Image:
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content)).convert("RGB")
    print(f"✅ Image loaded from URL  ({img.size[0]}×{img.size[1]} px)")
    return img


def load_image_from_file(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    print(f"✅ Image loaded from file ({img.size[0]}×{img.size[1]} px)")
    return img


# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
def preprocess(img: Image.Image) -> tf.Tensor:
    """
    Resize → numpy array → add batch dim.
    Do NOT divide by 255 — EfficientNetB3 handles that internally.
    """
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32)   # shape (300, 300, 3), range [0, 255]
    arr = np.expand_dims(arr, axis=0)        # shape (1, 300, 300, 3)
    return arr


# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
def predict(model, img_array, idx_to_class, top_k=TOP_K):
    preds = model.predict(img_array, verbose=0)[0]   # shape (120,)

    top_indices = np.argsort(preds)[::-1][:top_k]
    results = [
        {
            "rank":       i + 1,
            "breed":      idx_to_class[idx].replace("_", " ").title(),
            "confidence": float(preds[idx]) * 100,
        }
        for i, idx in enumerate(top_indices)
    ]
    return results


# ─────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────
def display_results(img: Image.Image, results: list):
    fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: the image
    ax_img.imshow(img)
    ax_img.axis("off")
    ax_img.set_title(f"Prediction: {results[0]['breed']}", fontsize=13, fontweight="bold")

    # Right: horizontal bar chart of top-K
    breeds      = [r["breed"] for r in reversed(results)]
    confidences = [r["confidence"] for r in reversed(results)]
    colors      = ["#1D9E75" if i == len(results) - 1 else "#9FE1CB"
                   for i in range(len(results))]

    bars = ax_bar.barh(breeds, confidences, color=colors, height=0.5)
    ax_bar.set_xlabel("Confidence (%)")
    ax_bar.set_title(f"Top-{TOP_K} Predictions")
    ax_bar.set_xlim(0, 100)

    for bar, conf in zip(bars, confidences):
        ax_bar.text(
            bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{conf:.1f}%", va="center", fontsize=10,
        )

    plt.tight_layout()
    plt.savefig("prediction_result.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\n📊 Result saved → prediction_result.png")


def print_results(results: list):
    print("\n" + "=" * 45)
    print(f"  TOP-{TOP_K} BREED PREDICTIONS")
    print("=" * 45)
    for r in results:
        bar = "█" * int(r["confidence"] / 5)
        print(f"  #{r['rank']}  {r['breed']:<30} {r['confidence']:5.1f}%  {bar}")
    print("=" * 45)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    model, idx_to_class = load_model_and_classes()

    # ── OPTION A: paste your image URL here ──
    #img = load_image_from_url("https://YOUR_IMAGE_URL_HERE.jpg")

    # ── OPTION B: paste your local file path here ──
    img = load_image_from_file(r"C:\machine learning\Image Classification NN\Dog Breed Classifier\dog_breed_classifier\2.jpg")

    img_array = preprocess(img)
    results   = predict(model, img_array, idx_to_class)
    print_results(results)
    display_results(img, results)