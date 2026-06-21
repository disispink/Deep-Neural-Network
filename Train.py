"""
Dog Breed Classification using Transfer Learning (EfficientNetB3)
Stanford Dogs Dataset — 120 Breeds  |  FIXED VERSION

Bugs fixed vs original:
  1. Removed rescale=1/255 — EfficientNetB3 rescales internally via its own
     preprocessing; dividing by 255 first destroys pretrained feature alignment.
  2. Replaced hardcoded `base_model(inputs, training=False)` with a proper
     Keras functional build so the `training` flag flows through model.fit.
  3. Added class_weight computation to handle breed imbalance.
  4. Stronger augmentation pipeline using tf.keras.layers (GPU-accelerated).
  5. EPOCHS_HEAD raised to 20; FINE_TUNE_AT raised to 250 (only unfreeze ~50 top
     layers instead of ~200) to avoid catastrophic forgetting.
  6. Label smoothing = 0.1 on categorical crossentropy.
  7. Replaced ImageDataGenerator with a tf.data pipeline (faster, more flexible).
  8. Added Cosine Decay LR schedule instead of raw constant LR.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetB3
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
IMG_SIZE      = 300          # EfficientNetB3 native input
BATCH_SIZE    = 32
EPOCHS_HEAD   = 20           # ↑ was 10 — head needs more epochs for 120 classes
EPOCHS_FINE   = 30           # Phase 2: fine-tune top layers
FINE_TUNE_AT  = 250          # ↑ was 100 — unfreeze only the last ~50 layers
LR_HEAD       = 1e-3
LR_FINE       = 5e-5         # slightly higher than before; cosine decay handles warmdown
NUM_CLASSES   = 120

DATA_DIR     = "dataset"
TRAIN_DIR    = os.path.join(DATA_DIR, "train")
VAL_DIR      = os.path.join(DATA_DIR, "val")
TEST_DIR     = os.path.join(DATA_DIR, "test")
MODEL_PATH   = "dog_classifier_final.keras"
HISTORY_PATH = "training_history.json"

AUTOTUNE = tf.data.AUTOTUNE


# ─────────────────────────────────────────────
# STEP 1 — tf.data PIPELINE  (replaces ImageDataGenerator)
# ─────────────────────────────────────────────
def load_dataset(directory, training=False):
    """
    Build a fast tf.data pipeline.
    FIX 1: No rescaling here — EfficientNetB3 includes its own preprocessing.
    FIX 7: Using tf.data instead of ImageDataGenerator for speed.
    """
    ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=training,
        seed=42,
    )

    if training:
        # FIX 4: stronger, GPU-accelerated augmentation via Keras layers
        augment = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.15),
            layers.RandomZoom(0.15),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomContrast(0.2),          # ← new: handles lighting variation
            layers.RandomBrightness(0.15),       # ← new: dogs photographed in varied light
        ], name="augmentation")
        ds = ds.map(lambda x, y: (augment(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)

    # NOTE: do NOT add rescaling. EfficientNetB3 preprocesses pixels internally.
    return ds.prefetch(AUTOTUNE)


def get_class_info(train_dir):
    """Return class names and a class_weight dict to handle imbalance."""
    class_names = sorted(os.listdir(train_dir))
    counts = np.array([
        len(os.listdir(os.path.join(train_dir, c))) for c in class_names
    ])
    total = counts.sum()
    # sklearn-style balanced weighting
    class_weight = {
        i: total / (NUM_CLASSES * counts[i]) for i in range(len(class_names))
    }
    return class_names, class_weight


# ─────────────────────────────────────────────
# STEP 2 — MODEL CONSTRUCTION
# ─────────────────────────────────────────────
def build_model(num_classes: int):
    """
    FIX 2: Build model so the `training` flag properly flows into EfficientNet's
    BatchNorm layers at fit time. The key: do NOT pass training=False at graph-
    build time — let Keras handle it during model.fit / model.evaluate.

    Architecture:
      Input (300×300×3)
        → EfficientNetB3 [frozen in Phase 1, partially unfrozen in Phase 2]
        → GlobalAveragePooling2D
        → Dense(512) + BatchNorm + Dropout(0.4)
        → Dense(256) + Dropout(0.3)
        → Dense(120, softmax)
    """
    base_model = EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    base_model.trainable = False   # frozen for Phase 1

    # FIX 2: build the head using base_model as a layer (not called with training=False)
    inputs  = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x       = base_model(inputs)                   # ← no hardcoded training=False
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.Dense(512, activation="relu")(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dropout(0.4)(x)
    x       = layers.Dense(256, activation="relu")(x)
    x       = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="DogBreedClassifier")
    return model, base_model


# ─────────────────────────────────────────────
# STEP 3 — COMPILE
# ─────────────────────────────────────────────
def compile_model(model, lr, total_steps=None):
    """
    FIX 5 (partial): Cosine decay schedule warms down LR smoothly.
    FIX 6: label_smoothing=0.1 prevents overconfident predictions on
           visually similar breeds (e.g. Golden vs. Labrador Retriever).
    """
    if total_steps:
        lr = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=lr,
            decay_steps=total_steps,
            alpha=1e-7,
        )

    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=0.1          # FIX 6
        ),
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_acc"),
        ],
    )


# ─────────────────────────────────────────────
# STEP 4 — CALLBACKS
# ─────────────────────────────────────────────
def get_callbacks(phase: str):
    return [
        callbacks.ModelCheckpoint(
            filepath=f"best_model_{phase}.keras",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=7,              # ↑ was 5 — give fine-tuning more room
            restore_best_weights=True,
            verbose=1,
        ),
        # ReduceLROnPlateau is less useful with CosineDecay but keep as safety net
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-8,
            verbose=1,
        ),
        callbacks.TensorBoard(log_dir=f"logs/{phase}"),
    ]


# ─────────────────────────────────────────────
# STEP 5 — TRAINING
# ─────────────────────────────────────────────
def train(model, base_model, train_ds, val_ds, class_weight):
    history_all = {}
    n_train_batches = len(train_ds)

    # ── Phase 1: Head only ───────────────────
    print("\n" + "="*55)
    print("PHASE 1: Training classification head (base frozen)")
    print("="*55)
    compile_model(model, LR_HEAD,
                  total_steps=EPOCHS_HEAD * n_train_batches)
    model.summary()

    h1 = model.fit(
        train_ds,
        epochs=EPOCHS_HEAD,
        validation_data=val_ds,
        class_weight=class_weight,       # FIX 3
        callbacks=get_callbacks("phase1"),
    )
    history_all["phase1"] = h1.history

    # ── Phase 2: Fine-tune top layers ────────
    print("\n" + "="*55)
    print(f"PHASE 2: Fine-tuning from layer {FINE_TUNE_AT} onward")
    print("="*55)

    base_model.trainable = True
    for layer in base_model.layers[:FINE_TUNE_AT]:
        layer.trainable = False
    # Keep BN in inference mode for the frozen portion
    for layer in base_model.layers[:FINE_TUNE_AT]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    unfrozen = sum(1 for l in base_model.layers if l.trainable)
    print(f"   Unfrozen base layers: {unfrozen} / {len(base_model.layers)}")

    compile_model(model, LR_FINE,
                  total_steps=EPOCHS_FINE * n_train_batches)

    h2 = model.fit(
        train_ds,
        epochs=EPOCHS_FINE,
        validation_data=val_ds,
        class_weight=class_weight,       # FIX 3
        callbacks=get_callbacks("phase2"),
    )
    history_all["phase2"] = h2.history

    with open(HISTORY_PATH, "w") as f:
        json.dump(history_all, f, indent=2)

    return h1, h2


# ─────────────────────────────────────────────
# STEP 6 — EVALUATION & PLOTS
# ─────────────────────────────────────────────
def evaluate(model, test_ds):
    print("\n" + "="*55)
    print("EVALUATING on test set …")
    print("="*55)
    loss, acc, top5 = model.evaluate(test_ds, verbose=1)
    print(f"\n  Test Loss      : {loss:.4f}")
    print(f"  Test Accuracy  : {acc*100:.2f}%")
    print(f"  Top-5 Accuracy : {top5*100:.2f}%")
    return loss, acc, top5


def plot_history(h1, h2):
    acc      = h1.history["accuracy"]     + h2.history["accuracy"]
    val_acc  = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    loss     = h1.history["loss"]         + h2.history["loss"]
    val_loss = h1.history["val_loss"]     + h2.history["val_loss"]
    epochs   = range(1, len(acc) + 1)
    split    = len(h1.history["accuracy"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, acc,     label="Train Accuracy")
    ax1.plot(epochs, val_acc, label="Val Accuracy")
    ax1.axvline(split, color="gray", linestyle="--", label="Fine-tune start")
    ax1.set_title("Accuracy"); ax1.set_xlabel("Epoch"); ax1.legend()

    ax2.plot(epochs, loss,     label="Train Loss")
    ax2.plot(epochs, val_loss, label="Val Loss")
    ax2.axvline(split, color="gray", linestyle="--", label="Fine-tune start")
    ax2.set_title("Loss"); ax2.set_xlabel("Epoch"); ax2.legend()

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.show()
    print("\n📊 Plot saved → training_curves.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"GPUs available: {len(gpus)}")

    # Build datasets
    train_ds = load_dataset(TRAIN_DIR, training=True)
    val_ds   = load_dataset(VAL_DIR,   training=False)
    test_ds  = load_dataset(TEST_DIR,  training=False)

    # Class info
    class_names, class_weight = get_class_info(TRAIN_DIR)
    print(f"\n✅ {len(class_names)} classes found")
    with open("class_indices.json", "w") as f:
        json.dump({name: i for i, name in enumerate(class_names)}, f, indent=2)

    # Build, train, evaluate
    model, base_model = build_model(NUM_CLASSES)
    h1, h2 = train(model, base_model, train_ds, val_ds, class_weight)
    evaluate(model, test_ds)
    plot_history(h1, h2)

    model.save(MODEL_PATH)
    print(f"\n✅ Model saved → {MODEL_PATH}")