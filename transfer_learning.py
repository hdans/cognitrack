"""
Transfer Learning CNN (MobileNetV2) for Facial Emotion Recognition.
"""

import os

# Backend setup
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
import keras
from keras import layers
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# Global Config
IMG_SIZE        = 224
BATCH_SIZE      = 32
EPOCHS          = 50
LEARNING_RATE   = 1e-3
DROPOUT_RATE    = 0.5
NUM_CLASSES     = 3
TRAIN_DIR       = "dataset/train"
VAL_DIR         = "dataset/val"
BEST_MODEL_PATH = "best_fer_model.keras"   # Format .keras lebih stabil di Keras 3
CLASS_NAMES     = ["Negatif", "Netral", "Positif"]  # akan di-override oleh dataset

keras.utils.set_random_seed(42)


# Data Loader

def make_augmentation_layer() -> keras.Sequential:
    """
    Data augmentation layer for training.
    """
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.04),
        layers.RandomZoom(height_factor=0.10, width_factor=0.10),
        layers.RandomTranslation(height_factor=0.10, width_factor=0.10),
    ], name="augmentation")


def preprocess_fn(image, label):
    """
    Preprocessing function.
    """
    image = preprocess_input(image)
    return image, label


def build_datasets(train_dir: str, val_dir: str):
    """
    Builds tf.data.Dataset for training and validation.
    """
    # --- Training dataset ---
    train_ds_raw = keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="categorical",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42,
    )

    # --- Validasi dataset ---
    val_ds_raw = keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="categorical",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    class_names = train_ds_raw.class_names
    n_train     = sum(1 for _ in train_ds_raw.unbatch())
    n_val       = sum(1 for _ in val_ds_raw.unbatch())

    print("\n=== Informasi Dataset ===")
    print(f"Nama kelas (urutan): {class_names}")
    print(f"Total sampel training  : {n_train}")
    print(f"Total sampel validasi  : {n_val}")

    # Terapkan preprocessing (rescaling ke [-1,1])
    train_ds = train_ds_raw.map(preprocess_fn, num_parallel_calls=-1)
    val_ds   = val_ds_raw.map(preprocess_fn, num_parallel_calls=-1)

    # Optimasi pipeline I/O
    train_ds = train_ds.prefetch(buffer_size=-1)
    val_ds   = val_ds.prefetch(buffer_size=-1)

    return train_ds, val_ds, class_names, n_train, n_val


# Compute Class Weights

def compute_class_weights_from_dir(train_dir: str, class_names: list) -> dict:
    """
    Computes class weights from the directory structure.
    """
    counts = []
    for cls in class_names:
        cls_path = os.path.join(train_dir, cls)
        count = len([
            f for f in os.listdir(cls_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ])
        counts.append(count)

    labels_flat = []
    for idx, count in enumerate(counts):
        labels_flat.extend([idx] * count)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(NUM_CLASSES),
        y=np.array(labels_flat)
    )
    cw_dict = {i: float(w) for i, w in enumerate(weights)}

    print("\n=== Class Weights ===")
    for idx, w in cw_dict.items():
        print(f"  [{idx}] {class_names[idx]:<12}  count={counts[idx]:>5}  weight={w:.4f}")
    return cw_dict


# Model Building

def build_model(num_classes: int = NUM_CLASSES) -> keras.Model:
    """
    Builds the MobileNetV2 based model.
    """
    augmentation = make_augmentation_layer()

    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False   # Bekukan seluruh MobileNetV2

    inputs  = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_image")
    x       = augmentation(inputs)                          # augmentasi inline
    x       = base_model(x, training=False)                 # BN tetap inference mode
    x       = layers.GlobalAveragePooling2D(name="gap")(x)
    x       = layers.Dropout(DROPOUT_RATE, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="FER_MobileNetV2")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\n=== Arsitektur Model ===")
    model.summary()
    return model


# Callbacks

def build_callbacks() -> list:
    """
    Builds callbacks for training.
    """
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    checkpoint = ModelCheckpoint(
        filepath=BEST_MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
    return [early_stop, checkpoint]


# Evaluation

def evaluate_model(model: keras.Model, val_ds, class_names: list):
    """
    Evaluates the model.
    """
    print("\n=== Evaluasi Model ===")

    y_pred_list, y_true_list = [], []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_pred_list.append(np.argmax(preds, axis=1))
        y_true_list.append(np.argmax(labels, axis=1))

    y_pred = np.concatenate(y_pred_list)
    y_true = np.concatenate(y_true_list)

    f1_macro    = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")

    print(f"\nF1-Score (Macro)    : {f1_macro:.4f}")
    print(f"F1-Score (Weighted) : {f1_weighted:.4f}")
    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix — FER MobileNetV2")
    plt.ylabel("Label Aktual")
    plt.xlabel("Label Prediksi")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()
    print("Confusion matrix disimpan: 'confusion_matrix.png'")
    return f1_macro, cm


def plot_training_history(history):
    """Plots training history."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["loss"],         label="Train Loss",     color="royalblue")
    axes[0].plot(history.history["val_loss"],     label="Val Loss",       color="tomato")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.history["accuracy"],     label="Train Accuracy", color="royalblue")
    axes[1].plot(history.history["val_accuracy"], label="Val Accuracy",   color="tomato")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle("Training History — FER MobileNetV2", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)
    plt.show()
    print("Kurva training disimpan: 'training_history.png'")


# Main Pipeline

def main():
    print("=" * 60)
    print("  FASE 1: TRANSFER LEARNING CNN — Facial Emotion Recognition")
    print(f"  Backend Keras : {keras.backend.backend()}")
    print("=" * 60)

    # Validasi folder dataset
    if not os.path.exists(TRAIN_DIR) or not os.path.exists(VAL_DIR):
        raise FileNotFoundError(
            f"Folder dataset tidak ditemukan!\n"
            f"Buat folder '{TRAIN_DIR}' dan '{VAL_DIR}' dengan sub-folder kelas:\n"
            f"  Positif/, Netral/, Negatif/"
        )

    # 1. Bangun dataset pipeline
    train_ds, val_ds, class_names, n_train, n_val = build_datasets(TRAIN_DIR, VAL_DIR)

    # 2. Class weights
    class_weights = compute_class_weights_from_dir(TRAIN_DIR, class_names)

    # 3. Model
    model = build_model(num_classes=len(class_names))

    # 4. Callbacks
    callbacks = build_callbacks()

    # 5. Training
    print("\n=== Memulai Training ===")
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    print(f"\n✓ Training selesai. Model terbaik disimpan: '{BEST_MODEL_PATH}'")

    # 6. Kurva training
    plot_training_history(history)

    # 7. Evaluasi final
    evaluate_model(model, val_ds, class_names)

    print("\n✓ Pipeline Fase 1 selesai.")
    print(f"  Model         : {BEST_MODEL_PATH}")
    print(f"  Confusion mtx : confusion_matrix.png")
    print(f"  Training curve: training_history.png")


if __name__ == "__main__":
    main()
