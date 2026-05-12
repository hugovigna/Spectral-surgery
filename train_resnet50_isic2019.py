"""
train_resnet50_isic2019.py
--------------------------
Fine-tune ResNet50 (poids ImageNet) sur ISIC 2019 SANS rééquilibrage.

Objectif : produire un modèle naturellement biaisé vers les classes majoritaires
(NV ~50% du dataset), qui servira de baseline pour les expériences de Spectral
Surgery / Focal Loss / Class-balanced FT.

Stratégie : full fine-tuning end-to-end dès le début (comme CIFAR-10/100),
sans class_weights, BN gelés pour stabilité Metal GPU.

Usage :
    python3 train_resnet50_isic2019.py
    python3 train_resnet50_isic2019.py --data-dir archive
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import pathlib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# ════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════
CONFIG = {
    "data_dir"     : "archive",
    "img_size"     : 224,
    "batch_size"   : 128,
    "seed"         : 42,
    "val_split"    : 0.15,
    "epochs"       : 40,
    "lr"           : 1e-4,
    "weight_decay" : 1e-3,
    "label_smoothing": 0.1,
    "output_dir"   : "results/isic2019/training",
    "model_name"   : "resnet50_isic2019_v3",
}

CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

# ════════════════════════════════════════════════════════════════════════════
# 1. Chargement des chemins depuis les dossiers par classe
# ════════════════════════════════════════════════════════════════════════════

CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

# ════════════════════════════════════════════════════════════════════════════
# 1. Chargement des chemins depuis les dossiers par classe
# ════════════════════════════════════════════════════════════════════════════

def load_metadata(data_dir: pathlib.Path):
    paths, labels = [], []
    print("    Distribution des classes :")
    for i, cls in enumerate(CLASSES):
        cls_dir = data_dir / cls
        if not cls_dir.exists():
            print(f"      [{i}] {cls:4s} : DOSSIER ABSENT — ignoré")
            continue
        imgs = sorted(cls_dir.glob("*.jpg"))
        print(f"      [{i}] {cls:4s} : {len(imgs):5d} images")
        paths.extend([str(p) for p in imgs])
        labels.extend([i] * len(imgs))

    if not paths:
        raise RuntimeError(
            f"Aucune image trouvée dans {data_dir}.\n"
            "Vérifie que les dossiers MEL/, NV/, BCC/, etc. existent."
        )

    return np.array(paths), np.array(labels, dtype=np.int32)


# ════════════════════════════════════════════════════════════════════════════
# 2. Pré-cache des images (decode JPEG + resize une seule fois)
# ════════════════════════════════════════════════════════════════════════════

IMG_SIZE = CONFIG["img_size"]
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float16)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float16)

def preprocess_to_cache(paths, labels, cache_path: pathlib.Path):
    """Lit tous les JPEGs, resize, normalise, sauvegarde en float16 npz."""
    import cv2
    n = len(paths)
    imgs = np.empty((n, IMG_SIZE, IMG_SIZE, 3), dtype=np.float16)
    for i, p in enumerate(paths):
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        imgs[i] = (img.astype(np.float16) / 255.0 - _MEAN) / _STD
        if (i + 1) % 2000 == 0:
            print(f"    {i+1}/{n} images prétraitées ...")
    np.savez_compressed(cache_path, imgs=imgs, labels=labels)
    print(f"    Cache sauvegardé : {cache_path}  ({imgs.nbytes/1e9:.1f} Go)")
    return imgs, labels

def load_cache(cache_path: pathlib.Path):
    data = np.load(cache_path)
    return data["imgs"], data["labels"]

def get_or_build_cache(paths, labels, cache_path: pathlib.Path):
    if cache_path.exists():
        print(f"    Chargement du cache : {cache_path}")
        return load_cache(cache_path)
    print(f"    Construction du cache (première fois) ...")
    return preprocess_to_cache(paths, labels, cache_path)

# ════════════════════════════════════════════════════════════════════════════
# 3. Pipeline tf.data (depuis cache numpy)
# ════════════════════════════════════════════════════════════════════════════

def augment_image(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, 0.15)
    img = tf.image.random_contrast(img, 0.85, 1.15)
    img = tf.image.random_saturation(img, 0.85, 1.15)
    img = tf.image.resize_with_crop_or_pad(img, IMG_SIZE + 20, IMG_SIZE + 20)
    img = tf.image.random_crop(img, [IMG_SIZE, IMG_SIZE, 3])
    return img, label

def make_dataset(imgs, labels, batch_size, augment=False, shuffle=False):
    # Forcer le tenseur source sur CPU pour éviter que TF copie 6 Go d'un coup sur le GPU
    with tf.device("/CPU:0"):
        ds = tf.data.Dataset.from_tensor_slices(
            (imgs.astype(np.float32), labels.astype(np.int32))
        )
    if shuffle:
        ds = ds.shuffle(len(imgs), seed=CONFIG["seed"])
    if augment:
        ds = ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ════════════════════════════════════════════════════════════════════════════
# 3. Construction du modèle
# ════════════════════════════════════════════════════════════════════════════

def build_model(n_classes: int):
    base = tf.keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling="avg",
    )
    # Full fine-tuning — BN gelés pour stabilité Metal
    base.trainable = True
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    inputs  = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x       = base(inputs, training=False)
    x       = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="ResNet50_ISIC2019")


# ════════════════════════════════════════════════════════════════════════════
# 4. Évaluation par classe
# ════════════════════════════════════════════════════════════════════════════

def per_class_accuracy(model, ds, n_classes):
    all_preds, all_labels = [], []
    for x_batch, y_batch in ds:
        preds = model.predict(x_batch, verbose=0).argmax(axis=1)
        all_preds.extend(preds.tolist())
        all_labels.extend(y_batch.numpy().tolist())
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    accs = []
    for c in range(n_classes):
        mask = all_labels == c
        accs.append((all_preds[mask] == c).mean() if mask.sum() > 0 else 0.0)
    return np.array(accs)


# ════════════════════════════════════════════════════════════════════════════
# 5. Callbacks
# ════════════════════════════════════════════════════════════════════════════

def make_callbacks(output_dir: pathlib.Path):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4,
            min_lr=1e-7, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(output_dir / "history.csv")),
    ]


# ════════════════════════════════════════════════════════════════════════════
# 6. Point d'entrée
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=CONFIG["data_dir"])
    args = parser.parse_args()

    data_dir   = pathlib.Path(args.data_dir)
    output_dir = pathlib.Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Données ─────────────────────────────────────────────────────────
    print("[1] Chargement des images ISIC 2019 ...")
    print(f"    Répertoire : {data_dir.resolve()}")
    paths, labels = load_metadata(data_dir)
    print(f"    Total : {len(paths):,} images  |  {len(CLASSES)} classes")

    paths_train, paths_val, y_train, y_val = train_test_split(
        paths, labels,
        test_size=CONFIG["val_split"],
        stratify=labels,
        random_state=CONFIG["seed"],
    )
    print(f"    Train : {len(paths_train):,}  |  Val : {len(paths_val):,}")

    # ── Pré-cache ────────────────────────────────────────────────────────
    print("\n[2] Pré-traitement et mise en cache des images ...")
    cache_dir = pathlib.Path("data/isic2019_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    imgs_train, y_train = get_or_build_cache(
        paths_train, y_train, cache_dir / "train.npz")
    imgs_val, y_val = get_or_build_cache(
        paths_val, y_val, cache_dir / "val_ss.npz")

    ds_train = make_dataset(imgs_train, y_train,
                            CONFIG["batch_size"], augment=True, shuffle=True)
    ds_val   = make_dataset(imgs_val, y_val,
                            CONFIG["batch_size"], augment=False, shuffle=False)

    n_classes = len(CLASSES)

    # ── Modèle ───────────────────────────────────────────────────────────
    print(f"\n[3] Construction du modèle (full FT, sans class_weights) ...")
    model = build_model(n_classes)
    n_bn = sum(1 for l in model.layers[1].layers
               if isinstance(l, tf.keras.layers.BatchNormalization))
    print(f"    {n_bn} BatchNorm layers gelées (stabilité Metal GPU)")
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            CONFIG["lr"], weight_decay=CONFIG["weight_decay"]
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    n_params = sum(int(tf.size(v)) for v in model.trainable_variables)
    print(f"    Paramètres entraînables : {n_params:,}")

    # ── Entraînement ─────────────────────────────────────────────────────
    print(f"\n[4] Entraînement (max {CONFIG['epochs']} epochs, lr={CONFIG['lr']}) ...")
    model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=CONFIG["epochs"],
        callbacks=make_callbacks(output_dir),
        verbose=1,
    )

    # ── Évaluation finale ────────────────────────────────────────────────
    print("\n[5] Évaluation finale ...")
    loss, acc = model.evaluate(ds_val, verbose=0)
    print(f"    Loss     : {loss:.4f}")
    print(f"    Accuracy : {acc:.4f}  ({acc*100:.2f}%)")

    print("\n    Accuracy par classe :")
    per_cls = per_class_accuracy(model, ds_val, n_classes)
    for i, cls in enumerate(CLASSES):
        print(f"      [{i}] {cls:4s} : {per_cls[i]*100:.1f}%")
    bal_acc = per_cls.mean()
    print(f"\n    Balanced accuracy : {bal_acc:.4f}  ({bal_acc*100:.2f}%)")

    # ── Sauvegarde ────────────────────────────────────────────────────────
    print("\n[6] Sauvegarde ...")
    final_path = output_dir / f"{CONFIG['model_name']}.keras"
    model.save(str(final_path))
    print(f"    {final_path}")

    # Copie à la racine pour les scripts Hessian Surgery
    root_path = pathlib.Path(f"{CONFIG['model_name']}.keras")
    model.save(str(root_path))
    print(f"    {root_path}  (racine, pour Hessian Surgery)")

    summary = pd.DataFrame([{
        "val_loss"     : loss,
        "val_accuracy" : acc,
        "balanced_acc" : bal_acc,
        **{f"acc_{CLASSES[i]}": per_cls[i] for i in range(n_classes)},
    }])
    summary.to_csv(output_dir / "final_metrics.csv", index=False)
    print(f"    {output_dir}/final_metrics.csv")
