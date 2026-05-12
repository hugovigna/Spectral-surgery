"""
focal_loss_isic2019.py
----------------------
Fine-tuning de resnet50_isic2019.keras avec Focal Loss (γ=2) sur ISIC 2019.
Baseline de comparaison pour Spectral Surgery.

Usage:
    python3.12 focal_loss_isic2019.py
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
import pandas as pd
import time

CLASSES  = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
N_CLASSES = 8

CONFIG = {
    "model_path"  : "resnet50_isic2019.keras",
    "cache_train" : "data/isic2019_cache/train.npz",
    "cache_val"   : "data/isic2019_cache/val_ss.npz",   # monitoring pendant FT
    "cache_test"  : "data/isic2019_cache/test_ss.npz",  # éval finale uniquement
    "lr"          : 5e-5,
    "epochs"      : 10,
    "batch_size"  : 128,
    "gamma"       : 2.0,
    "seed"        : 42,
    "output_dir"  : "results/isic2019/focal_loss",
}

# ── Focal Loss ────────────────────────────────────────────────────────────────
class SparseCategoricalFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma

    def call(self, y_true, y_pred):
        y_true   = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        # proba de la vraie classe
        p_t      = tf.reduce_sum(
            y_pred * tf.one_hot(y_true, N_CLASSES), axis=-1)
        p_t      = tf.clip_by_value(p_t, 1e-7, 1.0)
        focal_w  = tf.pow(1.0 - p_t, self.gamma)
        return -tf.reduce_mean(focal_w * tf.math.log(p_t))


def per_class_accuracy(model, x, y):
    preds = model.predict(x, verbose=0, batch_size=CONFIG["batch_size"]).argmax(axis=1)
    return np.array(
        [(preds[y == c] == c).mean() if (y == c).sum() > 0 else 0.0
         for c in range(N_CLASSES)]
    )

def make_dataset(imgs, labels, augment=False, shuffle=False):
    IMG_SIZE = 224
    with tf.device("/CPU:0"):
        ds = tf.data.Dataset.from_tensor_slices(
            (imgs.astype(np.float32), labels.astype(np.int32))
        )
    if shuffle:
        ds = ds.shuffle(len(imgs), seed=CONFIG["seed"])
    if augment:
        def aug(img, lbl):
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)
            img = tf.image.random_brightness(img, 0.15)
            img = tf.image.random_contrast(img, 0.85, 1.15)
            img = tf.image.resize_with_crop_or_pad(img, IMG_SIZE + 20, IMG_SIZE + 20)
            img = tf.image.random_crop(img, [IMG_SIZE, IMG_SIZE, 3])
            return img, lbl
        ds = ds.map(aug, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(CONFIG["batch_size"]).prefetch(tf.data.AUTOTUNE)


if __name__ == "__main__":
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    rng = np.random.default_rng(CONFIG["seed"])

    print("[1] Chargement des données ...")
    val_data   = np.load(CONFIG["cache_val"])
    x_val_mon  = val_data["imgs"].astype(np.float32)
    y_val_mon  = val_data["labels"].astype(np.int32)

    test_data  = np.load(CONFIG["cache_test"])
    x_held     = test_data["imgs"].astype(np.float32)
    y_held     = test_data["labels"].astype(np.int32)

    train_data = np.load(CONFIG["cache_train"])
    x_train    = train_data["imgs"].astype(np.float32)
    y_train    = train_data["labels"].astype(np.int32)
    print(f"    Train: {len(x_train):,}  |  Val monitor: {len(x_val_mon):,}  |  Test held-out: {len(x_held):,}")

    print("[2] Chargement du modèle baseline ...")
    model = tf.keras.models.load_model(CONFIG["model_path"])

    # Baseline sur held-out
    acc_base = per_class_accuracy(model, x_held, y_held)
    print("    Baseline held-out :")
    for i, c in enumerate(CLASSES):
        print(f"      {c:4s}: {acc_base[i]*100:.1f}%")

    print(f"\n[3] Fine-tuning — Focal Loss γ={CONFIG['gamma']} ...")
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(CONFIG["lr"], weight_decay=1e-4),
        loss=SparseCategoricalFocalLoss(gamma=CONFIG["gamma"]),
        metrics=["accuracy"],
    )

    ds_train = make_dataset(x_train,   y_train,   augment=True,  shuffle=True)
    ds_val   = make_dataset(x_val_mon, y_val_mon, augment=False)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(CONFIG["output_dir"], "best.keras"),
            monitor="val_accuracy", save_best_only=True, verbose=0,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=4,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3,
            min_lr=1e-7, verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(CONFIG["output_dir"], "history.csv")),
    ]

    t0 = time.time()
    model.fit(
        ds_train, validation_data=ds_val,
        epochs=CONFIG["epochs"], callbacks=callbacks, verbose=1,
    )
    print(f"    Durée : {time.time()-t0:.0f}s")

    print("\n[4] Évaluation finale sur held-out ...")
    acc_final  = per_class_accuracy(model, x_held, y_held)
    loss_val   = model.evaluate(ds_val, verbose=0)[0]
    bal_acc    = acc_final.mean()
    std_final  = np.std(acc_final)
    global_acc = (model.predict(x_held, verbose=0, batch_size=64).argmax(axis=1) == y_held).mean()

    print(f"    Global: {global_acc*100:.1f}%  |  Balanced: {bal_acc*100:.1f}%  |  std: {std_final*100:.1f}%")
    print(f"\n  {'Classe':6s}  {'Baseline':>9s}  {'FocalLoss':>9s}  {'Δ':>7s}")
    for i, c in enumerate(CLASSES):
        delta = (acc_final[i] - acc_base[i]) * 100
        print(f"  {c:6s}  {acc_base[i]*100:>8.1f}%  {acc_final[i]*100:>8.1f}%  {delta:>+6.1f}%")

    summary = pd.DataFrame([{
        "method": "focal_loss",
        "gamma": CONFIG["gamma"],
        "global_acc": global_acc,
        "balanced_acc": bal_acc,
        "std": std_final,
        **{f"acc_{CLASSES[i]}": acc_final[i] for i in range(N_CLASSES)},
        **{f"base_{CLASSES[i]}": acc_base[i] for i in range(N_CLASSES)},
    }])
    summary.to_csv(os.path.join(CONFIG["output_dir"], "eval_heldout.csv"), index=False)
    model.save(os.path.join(CONFIG["output_dir"], "model_focal.keras"))
    print(f"\n    Résultats : {CONFIG['output_dir']}/")
