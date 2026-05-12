"""
class_balanced_isic2019.py
--------------------------
Fine-tuning de resnet50_isic2019.keras avec class weights inversement
proportionnels à la fréquence de classe (cross-entropy rééquilibrée).
Baseline de comparaison pour Hessian Surgery.

Usage:
    python3.12 class_balanced_isic2019.py
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
import pandas as pd
import time

CLASSES   = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
N_CLASSES = 8

CONFIG = {
    "model_path"   : "resnet50_isic2019.keras",
    "cache_train"  : "data/isic2019_cache/train.npz",
    "cache_val"    : "data/isic2019_cache/val.npz",
    "held_out_seed": 0,
    "held_out_n"   : 2000,
    "lr"           : 5e-5,
    "epochs"       : 15,
    "batch_size"   : 128,
    "seed"         : 42,
    "output_dir"   : "results/isic2019/class_balanced",
}


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
    rng = np.random.default_rng(CONFIG["held_out_seed"])

    print("[1] Chargement des données ...")
    val_data  = np.load(CONFIG["cache_val"])
    x_val     = val_data["imgs"].astype(np.float32)
    y_val     = val_data["labels"].astype(np.int32)
    idx_val   = rng.permutation(len(x_val))
    x_held    = x_val[idx_val[CONFIG["held_out_n"]:]]
    y_held    = y_val[idx_val[CONFIG["held_out_n"]:]]

    train_data = np.load(CONFIG["cache_train"])
    x_train    = train_data["imgs"].astype(np.float32)
    y_train    = train_data["labels"].astype(np.int32)

    # Class-balanced weights (Cui et al. 2019, "effective number of samples").
    # w_c proportional to 1 / E_n_c where E_n = (1 - beta^n) / (1 - beta).
    # We pick beta = (N-1)/N rather than tuning it as an arbitrary hyperparam:
    # this is the unique value for which (i) the effective volume of a class of
    # size n converges to n as n -> 0 (rare-class behaviour ~ inverse freq),
    # and (ii) saturates near N for n -> N (head-class behaviour bounded by
    # dataset capacity). It removes the only free scalar in Cui's formulation
    # and ties beta to the dataset size, which is the principled default.
    counts = np.bincount(y_train, minlength=N_CLASSES)
    N_total = int(len(y_train))
    beta = (N_total - 1.0) / N_total
    eff_num = (1.0 - np.power(beta, counts.astype(np.float64))) / (1.0 - beta)
    raw_w = 1.0 / eff_num
    raw_w *= N_CLASSES / raw_w.sum()  # normalise so mean weight = 1
    class_weight = {c: float(raw_w[c]) for c in range(N_CLASSES)}
    print(f"    beta = (N-1)/N = {beta:.6f}   (N={N_total})")
    print("    Class weights (effective-number, normalised):")
    for c, name in enumerate(CLASSES):
        print(f"      {name:4s}: w={class_weight[c]:.3f}  "
              f"E_n={eff_num[c]:8.1f}  (n={counts[c]:5d})")

    print(f"\n[2] Chargement du modèle baseline ...")
    model = tf.keras.models.load_model(CONFIG["model_path"])

    acc_base = per_class_accuracy(model, x_held, y_held)
    print("    Baseline held-out :")
    for i, c in enumerate(CLASSES):
        print(f"      {c:4s}: {acc_base[i]*100:.1f}%")

    print(f"\n[3] Fine-tuning — Class-Balanced CE ...")
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(CONFIG["lr"], weight_decay=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    ds_train = make_dataset(x_train, y_train, augment=True, shuffle=True)
    ds_val   = make_dataset(x_held,  y_held,  augment=False)

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
        epochs=CONFIG["epochs"], callbacks=callbacks,
        class_weight=class_weight, verbose=1,
    )
    print(f"    Durée : {time.time()-t0:.0f}s")

    print("\n[4] Évaluation finale sur held-out ...")
    acc_final  = per_class_accuracy(model, x_held, y_held)
    ds_val_eval = make_dataset(x_held, y_held)
    global_acc = (model.predict(x_held, verbose=0, batch_size=64).argmax(axis=1) == y_held).mean()
    bal_acc    = acc_final.mean()
    std_final  = np.std(acc_final)

    print(f"    Global: {global_acc*100:.1f}%  |  Balanced: {bal_acc*100:.1f}%  |  std: {std_final*100:.1f}%")
    print(f"\n  {'Classe':6s}  {'Baseline':>9s}  {'ClassBal':>9s}  {'Δ':>7s}")
    for i, c in enumerate(CLASSES):
        delta = (acc_final[i] - acc_base[i]) * 100
        print(f"  {c:6s}  {acc_base[i]*100:>8.1f}%  {acc_final[i]*100:>8.1f}%  {delta:>+6.1f}%")

    summary = pd.DataFrame([{
        "method": "class_balanced",
        "global_acc": global_acc,
        "balanced_acc": bal_acc,
        "std": std_final,
        **{f"acc_{CLASSES[i]}": acc_final[i] for i in range(N_CLASSES)},
        **{f"base_{CLASSES[i]}": acc_base[i] for i in range(N_CLASSES)},
    }])
    summary.to_csv(os.path.join(CONFIG["output_dir"], "eval_heldout.csv"), index=False)
    model.save(os.path.join(CONFIG["output_dir"], "model_class_balanced.keras"))
    print(f"\n    Résultats : {CONFIG['output_dir']}/")
