"""
tta_bootstrap_eval.py
---------------------
Évaluation robuste sur test set :
  1. TTA (N augmentations par image, moyenne des softmax)
  2. Bootstrap sur les prédictions TTA → IC 95% par classe

Usage:
    python3.12 tta_bootstrap_eval.py
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
import pandas as pd

CLASSES   = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
N_CLASSES = 8

CONFIG = {
    "models": {
        "CB"      : "results/isic2019/class_balanced/best.keras",
        "CB+SS"   : "resnet50_isic2019_cb_ss.keras",
    },
    "cache_test"   : "data/isic2019_cache/test_ss.npz",
    "n_tta"        : 50,    # augmentations par image
    "n_bootstrap"  : 2000,  # itérations bootstrap
    "batch_size"   : 128,
    "seed"         : 42,
    "output_dir"   : "results/isic2019/tta_bootstrap",
    "IMG_SIZE"     : 224,
}

IMG_SIZE = CONFIG["IMG_SIZE"]


def augment_single(img):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, 0.1)
    img = tf.image.random_contrast(img, 0.9, 1.1)
    img = tf.image.resize_with_crop_or_pad(img, IMG_SIZE + 16, IMG_SIZE + 16)
    img = tf.image.random_crop(img, [IMG_SIZE, IMG_SIZE, 3])
    return img


def tta_predict(model, x, n_tta, batch_size):
    """Retourne softmax moyennée sur n_tta augmentations. Shape: (N, C)"""
    n = len(x)
    probs_sum = np.zeros((n, N_CLASSES), dtype=np.float32)
    for _ in range(n_tta):
        x_aug = np.stack([augment_single(img).numpy() for img in x])
        probs = model.predict(x_aug, verbose=0, batch_size=batch_size)
        probs_sum += probs
    return probs_sum / n_tta


def bootstrap_ci(correct, n_boot, seed):
    """Bootstrap IC 95% sur accuracy binaire (correct[i] ∈ {0,1})."""
    rng  = np.random.default_rng(seed)
    n    = len(correct)
    boot = np.array([rng.choice(correct, n, replace=True).mean() for _ in range(n_boot)])
    return boot.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)


if __name__ == "__main__":
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    print("[1] Chargement test set ...")
    test_data = np.load(CONFIG["cache_test"])
    x_test    = test_data["imgs"].astype(np.float32)
    y_test    = test_data["labels"].astype(np.int32)
    print(f"    {len(x_test)} images")
    for c, name in enumerate(CLASSES):
        print(f"      {name}: {(y_test==c).sum()}")

    all_results = {}

    for model_name, model_path in CONFIG["models"].items():
        if not os.path.exists(model_path):
            print(f"\n[SKIP] {model_name} — fichier introuvable : {model_path}")
            continue

        print(f"\n{'='*60}")
        print(f"  Modèle : {model_name}")
        print(f"{'='*60}")

        model = tf.keras.models.load_model(model_path, compile=False)

        print(f"  TTA ({CONFIG['n_tta']} passes) ...")
        probs = tta_predict(model, x_test, CONFIG["n_tta"], CONFIG["batch_size"])
        preds = probs.argmax(axis=1)

        # Accuracy simple (TTA)
        global_acc = (preds == y_test).mean()
        print(f"  Global TTA : {global_acc*100:.1f}%")

        # Bootstrap par classe
        rows = []
        print(f"\n  {'Classe':6s}  {'n':>4s}  {'Acc TTA':>8s}  {'IC 95%':>18s}")
        for c, cname in enumerate(CLASSES):
            mask    = y_test == c
            n_c     = mask.sum()
            correct = (preds[mask] == c).astype(float)
            if n_c == 0:
                continue
            acc_tta = correct.mean()
            mean_b, lo, hi = bootstrap_ci(correct, CONFIG["n_bootstrap"], CONFIG["seed"] + c)
            print(f"  {cname:6s}  {n_c:>4d}  {acc_tta*100:>7.1f}%  [{lo*100:>5.1f}%, {hi*100:>5.1f}%]")
            rows.append({
                "model": model_name, "class": cname, "n": n_c,
                "acc_tta": acc_tta, "boot_mean": mean_b, "ci_lo": lo, "ci_hi": hi,
            })

        bal_acc = np.mean([r["acc_tta"] for r in rows if r["model"] == model_name])
        print(f"\n  bal_acc TTA : {bal_acc*100:.1f}%  global: {global_acc*100:.1f}%")
        all_results[model_name] = rows
        tf.keras.backend.clear_session()

    # Sauvegarde
    all_rows = [r for rows in all_results.values() for r in rows]
    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(CONFIG["output_dir"], "tta_bootstrap_cb_results.csv"), index=False)

    # Tableau synthétique VASC uniquement
    print(f"\n{'='*60}")
    print("  VASC — comparaison inter-modèles (TTA + IC 95%)")
    print(f"{'='*60}")
    vasc_rows = df[df["class"] == "VASC"]
    for _, r in vasc_rows.iterrows():
        print(f"  {r['model']:15s}  {r['acc_tta']*100:>5.1f}%  [{r['ci_lo']*100:>5.1f}%, {r['ci_hi']*100:>5.1f}%]  (n={r['n']})")

    print(f"\n  Résultats : {CONFIG['output_dir']}/tta_bootstrap_results.csv")
