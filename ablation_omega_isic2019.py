"""
ablation_omega_isic2019.py
---------------------------
Ablation sur le mode omega (sqrt / linear / square) de Hessian Surgery
appliquée au baseline ResNet-50/ISIC-2019 (sans Focal Loss).

10 itérations par run. Budget per-spike activé (défaut ISIC).
Sensitivity set : train stratifié (max 250/classe).
Val set : val_ss.npz  (monitoring).
Test set : test_ss.npz (éval finale uniquement).

Usage :
    python3.12 -u ablation_omega_isic2019.py 2>&1 | tee results/isic2019/ablation_omega/log.txt
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf

from hessian_surgery import HessianSurgery

CLASSES    = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
CACHE_DIR  = "data/isic2019_cache"

BASE_CFG = {
    "model_path"        : "resnet50_isic2019.keras",
    "class_names"       : CLASSES,
    "n_hvp_samples"     : 64,
    "lanczos_m"         : 10,
    "n_spikes"          : 7,
    "n_iter"            : 10,
    "patience"          : 3,
    "omega_mode"        : "linear",     # surchargé à chaque run
    "max_degrade_total" : 0.06,
    "max_degrade_iter"  : 0.03,
    "alpha_max_init"    : 0.01,
    "alpha_min"         : 0.001,
    "beta_ema"          : 0.7,
    "rollback_std_tol"  : 0.005,
    "rollback_drop_tol" : 0.07,
    "per_spike_budget"  : True,
    "sens_max_per_class": 250,
    "output_dir"        : "results/isic2019/ablation_omega/linear",  # surchargé
    "model_out"         : "resnet50_isic2019_ss_omega_linear.keras",
    "save_model"        : True,
    "seed"              : 0,
}

OMEGA_RUNS = [
    ("sqrt",   "results/isic2019/ablation_omega/sqrt",   "resnet50_isic2019_ss_omega_sqrt.keras"),
    ("linear", "results/isic2019/ablation_omega/linear", "resnet50_isic2019_ss_omega_linear.keras"),
    ("square", "results/isic2019/ablation_omega/square", "resnet50_isic2019_ss_omega_square.keras"),
]

if __name__ == "__main__":
    os.makedirs("results/isic2019/ablation_omega", exist_ok=True)
    rng = np.random.default_rng(BASE_CFG["seed"])

    # ── Données — chargées une seule fois ─────────────────────────────────
    print("[1] Chargement données ISIC-2019 ...")

    train_data = np.load(f"{CACHE_DIR}/train.npz")
    x_tr = train_data["imgs"].astype(np.float32)
    y_tr = train_data["labels"].astype(np.int32)

    cap = BASE_CFG["sens_max_per_class"]
    idx_sens = np.concatenate([
        rng.choice(np.where(y_tr == c)[0],
                   min(cap, (y_tr == c).sum()), replace=False)
        for c in range(len(CLASSES))
    ])
    x_sens, y_sens = x_tr[idx_sens], y_tr[idx_sens]
    print(f"    Sensitivity set : {len(x_sens)} imgs  "
          f"({dict(zip(CLASSES, [(y_sens==c).sum() for c in range(len(CLASSES))]))})")

    val_data  = np.load(f"{CACHE_DIR}/val_ss.npz")
    x_val  = val_data["imgs"].astype(np.float32)
    y_val  = val_data["labels"].astype(np.int32)

    test_data = np.load(f"{CACHE_DIR}/test_ss.npz")
    x_test = test_data["imgs"].astype(np.float32)
    y_test = test_data["labels"].astype(np.int32)

    hvp_idx = rng.choice(len(x_tr), BASE_CFG["n_hvp_samples"], replace=False)
    x_hvp, y_hvp = x_tr[hvp_idx].astype(np.float32), y_tr[hvp_idx].astype(np.int32)
    print(f"    HVP batch : {len(x_hvp)}  |  Val : {len(x_val)}  |  Test : {len(x_test)}")

    del x_tr, y_tr, train_data  # libère mémoire (~13 GB)

    # ── 3 runs séquentiels ────────────────────────────────────────────────
    for omega, out_dir, model_out in OMEGA_RUNS:
        print(f"\n{'#'*72}")
        print(f"#  OMEGA = {omega.upper()}")
        print(f"{'#'*72}\n")

        cfg = {**BASE_CFG, "omega_mode": omega,
               "output_dir": out_dir, "model_out": model_out}
        os.makedirs(out_dir, exist_ok=True)

        print("[2] Chargement modèle ...")
        model   = tf.keras.models.load_model(cfg["model_path"], compile=False)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
        print(f"    {cfg['model_path']}  —  "
              f"{sum(np.prod(v.shape) for v in model.trainable_variables):,} params")

        runner = HessianSurgery(
            model, loss_fn,
            x_sens, y_sens,
            x_val,  y_val,
            x_test, y_test,
            x_hvp,  y_hvp,
            cfg,
        )
        runner.run()
        print(f"\n  Run omega={omega} terminé → {out_dir}/\n")
