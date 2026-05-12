"""
ablation_omega_cifar10.py
--------------------------
Ablation sur le mode omega (sqrt / linear / square) de Spectral Surgery
appliquée au baseline ResNet-50/CIFAR-10 (sans Focal Loss).

Enchaîne 3 runs séquentiels sur le même jeu de données chargé une seule fois.

Usage :
    python3.12 -u ablation_omega_cifar10.py 2>&1 | tee results/cifar10/ablation_omega/log.txt
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf

from spectral_surgery import SpectralSurgery

# ════════════════════════════════════════════════════════════════════════════
# CONFIG de base — partagée entre les 3 runs, omega_mode surchargé à l'exécution
# ════════════════════════════════════════════════════════════════════════════

BASE_CFG = {
    "model_path"        : "resnet50_cifar10.keras",
    "class_names"       : ["plane","car","bird","cat","deer",
                           "dog","frog","horse","ship","truck"],
    "n_hvp_samples"     : 128,
    "lanczos_m"         : 10,
    "n_spikes"          : 9,          # C-1 = 9
    "n_iter"            : 15,
    "patience"          : 4,
    "omega_mode"        : "linear",   # surchargé à chaque run
    "max_degrade_total" : 0.06,
    "max_degrade_iter"  : 0.03,
    "alpha_max_init"    : 0.02,
    "alpha_min"         : 0.002,
    "beta_ema"          : 0.7,
    "rollback_std_tol"  : 0.005,
    "rollback_drop_tol" : 0.07,
    "per_spike_budget"  : False,   # L2 norm globale pour CIFAR-10 (ratio λ_max/λ_min ≈ 40)
    "output_dir"        : "results/cifar10/ablation_omega/linear",   # surchargé
    "model_out"         : "resnet50_cifar10_ss_omega_linear.keras",
    "save_model"        : True,
    "seed"              : 0,
}

OMEGA_RUNS = [
    ("sqrt",   "results/cifar10/ablation_omega/sqrt",   "resnet50_cifar10_ss_omega_sqrt.keras"),
    ("linear", "results/cifar10/ablation_omega/linear", "resnet50_cifar10_ss_omega_linear.keras"),
    ("square", "results/cifar10/ablation_omega/square", "resnet50_cifar10_ss_omega_square.keras"),
]

# ════════════════════════════════════════════════════════════════════════════
# Données — chargées une seule fois pour les 3 runs
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.makedirs("results/cifar10/ablation_omega", exist_ok=True)

    print("[1] Chargement CIFAR-10 ...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train = y_train.flatten().astype(np.int32)
    y_test  = y_test.flatten().astype(np.int32)
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0

    rng      = np.random.default_rng(BASE_CFG["seed"])
    idx_test = rng.permutation(len(x_test))
    x_sens, y_sens = x_test[idx_test[:5000]], y_test[idx_test[:5000]]
    x_eval, y_eval = x_test[idx_test[5000:]], y_test[idx_test[5000:]]
    print(f"    Sensitivity : {len(x_sens)}  |  Held-out : {len(x_eval)}")

    hvp_idx = rng.choice(len(x_train), BASE_CFG["n_hvp_samples"], replace=False)
    x_hvp   = x_train[hvp_idx]
    y_hvp   = y_train[hvp_idx]
    print(f"    HVP batch   : {len(x_hvp)}")

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

        runner = SpectralSurgery(
            model, loss_fn,
            x_sens, y_sens,   # sensitivity set
            x_eval, y_eval,   # val = held-out (monitoring + éval finale)
            x_eval, y_eval,   # test = held-out (éval finale)
            x_hvp,  y_hvp,
            cfg,
        )
        runner.run()
        print(f"\n  Run omega={omega} terminé → {out_dir}/\n")
