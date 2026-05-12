"""
ss_cifar10_homogeneous.py
-------------------------
Hessian Surgery seul sur CIFAR-10 baseline (CE), omega_mode=homogeneous,
15 itérations. Objectif : tester si le mode uniforme produit un gain net
d'accuracy globale (et pas seulement de redistribution).

Usage :
    python3.12 -u ss_cifar10_homogeneous.py 2>&1 | tee \
        results/cifar10/ss/log.txt
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf

from hessian_surgery import HessianSurgery

CFG = {
    "model_path"        : "resnet50_cifar10.keras",
    "class_names"       : ["plane","car","bird","cat","deer",
                           "dog","frog","horse","ship","truck"],
    "n_hvp_samples"     : 128,
    "lanczos_m"         : 10,
    "n_spikes"          : 9,
    "n_iter"            : 15,
    "patience"          : 4,
    "omega_mode"        : "homogeneous",
    "max_degrade_total" : 0.06,
    "max_degrade_iter"  : 0.03,
    "alpha_max_init"    : 0.02,
    "alpha_min"         : 0.002,
    "beta_ema"          : 0.7,
    "rollback_std_tol"  : 0.005,
    "rollback_drop_tol" : 0.07,
    "per_spike_budget"  : False,
    "output_dir"        : "results/cifar10/ss",
    "model_out"         : "resnet50_cifar10_ss.keras",
    "save_model"        : True,
    "seed"              : 0,
}

if __name__ == "__main__":
    os.makedirs(CFG["output_dir"], exist_ok=True)

    print("[1] Chargement CIFAR-10 ...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train = y_train.flatten().astype(np.int32)
    y_test  = y_test.flatten().astype(np.int32)
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0

    rng      = np.random.default_rng(CFG["seed"])
    idx_test = rng.permutation(len(x_test))
    x_sens, y_sens = x_test[idx_test[:5000]], y_test[idx_test[:5000]]
    x_eval, y_eval = x_test[idx_test[5000:]], y_test[idx_test[5000:]]
    print(f"    Sensitivity : {len(x_sens)}  |  Held-out : {len(x_eval)}")

    hvp_idx = rng.choice(len(x_train), CFG["n_hvp_samples"], replace=False)
    x_hvp   = x_train[hvp_idx]
    y_hvp   = y_train[hvp_idx]
    print(f"    HVP batch   : {len(x_hvp)}")

    print("[2] Chargement modèle ...")
    model   = tf.keras.models.load_model(CFG["model_path"], compile=False)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
    print(f"    {CFG['model_path']}  -  "
          f"{sum(np.prod(v.shape) for v in model.trainable_variables):,} params")

    runner = HessianSurgery(
        model, loss_fn,
        x_sens, y_sens,
        x_eval, y_eval,
        x_eval, y_eval,
        x_hvp,  y_hvp,
        CFG,
    )
    runner.run()
    print(f"\n  Done -> {CFG['output_dir']}/\n")
