"""
spike_optimizer_isic2019_cb.py
------------------------------
Spectral Surgery sur checkpoint Class-Balanced — ISIC-2019.
Thin wrapper autour de isic_ss.SpikeOptimizerISIC.

Spécificités CB vs CE/FL :
  - HVP balancé par classe (estimateur de la Hessienne de la loss balancée)
  - β1_a/β2_a Adam scalés sur n_iter (`beta_adaptive=True`)
  - Checkpoint best-on-val parallèle (`save_best_on_val=True`)

Usage : python spike_optimizer_isic2019_cb.py
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import tensorflow as tf

from isic_ss import (
    SpikeOptimizerISIC, ISIC_CLASSES,
    per_class_accuracy, load_isic_caches,
    sample_hvp_balanced, sample_sensitivity_stratified,
)

CONFIG = {
    "model_path"        : "results/isic2019/class_balanced/best.keras",
    "cache_train"       : "data/isic2019_cache/train.npz",
    "cache_val"         : "data/isic2019_cache/val_ss.npz",
    "cache_test"        : "data/isic2019_cache/test_ss.npz",
    "n_hvp_samples"     : 256,
    "hvp_batch_size"    : 32,
    "seed"              : 0,
    "lanczos_m"         : 10,
    "n_spikes"          : 7,
    "eps_probe"         : 0.01,
    "alpha_max_init"    : 0.01,
    "alpha_min"         : 0.001,
    "beta_ema"          : 0.7,
    "sens_max_per_class": 250,
    "save_model"        : True,
    "model_out"         : "resnet50_isic2019_cb_ss.keras",
    "n_iter"            : 5,
    "patience"          : 3,
    "output_dir"        : "results/isic2019/cb_ss",

    # CB-specific
    "beta_adaptive"     : True,
    "save_best_on_val"  : True,
}


if __name__ == "__main__":
    rng = np.random.default_rng(CONFIG["seed"])
    print("[1] Chargement caches ISIC ...")
    x_val_mon, y_val_mon, x_eval, y_eval, x_train, y_train = load_isic_caches(CONFIG)
    print(f"    val_mon={len(x_val_mon)}  test={len(x_eval)}  train={len(x_train)}")

    print("[2] HVP (balancé) + sensitivity (stratifié) ...")
    x_hvp, y_hvp = sample_hvp_balanced(x_train, y_train, CONFIG["n_hvp_samples"], rng)
    x_sens, y_sens = sample_sensitivity_stratified(
        x_train, y_train, CONFIG["sens_max_per_class"], rng)
    del x_train, y_train

    print("[3] Modèle ...")
    model   = tf.keras.models.load_model(CONFIG["model_path"], compile=False)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

    print("[4] Baseline val ...")
    acc_val  = per_class_accuracy(model, x_val_mon, y_val_mon)
    val_glob = model.evaluate(x_val_mon, y_val_mon, verbose=0, batch_size=64)[1]
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    pd.DataFrame([{
        "acc_global"  : float(val_glob),
        "balanced_acc": float(acc_val.mean()),
        "std"         : float(np.std(acc_val)),
        **{n: float(acc_val[c]) for c, n in enumerate(ISIC_CLASSES)},
    }]).to_csv(os.path.join(CONFIG["output_dir"], "baseline_val.csv"), index=False)

    SpikeOptimizerISIC(
        model=model, loss_fn=loss_fn,
        x_sens=x_sens, y_sens=y_sens,
        x_eval=x_eval, y_eval=y_eval,
        x_val_mon=x_val_mon, y_val_mon=y_val_mon,
        x_hvp=x_hvp, y_hvp=y_hvp,
        cfg=CONFIG,
    ).run()
