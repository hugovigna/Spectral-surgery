"""
eval_cifar100_heldout.py
Évaluation held-out du modèle CIFAR-100 après Deflated Surgery (phase 3).
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
import pandas as pd

sys_path = os.path.dirname(__file__)
import sys; sys.path.insert(0, sys_path)
from spike_optimizer_cifar100 import per_class_accuracy, CIFAR100_CLASSES

OUTPUT_DIR = "results/deflated_surgery_cifar100"
MODEL_PATH = os.path.join(OUTPUT_DIR, "phase3", "model.keras")

# Données
print("[1] Chargement de CIFAR-100 ...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
y_train, y_test = y_train.flatten(), y_test.flatten()
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0

# Même split que le script principal (seed=0)
rng = np.random.default_rng(0)
idx_test = rng.permutation(len(x_test))
x_sens_pool, y_sens_pool = x_test[idx_test[:5000]], y_test[idx_test[:5000]]
x_eval, y_eval = x_test[idx_test[5000:]], y_test[idx_test[5000:]]
print(f"    Split : {len(x_sens_pool)} sens + {len(x_eval)} eval")

# Modèle après phase 3
print(f"[2] Chargement du modèle : {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

# Baseline (modèle original)
print("[3] Baseline (modèle original) ...")
model_orig = tf.keras.models.load_model("resnet50_cifar100.keras")
acc_orig_sens = per_class_accuracy(model_orig, x_sens_pool, y_sens_pool)
acc_orig_eval = per_class_accuracy(model_orig, x_eval, y_eval)
g_orig_sens = model_orig.evaluate(x_sens_pool, y_sens_pool, verbose=0, batch_size=256)[1]
g_orig_eval = model_orig.evaluate(x_eval, y_eval, verbose=0, batch_size=256)[1]
del model_orig

# Après surgery
print("[4] Évaluation après Surgery ...")
acc_sens = per_class_accuracy(model, x_sens_pool, y_sens_pool)
acc_eval = per_class_accuracy(model, x_eval, y_eval)
g_sens = model.evaluate(x_sens_pool, y_sens_pool, verbose=0, batch_size=256)[1]
g_eval = model.evaluate(x_eval, y_eval, verbose=0, batch_size=256)[1]

# Résultats
print(f"\n{'='*70}")
print(f"  RÉSULTATS CIFAR-100 — Deflated Surgery (3 phases, 45 spikes)")
print(f"{'='*70}")
print(f"\n  --- Set sensibilité (5000 images, pilotage) ---")
print(f"  Baseline : Global={g_orig_sens*100:.1f}%  Std={np.std(acc_orig_sens)*100:.2f}%")
print(f"  Surgery  : Global={g_sens*100:.1f}%  Std={np.std(acc_sens)*100:.2f}%")
print(f"  Δσ = {(np.std(acc_sens)-np.std(acc_orig_sens))*100:+.2f} pp")

print(f"\n  --- Set held-out (5000 images, non contaminé) ---")
print(f"  Baseline : Global={g_orig_eval*100:.1f}%  Std={np.std(acc_orig_eval)*100:.2f}%")
print(f"  Surgery  : Global={g_eval*100:.1f}%  Std={np.std(acc_eval)*100:.2f}%")
print(f"  Δσ = {(np.std(acc_eval)-np.std(acc_orig_eval))*100:+.2f} pp")

# CSV eval held-out
eval_summary = {
    "acc_global_eval": float(g_eval),
    "std_eval": float(np.std(acc_eval)),
    "acc_global_baseline_eval": float(g_orig_eval),
    "std_baseline_eval": float(np.std(acc_orig_eval)),
    **{f"eval_{CIFAR100_CLASSES[c]}": float(acc_eval[c]) for c in range(100)},
    **{f"baseline_{CIFAR100_CLASSES[c]}": float(acc_orig_eval[c]) for c in range(100)},
}
pd.DataFrame([eval_summary]).to_csv(
    os.path.join(OUTPUT_DIR, "eval_heldout.csv"), index=False)

# CSV sens
sens_summary = {
    "acc_global_sens": float(g_sens),
    "std_sens": float(np.std(acc_sens)),
    "acc_global_baseline_sens": float(g_orig_sens),
    "std_baseline_sens": float(np.std(acc_orig_sens)),
    **{f"sens_{CIFAR100_CLASSES[c]}": float(acc_sens[c]) for c in range(100)},
    **{f"baseline_sens_{CIFAR100_CLASSES[c]}": float(acc_orig_sens[c]) for c in range(100)},
}
pd.DataFrame([sens_summary]).to_csv(
    os.path.join(OUTPUT_DIR, "sens_summary.csv"), index=False)

print(f"\n  Sauvegardé dans {OUTPUT_DIR}/eval_heldout.csv et sens_summary.csv")
print(f"{'='*70}")
