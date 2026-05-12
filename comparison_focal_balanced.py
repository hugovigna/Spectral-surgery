"""
comparison_focal_balanced.py
------------------------------
Comparaison de Hessian Surgery avec les baselines de rééquilibrage :

  4 méthodes (Surgery seul déjà documenté séparément) :
    1. Focal Loss FT (γ=2, 3 epochs)
    2. Class-Balanced Loss FT (3 epochs)
    3. Focal Loss FT → Hessian Surgery
    4. Class-Balanced Loss FT → Hessian Surgery

Usage :
    python3 comparison_focal_balanced.py
"""

import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
import pandas as pd
import time

sys.path.insert(0, os.path.dirname(__file__))
from spike_optimizer import (
    SpikeOptimizer, per_class_accuracy, save_weights, restore_weights,
    CIFAR10_CLASSES, CONFIG
)

# ════════════════════════════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════════════════════════════
FT_EPOCHS   = 3
FT_LR       = 1e-4
FT_BATCH    = 64
MODEL_PATH  = "resnet50_cifar10.keras"
OUTPUT_DIR  = "results/cifar10/comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ════════════════════════════════════════════════════════════════════════════
# Données
# ════════════════════════════════════════════════════════════════════════════
print("[1] Chargement de CIFAR-10 ...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_train, y_test = y_train.flatten(), y_test.flatten()
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0

# ── Split test set : 5000 sensibilité + 5000 évaluation held-out ──────
rng = np.random.default_rng(0)
idx_test = rng.permutation(len(x_test))
x_sens, y_sens = x_test[idx_test[:5000]], y_test[idx_test[:5000]]
x_eval, y_eval = x_test[idx_test[5000:]], y_test[idx_test[5000:]]
print(f"    Test split : {len(x_sens)} sensibilité + {len(x_eval)} évaluation held-out")

hvp_idx = rng.choice(len(x_train), 128, replace=False)
x_hvp, y_hvp = x_train[hvp_idx], y_train[hvp_idx]

# ════════════════════════════════════════════════════════════════════════════
# Losses
# ════════════════════════════════════════════════════════════════════════════

class FocalLoss(tf.keras.losses.Loss):
    """Focal Loss : -α(1-p)^γ log(p), réduit le poids des exemples faciles."""
    def __init__(self, gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma

    def call(self, y_true, y_pred):
        y_true = tf.cast(tf.squeeze(y_true), tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        p_true = tf.gather(y_pred, y_true, batch_dims=1)
        focal_weight = (1.0 - p_true) ** self.gamma
        return -tf.reduce_mean(focal_weight * tf.math.log(p_true))


class ClassBalancedLoss(tf.keras.losses.Loss):
    """Cross-entropy pondérée par l'inverse de la fréquence par classe."""
    def __init__(self, class_counts, beta=0.9999, **kwargs):
        super().__init__(**kwargs)
        # Effective number of samples (Cui et al. 2019)
        effective_num = 1.0 - np.power(beta, class_counts)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * len(class_counts)
        self.class_weights = tf.constant(weights, dtype=tf.float32)

    def call(self, y_true, y_pred):
        y_true = tf.cast(tf.squeeze(y_true), tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        p_true = tf.gather(y_pred, y_true, batch_dims=1)
        w = tf.gather(self.class_weights, y_true)
        return -tf.reduce_mean(w * tf.math.log(p_true))


# ════════════════════════════════════════════════════════════════════════════
# Fonctions utilitaires
# ════════════════════════════════════════════════════════════════════════════

def evaluate_model(model, x_test, y_test):
    """Retourne dict avec acc_global, std, et acc par classe."""
    acc_pc = per_class_accuracy(model, x_test, y_test)
    acc_global = model.evaluate(x_test, y_test, verbose=0, batch_size=256)[1]
    return {
        "acc_global": float(acc_global),
        "std": float(np.std(acc_pc)),
        **{CIFAR10_CLASSES[c]: float(acc_pc[c]) for c in range(10)},
    }

def finetune(model, loss_fn, x_train, y_train, x_val, y_val,
             epochs=FT_EPOCHS, lr=FT_LR, batch_size=FT_BATCH, label=""):
    """Fine-tune le modèle avec la loss donnée. Retourne le temps écoulé."""
    print(f"\n  [{label}] Fine-tuning {epochs} epochs, lr={lr} ...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=loss_fn,
        metrics=["accuracy"],
    )
    t0 = time.time()
    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )
    elapsed = time.time() - t0
    print(f"  [{label}] Terminé en {elapsed:.0f}s")
    return elapsed

def run_hessian_surgery(model, loss_fn, x_sens, y_sens, x_eval, y_eval,
                         x_hvp, y_hvp, label=""):
    """Lance Hessian Surgery et retourne (log, elapsed)."""
    print(f"\n  [{label}] Hessian Surgery (10 itérations) ...")
    cfg = CONFIG.copy()
    cfg["output_dir"] = os.path.join(OUTPUT_DIR, label.replace(" ", "_"))
    cfg["save_model"] = False
    t0 = time.time()
    optimizer = SpikeOptimizer(
        model=model, loss_fn=loss_fn,
        x_sens=x_sens, y_sens=y_sens,
        x_eval=x_eval, y_eval=y_eval,
        x_hvp=x_hvp, y_hvp=y_hvp,
        cfg=cfg,
    )
    log = optimizer.run()
    elapsed = time.time() - t0
    return log, elapsed


# ════════════════════════════════════════════════════════════════════════════
# Baseline
# ════════════════════════════════════════════════════════════════════════════
print("\n[2] Baseline ...")
model_base = tf.keras.models.load_model(MODEL_PATH)
baseline_sens = evaluate_model(model_base, x_sens, y_sens)
baseline_eval = evaluate_model(model_base, x_eval, y_eval)
print(f"    Sens  : Global={baseline_sens['acc_global']*100:.1f}%  std={baseline_sens['std']*100:.2f}%")
print(f"    Eval  : Global={baseline_eval['acc_global']*100:.1f}%  std={baseline_eval['std']*100:.2f}%")
baseline = baseline_sens  # pour la compatibilité
del model_base

results = []

# ════════════════════════════════════════════════════════════════════════════
# Fine-tuning direct
# ════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  FINE-TUNING DIRECT (3 epochs)")
print(f"{'='*70}")

# ── A1. Focal Loss FT ────────────────────────────────────────────────────
print("\n[A1] Focal Loss FT ...")
model_focal = tf.keras.models.load_model(MODEL_PATH)
focal_loss = FocalLoss(gamma=2.0)
elapsed_focal = finetune(
    model_focal, focal_loss, x_train, y_train, x_sens, y_sens,
    label="Focal Loss")
res_focal_sens = evaluate_model(model_focal, x_sens, y_sens)
res_focal_eval = evaluate_model(model_focal, x_eval, y_eval)
res_focal_sens["method"] = "Focal Loss FT"
res_focal_sens["time_s"] = elapsed_focal
res_focal_eval["method"] = "Focal Loss FT"
res_focal_eval["time_s"] = elapsed_focal
results.append({"sens": res_focal_sens, "eval": res_focal_eval})
print(f"    Sens : Global={res_focal_sens['acc_global']*100:.1f}%  std={res_focal_sens['std']*100:.2f}%")
print(f"    Eval : Global={res_focal_eval['acc_global']*100:.1f}%  std={res_focal_eval['std']*100:.2f}%")

# ── A2. Class-Balanced Loss FT ──────────────────────────────────────────
print("\n[A2] Class-Balanced Loss FT ...")
model_cb = tf.keras.models.load_model(MODEL_PATH)
class_counts = np.bincount(y_train, minlength=10).astype(np.float64)
cb_loss = ClassBalancedLoss(class_counts)
elapsed_cb = finetune(
    model_cb, cb_loss, x_train, y_train, x_sens, y_sens,
    label="Class-Balanced")
res_cb_sens = evaluate_model(model_cb, x_sens, y_sens)
res_cb_eval = evaluate_model(model_cb, x_eval, y_eval)
res_cb_sens["method"] = "Class-Balanced FT"
res_cb_sens["time_s"] = elapsed_cb
res_cb_eval["method"] = "Class-Balanced FT"
res_cb_eval["time_s"] = elapsed_cb
results.append({"sens": res_cb_sens, "eval": res_cb_eval})
print(f"    Sens : Global={res_cb_sens['acc_global']*100:.1f}%  std={res_cb_sens['std']*100:.2f}%")
print(f"    Eval : Global={res_cb_eval['acc_global']*100:.1f}%  std={res_cb_eval['std']*100:.2f}%")

# ════════════════════════════════════════════════════════════════════════════
# Résumé final — Table 11 du papier
# ════════════════════════════════════════════════════════════════════════════

# Charger les résultats Hessian Surgery depuis le run précédent
ss_sens_csv = pd.read_csv("results/cifar10/ss/summary.csv")
ss_eval_csv = pd.read_csv("results/cifar10/ss/eval_heldout.csv")

print(f"\n{'='*100}")
print("  TABLE 11 — Comparaison des méthodes de rééquilibrage")
print(f"{'='*100}")

# ── Tableau VALIDATION SET (sensibilité, 5000 images) ──
print(f"\n  --- Validation set (5000 images, utilisé pour le pilotage) ---")
print(f"  {'Méthode':<22s}  {'Global':>7s}  {'σ':>7s}  {'Δσ (pp)':>8s}  {'Post-hoc?':>9s}")
print(f"  {'-'*60}")
print(f"  {'Baseline':<22s}  {baseline_sens['acc_global']*100:>6.1f}%  "
      f"{baseline_sens['std']*100:>6.2f}%  {'—':>8s}  {'—':>9s}")
for r in results:
    s = r["sens"]
    d_s = (s['std'] - baseline_sens['std']) * 100
    post_hoc = "Non"
    print(f"  {s['method']:<22s}  {s['acc_global']*100:>6.1f}%  "
          f"{s['std']*100:>6.2f}%  {d_s:>+7.2f}  {post_hoc:>9s}")
# Hessian Surgery (depuis summary.csv)
ss_std_sens = float(ss_sens_csv["std_f"].iloc[0]) * 100
ss_global_sens = float(ss_sens_csv["acc_global_f"].iloc[0]) * 100
d_ss = ss_std_sens - baseline_sens['std'] * 100
print(f"  {'Hessian Surgery':<22s}  {ss_global_sens:>6.1f}%  "
      f"{ss_std_sens:>6.2f}%  {d_ss:>+7.2f}  {'Oui':>9s}")

# ── Tableau TEST SET (held-out, 5000 images) ──
print(f"\n  --- Test set held-out (5000 images, non contaminé) ---")
print(f"  {'Méthode':<22s}  {'Global':>7s}  {'σ':>7s}  {'Δσ (pp)':>8s}  {'Post-hoc?':>9s}")
print(f"  {'-'*60}")
print(f"  {'Baseline':<22s}  {baseline_eval['acc_global']*100:>6.1f}%  "
      f"{baseline_eval['std']*100:>6.2f}%  {'—':>8s}  {'—':>9s}")
for r in results:
    e = r["eval"]
    d_e = (e['std'] - baseline_eval['std']) * 100
    post_hoc = "Non"
    print(f"  {e['method']:<22s}  {e['acc_global']*100:>6.1f}%  "
          f"{e['std']*100:>6.2f}%  {d_e:>+7.2f}  {post_hoc:>9s}")
# Hessian Surgery (depuis eval_heldout.csv)
ss_std_eval = float(ss_eval_csv["std_eval"].iloc[0]) * 100
ss_global_eval = float(ss_eval_csv["acc_global_eval"].iloc[0]) * 100
d_ss_eval = ss_std_eval - baseline_eval['std'] * 100
print(f"  {'Hessian Surgery':<22s}  {ss_global_eval:>6.1f}%  "
      f"{ss_std_eval:>6.2f}%  {d_ss_eval:>+7.2f}  {'Oui':>9s}")

print(f"\n{'='*100}")

# Sauvegarde CSV
rows = []
rows.append({"method": "Baseline", "global_sens": baseline_sens["acc_global"],
             "std_sens": baseline_sens["std"], "global_eval": baseline_eval["acc_global"],
             "std_eval": baseline_eval["std"], "post_hoc": True})
for r in results:
    rows.append({"method": r["sens"]["method"],
                 "global_sens": r["sens"]["acc_global"], "std_sens": r["sens"]["std"],
                 "global_eval": r["eval"]["acc_global"], "std_eval": r["eval"]["std"],
                 "post_hoc": False})
rows.append({"method": "Hessian Surgery",
             "global_sens": ss_global_sens/100, "std_sens": ss_std_sens/100,
             "global_eval": ss_global_eval/100, "std_eval": ss_std_eval/100,
             "post_hoc": True})
pd.DataFrame(rows).to_csv(
    os.path.join(OUTPUT_DIR, "comparison_table11.csv"), index=False)
print(f"\n  Sauvegardé dans {OUTPUT_DIR}/comparison_table11.csv")
