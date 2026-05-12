"""
linearization_diagnostic.py
---------------------------
Balayage dédié pour séparer les deux sources d'erreur dans le diagnostic
de linéarisation : bruit d'estimation de S vs non-linéarité.

Protocole :
  1. Calculer eigenvectors et S une seule fois
  2. Pour chaque alpha_max dans une grille, optimiser α, appliquer le choc,
     mesurer predicted vs observed, puis rollback
  3. Régression lin_err ~ a + b·‖α‖² pour estimer le plancher (bruit S)
     et la composante quadratique (non-linéarité)
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
import pandas as pd
from scipy import stats

from spike_optimizer import (
    per_class_accuracy, save_weights, restore_weights,
    apply_perturbation, compute_eigenvectors, compute_sensitivity,
    optimize_alpha, CONFIG, CIFAR10_CLASSES,
)

# ── Config du diagnostic ─────────────────────────────────────────────────
ALPHA_GRID = sorted(set(
    [round(x, 4) for x in np.linspace(0.0005, 0.020, 30)]
    + [0.022, 0.025, 0.028, 0.030, 0.035, 0.040, 0.045, 0.050]
))
OUTPUT_DIR = "results/spectral/linearization"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Données ──────────────────────────────────────────────────────────────
print("[1] Chargement CIFAR-10 ...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_train, y_test = y_train.flatten(), y_test.flatten()
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0

rng = np.random.default_rng(CONFIG["seed"])
idx_test = rng.permutation(len(x_test))
x_sens, y_sens = x_test[idx_test[:5000]], y_test[idx_test[:5000]]

hvp_idx = rng.choice(len(x_train), CONFIG["n_hvp_samples"], replace=False)
x_hvp, y_hvp = x_train[hvp_idx], y_train[hvp_idx]

# ── Modèle ───────────────────────────────────────────────────────────────
print("[2] Chargement du modèle ...")
model   = tf.keras.models.load_model(CONFIG["model_path"], compile=False)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

# ── Eigenvectors (une seule fois) ────────────────────────────────────────
print("[3] Lanczos ...")
ritz_vals, ritz_vecs = compute_eigenvectors(
    model, loss_fn, x_hvp, y_hvp,
    CONFIG["lanczos_m"], CONFIG["n_spikes"],
)
n_sp = min(CONFIG["n_spikes"], ritz_vecs.shape[1])
print(f"    λ top-3 : {ritz_vals[:3].round(1).tolist()}")

# ── Sensibilité (une seule fois) ─────────────────────────────────────────
print("[4] Matrice de sensibilité S ...")
S = compute_sensitivity(
    model, ritz_vecs, n_sp,
    CONFIG["eps_probe"], x_sens, y_sens,
)

# ── Balayage ─────────────────────────────────────────────────────────────
print(f"\n[5] Balayage de {len(ALPHA_GRID)} niveaux d'alpha_max ...")
acc_current = per_class_accuracy(model, x_sens, y_sens)
weights_ref = save_weights(model)

results = []
for alpha_max in ALPHA_GRID:
    # Optimiser α sous contrainte ‖α‖ ≤ alpha_max
    alpha = optimize_alpha(S, acc_current, alpha_max, ritz_vals[:n_sp])
    alpha_norm = float(np.linalg.norm(alpha))

    # Prédiction linéaire
    predicted_delta = S.T @ alpha

    # Appliquer le choc
    delta = np.zeros(ritz_vecs.shape[0], dtype=np.float64)
    for s in range(n_sp):
        delta += alpha[s] * ritz_vecs[:, s]
    apply_perturbation(model, delta.astype(np.float32))

    # Mesurer
    acc_after = per_class_accuracy(model, x_sens, y_sens)
    observed_delta = acc_after - acc_current

    # Écart de linéarisation
    err_abs = float(np.linalg.norm(predicted_delta - observed_delta))
    pred_norm = float(np.linalg.norm(predicted_delta))
    err_rel = err_abs / pred_norm if pred_norm > 1e-12 else 0.0

    print(f"  alpha_max={alpha_max:.3f}  ‖α‖={alpha_norm:.5f}  "
          f"lin_err_rel={err_rel:.4f}  ‖pred‖={pred_norm:.5f}  ‖obs‖={float(np.linalg.norm(observed_delta)):.5f}")

    results.append({
        "alpha_max": alpha_max,
        "alpha_norm": alpha_norm,
        "lin_err_rel": err_rel,
        "lin_err_abs": err_abs,
        "pred_norm": pred_norm,
        "obs_norm": float(np.linalg.norm(observed_delta)),
    })

    # Rollback
    restore_weights(model, weights_ref)

# ── Sauvegarde ───────────────────────────────────────────────────────────
df = pd.DataFrame(results)
csv_path = os.path.join(OUTPUT_DIR, "linearization_sweep.csv")
df.to_csv(csv_path, index=False)
print(f"\n  CSV : {csv_path}")

# ── Régression ───────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  RÉGRESSION  lin_err_rel ~ a + b·‖α‖²")
print(f"{'='*60}")
x = df["alpha_norm"].values
y = df["lin_err_rel"].values
x2 = x**2

slope, intercept, r, p, se = stats.linregress(x2, y)
print(f"  intercept (a) = {intercept:.4f}   (plancher = bruit estimation S)")
print(f"  slope (b)     = {slope:.1f}")
print(f"  R²            = {r**2:.4f}")
print(f"  p-value       = {p:.6f}")

rho, p_sp = stats.spearmanr(x, y)
print(f"  Spearman(‖α‖, lin_err) = {rho:.3f}  p={p_sp:.4f}")

# Régression sur lin_err_abs aussi (pas de normalisation)
slope2, intercept2, r2, p2, _ = stats.linregress(x2, df["lin_err_abs"].values)
print(f"\n  --- lin_err_abs ~ a + b·‖α‖² ---")
print(f"  intercept (a) = {intercept2:.5f}")
print(f"  slope (b)     = {slope2:.1f}")
print(f"  R²            = {r2**2:.4f}")
print(f"  p-value       = {p2:.6f}")
