"""
test_hvp_batch_ablation.py
---------------------------
Teste la robustesse des eigenvectors HVP selon le nombre d'images (64, 128, 256, 512).

Sous-ensembles emboîtés : idx_512 ⊃ idx_256 ⊃ idx_128 ⊃ idx_64
Seed fixé pour Lanczos (vecteur de départ identique).
Matching optimal des eigenvectors (max |cosine| sur toutes les paires).

Métriques :
  - Valeurs propres top-9 (doivent être stables)
  - Cosine similarity (matching optimal) entre eigenvectors et référence n=512
  - Temps de calcul par HVP
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
import pandas as pd
import time
from scipy.linalg import subspace_angles

# ── Données ──────────────────────────────────────────────────────────────
print("[1] Chargement de CIFAR-10 ...")
(x_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()
y_train = y_train.flatten()
x_train = x_train.astype("float32") / 255.0

print("[2] Chargement du modèle ...")
model   = tf.keras.models.load_model("resnet50_cifar10.keras")
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

from spectral_tools import HessianVectorProduct, StochasticLanczosQuadrature

# ── Sous-ensembles emboîtés ──────────────────────────────────────────────
BATCH_SIZES = [64, 128, 256, 512]
rng = np.random.default_rng(42)

# On tire d'abord 512 indices, puis on prend les premiers n
idx_512 = rng.choice(len(x_train), 512, replace=False)

results = {}

for n in BATCH_SIZES:
    # Sous-ensemble emboîté : les n premiers indices de idx_512
    idx   = idx_512[:n]
    x_hvp = x_train[idx]
    y_hvp = y_train[idx]

    hvp = HessianVectorProduct(
        model=model, loss_fn=loss_fn,
        data_x=x_hvp, data_y=y_hvp, batch_size=None,
    )

    # Timing d'un seul HVP
    v = np.random.randn(hvp.n_params).astype(np.float32)
    t0 = time.time()
    _ = hvp.compute(v)
    hvp_time = time.time() - t0

    # Lanczos pour les top-9 eigenvectors — seed fixé pour le vecteur de départ
    slq = StochasticLanczosQuadrature(
        hvp=hvp, n_params=hvp.n_params, m=20, k=1, sigma2=1e-5,
    )

    # Fixer le seed numpy AVANT l'appel (estimate_top_eigenvalues utilise np.random.randn)
    np.random.seed(0)
    t0 = time.time()
    ritz_vals, ritz_vecs = slq.estimate_top_eigenvalues(m_lanczos=20, verbose=False)
    lanczos_time = time.time() - t0

    # Normaliser les vecteurs de Ritz (sécurité)
    for i in range(ritz_vecs.shape[1]):
        norm = np.linalg.norm(ritz_vecs[:, i])
        if norm > 1e-12:
            ritz_vecs[:, i] /= norm

    results[n] = {
        "vals": ritz_vals[:9],
        "vecs": ritz_vecs[:, :9],
        "hvp_time": hvp_time,
        "lanczos_time": lanczos_time,
    }
    print(f"  n={n:4d}  HVP={hvp_time:.1f}s  Lanczos={lanczos_time:.0f}s  "
          f"λ top-3: {ritz_vals[:3].round(1).tolist()}")

# ── Comparaison : matching optimal vs référence n=512 ─────────────────────
print(f"\n{'='*65}")
print("  COMPARAISON DES EIGENVECTORS (référence : n=512)")
print(f"{'='*65}")

ref_vecs = results[512]["vecs"]   # n_params × 9
ref_vals = results[512]["vals"]

rows = []
for n in BATCH_SIZES:
    vecs = results[n]["vecs"]

    # Matrice de cosine similarity : |cos(v_i, ref_j)| pour toutes les paires
    # Shape : (9, 9)
    cos_matrix = np.abs(vecs.T @ ref_vecs)

    # Matching greedy : pour chaque eigenvector du test, trouver le meilleur match
    # dans la référence (sans réutiliser un match déjà pris)
    used_ref = set()
    matched_sims = []
    for i in range(9):
        best_sim = -1
        best_j = -1
        for j in range(9):
            if j not in used_ref and cos_matrix[i, j] > best_sim:
                best_sim = cos_matrix[i, j]
                best_j = j
        used_ref.add(best_j)
        matched_sims.append(best_sim)

    mean_sim = np.mean(matched_sims)
    min_sim  = np.min(matched_sims)

    # Aussi afficher les cosines diagonaux (index-par-index) pour comparaison
    diag_sims = [cos_matrix[i, i] for i in range(9)]
    mean_diag = np.mean(diag_sims)

    print(f"  n={n:4d}  matched: mean={mean_sim:.4f} min={min_sim:.4f}  "
          f"diagonal: mean={mean_diag:.4f}  "
          f"HVP={results[n]['hvp_time']:.1f}s")
    rows.append({
        "n_hvp":        n,
        "cosine_matched_mean": mean_sim,
        "cosine_matched_min":  min_sim,
        "cosine_diag_mean":    mean_diag,
        "hvp_time_s":   results[n]["hvp_time"],
        "lanczos_time_s": results[n]["lanczos_time"],
        **{f"lambda_{i+1}": float(results[n]["vals"][i]) for i in range(9)},
    })

# ── Comparaison des valeurs propres ──────────────────────────────────────
print(f"\n  Valeurs propres top-9 par batch size :")
header = f"  {'n':>6s}" + "".join(f"  λ{i+1:>5d}" for i in range(9))
print(header)
for n in BATCH_SIZES:
    line = f"  {n:>6d}" + "".join(f"  {v:>6.1f}" for v in results[n]["vals"])
    print(line)

# ── Matrice de cosine similarity détaillée pour chaque n ──────────────────
print(f"\n  Matrices de cosine similarity |cos(v_i, ref_j)| :")
for n in [64, 128, 256]:
    vecs = results[n]["vecs"]
    cos_matrix = np.abs(vecs.T @ ref_vecs)
    print(f"\n  n={n} vs n=512 :")
    header = "        " + "".join(f" ref_{j+1:d}" for j in range(9))
    print(header)
    for i in range(9):
        line = f"  v_{i+1:d}  " + "".join(f"  {cos_matrix[i,j]:.3f}" for j in range(9))
        print(line)

print(f"\n{'='*65}")

# ── Angles principaux entre sous-espaces (top-k) ─────────────────────────
print("\n  ANGLES PRINCIPAUX entre sous-espace top-k et référence n=512")
print(f"  (en degrés — 0° = sous-espaces identiques)")
print(f"  {'n':>6s}  {'k':>4s}  {'max_angle':>10s}  {'mean_angle':>11s}  {'angles':}")
for n in [64, 128, 256]:
    V = results[n]["vecs"]      # n_params × 9
    for k in [3, 6, 9]:
        angles_rad = subspace_angles(V[:, :k], ref_vecs[:, :k])
        angles_deg = np.degrees(angles_rad)
        print(f"  {n:>6d}  k={k}  max={angles_deg.max():>7.1f}°  "
              f"mean={angles_deg.mean():>7.1f}°  "
              f"[{', '.join(f'{a:.1f}' for a in angles_deg)}]")

# Ajouter les angles principaux à rows
for row in rows:
    n = row["n_hvp"]
    if n == 512:
        for k in [3, 6, 9]:
            row[f"subspace_max_angle_k{k}"] = 0.0
            row[f"subspace_mean_angle_k{k}"] = 0.0
    else:
        V = results[n]["vecs"]
        for k in [3, 6, 9]:
            angles_rad = subspace_angles(V[:, :k], ref_vecs[:, :k])
            angles_deg = np.degrees(angles_rad)
            row[f"subspace_max_angle_k{k}"] = float(angles_deg.max())
            row[f"subspace_mean_angle_k{k}"] = float(angles_deg.mean())

print(f"\n{'='*65}")

# ── Sauvegarde ───────────────────────────────────────────────────────────
os.makedirs("results/ablation_hvp", exist_ok=True)
pd.DataFrame(rows).to_csv("results/ablation_hvp/summary.csv", index=False)
print(f"\n  Sauvegardé dans results/ablation_hvp/summary.csv")
