"""
run_spectral_density_resnet50.py
---------------------------------
Calcule la densité spectrale complète du Hessien pour ResNet-50/CIFAR-10
via SLQ (m=90, k=10) et sauvegarde le spectre + figure.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

# ── 1. Données ──────────────────────────────────────────────────────────
print("[1] Chargement de CIFAR-10 ...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_train, y_test = y_train.flatten(), y_test.flatten()
x_train = x_train.astype("float32") / 255.0

rng = np.random.default_rng(0)
hvp_idx = rng.choice(len(x_train), 128, replace=False)
x_hvp, y_hvp = x_train[hvp_idx], y_train[hvp_idx]

# ── 2. Modèle ──────────────────────────────────────────────────────────
print("[2] Chargement du modèle ...")
model = tf.keras.models.load_model("resnet50_cifar10.keras")
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
n_params = sum(np.prod(v.shape) for v in model.trainable_variables)
print(f"    Paramètres : {n_params:,}")

# ── 3. HVP ─────────────────────────────────────────────────────────────
from spectral_tools import HessianVectorProduct, StochasticLanczosQuadrature

print("[3] Configuration HVP ...")
hvp = HessianVectorProduct(
    model=model, loss_fn=loss_fn,
    data_x=x_hvp, data_y=y_hvp, batch_size=None,
)

# ── 4. Top eigenvalues (rapide, m=20) ──────────────────────────────────
print("[4] Top eigenvalues (Lanczos m=20) ...")
t0 = time.time()
slq = StochasticLanczosQuadrature(
    hvp=hvp, n_params=hvp.n_params, m=20, k=1, sigma2=1e-5,
)
ritz_vals, ritz_vecs = slq.estimate_top_eigenvalues(m_lanczos=20, verbose=True)
print(f"    Top-10 eigenvalues : {ritz_vals[:10].round(1).tolist()}")
print(f"    Temps : {time.time()-t0:.1f}s")

# ── 5. Densité spectrale complète (SLQ m=90, k=10) ────────────────────
print("\n[5] Densité spectrale complète (SLQ m=90, k=10) ...")
print("    ⚠ Ceci prend ~30 min sur CPU")
t0 = time.time()
slq_full = StochasticLanczosQuadrature(
    hvp=hvp, n_params=hvp.n_params, m=90, k=10, sigma2=1e-5,
)

# Grille de valeurs pour la densité
t_bulk = np.linspace(-1, 5, 2000)       # zoom sur le bulk
t_full = np.linspace(-1, 900, 5000)     # spectre complet

density_bulk = slq_full.estimate_density(t_bulk, verbose=True)
density_full = slq_full.estimate_density(t_full, verbose=False)
elapsed = time.time() - t0
print(f"    Temps total : {elapsed/60:.1f} min")

# ── 6. Sauvegarde des données ─────────────────────────────────────────
os.makedirs("results/spectral/density_cifar10", exist_ok=True)

np.savez("results/spectral/density_cifar10/spectral_data.npz",
         t_bulk=t_bulk, density_bulk=density_bulk,
         t_full=t_full, density_full=density_full,
         ritz_vals=ritz_vals)

# ── 7. Figure ─────────────────────────────────────────────────────────
print("[6] Création de la figure ...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                         gridspec_kw={'width_ratios': [2, 1]})
fig.suptitle("Hessian Spectral Density — ResNet-50 / CIFAR-10", fontsize=13)

# Panel gauche : spectre complet (densité + spikes en barres)
ax = axes[0]
ax.plot(t_full, density_full, color="steelblue", lw=1.2, label="SLQ density")
# Marquer les spikes comme des lignes verticales
for i, lam in enumerate(ritz_vals[:9]):
    label = f"spike {i+1}" if i == 0 else None
    ax.axvline(lam, color="crimson", lw=1.5, alpha=0.7, ls="--", label=label)
ax.set_xlabel(r"$\lambda$")
ax.set_ylabel(r"Spectral density $\phi(\lambda)$")
ax.set_title("Full spectrum")
ax.legend(fontsize=8)
ax.set_xlim(-10, max(ritz_vals[0] * 1.1, 900))
ax.grid(alpha=0.3)

# Panel droit : zoom sur le bulk
ax = axes[1]
ax.fill_between(t_bulk, density_bulk, color="steelblue", alpha=0.4)
ax.plot(t_bulk, density_bulk, color="steelblue", lw=1.5, label="Bulk density")
ax.set_xlabel(r"$\lambda$")
ax.set_ylabel(r"$\phi(\lambda)$")
ax.set_title("Bulk zoom")
ax.set_xlim(-0.5, 3)
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

plt.tight_layout()
fig.savefig("results/spectral/density_cifar10/spectral_density.png", dpi=200)
fig.savefig("results/spectral/density_cifar10/spectral_density.pdf")
plt.close()
print(f"  Figure sauvegardée : results/spectral/density_cifar10/spectral_density.{{png,pdf}}")

# ── 8. Stats ──────────────────────────────────────────────────────────
# Bulk median
bulk_mask = (t_full > -0.5) & (t_full < 5) & (density_full > 0.01)
if bulk_mask.sum() > 0:
    bulk_center = np.average(t_full[bulk_mask], weights=density_full[bulk_mask])
    print(f"\n  Statistiques :")
    print(f"    λ_max = {ritz_vals[0]:.1f}")
    print(f"    Centre du bulk ≈ {bulk_center:.4f}")
    print(f"    Anisotropie ζ = λ_max / bulk_center ≈ {ritz_vals[0]/max(bulk_center, 1e-6):.0f}")
    print(f"    Nombre de spikes (λ > 10) : {(ritz_vals > 10).sum()}")
