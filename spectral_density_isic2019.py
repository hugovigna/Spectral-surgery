"""
spectral_density_isic2019.py
-----------------------------
Densité spectrale du Hessien pour resnet50_isic2019.keras (baseline FL+SS).
Version allégée : m=30, k=5 (vs m=90, k=10 pour la version complète).
Affiche aussi le gap bulk/spike pour calibrer lanczos_m du spike optimizer.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

# ── Données ──────────────────────────────────────────────────────────────
print("[1] Chargement des données ...")
rng = np.random.default_rng(0)

train_data = np.load("data/isic2019_cache/train.npz")
x_train    = train_data["imgs"].astype(np.float32)
y_train    = train_data["labels"].astype(np.int32)

hvp_idx = rng.choice(len(x_train), 128, replace=False)
x_hvp   = x_train[hvp_idx]
y_hvp   = y_train[hvp_idx]
print(f"    HVP samples : {len(x_hvp)}")

# ── Modèle ────────────────────────────────────────────────────────────────
print("[2] Chargement du modèle ...")
model   = tf.keras.models.load_model("resnet50_isic2019.keras")
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
n_params = sum(np.prod(v.shape) for v in model.trainable_variables)
print(f"    Paramètres : {n_params:,}")

# ── HVP ───────────────────────────────────────────────────────────────────
from spectral_tools import HessianVectorProduct, StochasticLanczosQuadrature

print("[3] Configuration HVP ...")
hvp = HessianVectorProduct(
    model=model, loss_fn=loss_fn,
    data_x=x_hvp, data_y=y_hvp, batch_size=None,
)

# ── Top eigenvalues (m=15) ────────────────────────────────────────────────
print("\n[4] Top eigenvalues (Lanczos m=15) ...")
t0 = time.time()
slq_top = StochasticLanczosQuadrature(
    hvp=hvp, n_params=hvp.n_params, m=20, k=1, sigma2=1e-5,
)
ritz_vals, _ = slq_top.estimate_top_eigenvalues(m_lanczos=15, verbose=True)
print(f"    Top-15 eigenvalues : {ritz_vals[:15].round(2).tolist()}")
print(f"    Temps : {time.time()-t0:.1f}s")

# Gap ratio spike/bulk
if len(ritz_vals) >= 8:
    gap = ritz_vals[6] / ritz_vals[7] if ritz_vals[7] > 0 else float('inf')
    print(f"\n    Gap λ₇/λ₈ = {ritz_vals[6]:.2f} / {ritz_vals[7]:.2f} = {gap:.1f}x")
    print(f"    → lanczos_m={'10 suffit' if gap > 3 else '20 recommandé'} (gap {'> 3' if gap > 3 else '< 3'})")

# ── Densité spectrale (m=30, k=5) ────────────────────────────────────────
print("\n[5] Densité spectrale (SLQ m=30, k=5) ...")
print("    ⚠ ~20-40 min sur Apple Silicon avec Metal GPU")
t0 = time.time()
slq_full = StochasticLanczosQuadrature(
    hvp=hvp, n_params=hvp.n_params, m=30, k=5, sigma2=1e-5,
)
t_bulk = np.linspace(-1, 10, 2000)
t_full = np.linspace(-1, ritz_vals[0] * 1.2, 3000)
density_bulk = slq_full.estimate_density(t_bulk, verbose=True)
density_full = slq_full.estimate_density(t_full, verbose=False)
print(f"    Temps densité : {time.time()-t0:.1f}s")

# ── Sauvegarde ────────────────────────────────────────────────────────────
out_dir = "results/isic2019/spectral_density"
os.makedirs(out_dir, exist_ok=True)
np.savez(f"{out_dir}/spectral_data.npz",
         t_bulk=t_bulk, density_bulk=density_bulk,
         t_full=t_full, density_full=density_full,
         ritz_vals=ritz_vals)

# ── Plot ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Spectral density — ResNet50/ISIC2019 (SLQ m=30, k=5)", fontsize=12)

ax = axes[0]
ax.plot(t_full, density_full, color="steelblue", lw=1.2, label="SLQ density")
for i, v in enumerate(ritz_vals[:7]):
    ax.axvline(v, color="crimson", lw=1.2, alpha=0.7, ls="--",
               label="spikes" if i == 0 else None)
ax.set_xlabel(r"$\lambda$")
ax.set_ylabel(r"Spectral density $\phi(\lambda)$")
ax.set_title("Full spectrum (bulk + spikes)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

ax = axes[1]
ax.fill_between(t_bulk, density_bulk, color="steelblue", alpha=0.4)
ax.plot(t_bulk, density_bulk, color="steelblue", lw=1.5)
ax.set_xlabel(r"$\lambda$")
ax.set_ylabel(r"$\phi(\lambda)$")
ax.set_title("Zoom bulk")
ax.set_xlim(-0.5, 5)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{out_dir}/spectral_density.png", dpi=150)
plt.savefig(f"{out_dir}/spectral_density.pdf")
plt.close()
print(f"\n    Figure : {out_dir}/spectral_density.png")
print(f"    Top-15 : {ritz_vals[:15].round(2).tolist()}")
