"""
plot_spectral_resnet50.py
--------------------------
Régénère la figure spectrale ResNet-50 / CIFAR-10
depuis les données sauvegardées, avec lignes verticales
hachurées sur chaque spike.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Chargement des données ──────────────────────────────────────────────────
d = np.load("results/spectral/density_cifar10/spectral_data.npz")
t_bulk       = d["t_bulk"]
density_bulk = d["density_bulk"]
t_full       = d["t_full"]
density_full = d["density_full"]
ritz_vals    = d["ritz_vals"]

# Spikes = les C-1 = 9 plus grandes valeurs propres (théorie Papyan 2020)
n_spikes = 9
sorted_vals = np.sort(ritz_vals)[::-1]
spikes = sorted_vals[:n_spikes]
print(f"Spikes retenus ({len(spikes)}) : {spikes.round(1).tolist()}")

# ── Figure ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                         gridspec_kw={"width_ratios": [2, 1]})
fig.suptitle("Hessian Spectral Density — ResNet-50 / CIFAR-10", fontsize=13)

# Palette de couleurs pour les spikes
cmap = plt.cm.get_cmap("tab10", len(spikes))

# ── Panel gauche : spectre complet ─────────────────────────────────────────
ax = axes[0]
ax.fill_between(t_full, density_full, alpha=0.15, color="steelblue")
ax.plot(t_full, density_full, color="steelblue", lw=1.2, label="SLQ density")

for i, lam in enumerate(spikes):
    label = f"spike {i+1}  (λ={lam:.0f})"
    ax.axvline(lam, color=cmap(i), lw=1.4, alpha=0.85,
               ls="--", dashes=(5, 4), label=label)

ax.set_xlabel(r"$\lambda$", fontsize=11)
ax.set_ylabel(r"Spectral density $\phi(\lambda)$", fontsize=11)
ax.set_title("Full spectrum", fontsize=11)
ax.set_xlim(-20, spikes[0] * 1.08)
ax.set_ylim(bottom=0)
ax.grid(alpha=0.3)

# Légende compacte à l'extérieur
ax.legend(fontsize=7, ncol=2, loc="upper right",
          framealpha=0.85, handlelength=1.8)

# ── Panel droit : zoom bulk ─────────────────────────────────────────────────
ax = axes[1]
ax.fill_between(t_bulk, density_bulk, color="steelblue", alpha=0.35)
ax.plot(t_bulk, density_bulk, color="steelblue", lw=1.5, label="Bulk density")

# Quelques spikes proches du bulk si présents
for i, lam in enumerate(spikes):
    if -0.5 <= lam <= 3.0:
        ax.axvline(lam, color=cmap(i), lw=1.4, ls="--",
                   dashes=(5, 4), alpha=0.85, label=f"spike {i+1}")

ax.set_xlabel(r"$\lambda$", fontsize=11)
ax.set_ylabel(r"$\phi(\lambda)$", fontsize=11)
ax.set_title("Bulk zoom", fontsize=11)
ax.set_xlim(-0.5, 3.0)
ax.set_ylim(bottom=0)
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

plt.tight_layout()

out_png = "results/spectral/density_cifar10/spectral_density.png"
out_pdf = "results/spectral/density_cifar10/spectral_density.pdf"
fig.savefig(out_png, dpi=200)
fig.savefig(out_pdf)
plt.close()
print(f"Figures sauvegardées : {out_png}  |  {out_pdf}")
