"""
compute_sensitivity_matrix.py
Calcule et sauvegarde la matrice de sensibilité S[spike, classe]
pour le modèle baseline CIFAR-10 (avant Surgery).
Génère un heatmap pour l'article.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CIFAR10_CLASSES = [
    "plane", "car", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

def per_class_accuracy(model, x, y, n_classes=10):
    preds = model.predict(x, verbose=0, batch_size=256).argmax(axis=1)
    return np.array([(preds[y == c] == c).mean() for c in range(n_classes)])

def save_weights(model):
    return [v.numpy().copy() for v in model.trainable_variables]

def restore_weights(model, saved):
    for var, w in zip(model.trainable_variables, saved):
        var.assign(w)

def apply_perturbation(model, delta_flat):
    offset = 0
    for var in model.trainable_variables:
        size = int(np.prod(var.shape))
        var.assign_add(
            tf.constant(delta_flat[offset:offset+size].reshape(var.shape),
                        dtype=var.dtype))
        offset += size

# ── Données ──────────────────────────────────────────────────────────
print("[1] Chargement de CIFAR-10 ...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_train, y_test = y_train.flatten(), y_test.flatten()
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0

# Split held-out (même seed)
rng = np.random.default_rng(0)
idx_test = rng.permutation(len(x_test))
x_sens, y_sens = x_test[idx_test[:5000]], y_test[idx_test[:5000]]

# ── Modèle et eigenvecteurs ──────────────────────────────────────────
print("[2] Chargement du modèle baseline ...")
model = tf.keras.models.load_model("resnet50_cifar10.keras")
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

print("[3] Calcul des eigenvecteurs (Lanczos m=10) ...")
from spectral_tools import HessianVectorProduct, StochasticLanczosQuadrature

hvp_idx = rng.choice(len(x_train), 128, replace=False)
x_hvp, y_hvp = x_train[hvp_idx], y_train[hvp_idx]

hvp = HessianVectorProduct(
    model=model, loss_fn=loss_fn,
    data_x=x_hvp, data_y=y_hvp, batch_size=None,
)
slq = StochasticLanczosQuadrature(
    hvp=hvp, n_params=hvp.n_params, m=20, k=1, sigma2=1e-5,
)
ritz_vals, ritz_vecs = slq.estimate_top_eigenvalues(m_lanczos=10, verbose=True)
n_spikes = 9
print(f"    Top-9 eigenvalues: {ritz_vals[:n_spikes].round(1).tolist()}")

# ── Matrice de sensibilité ───────────────────────────────────────────
print("[4] Calcul de la matrice de sensibilité ...")
eps_probe = 0.02
current_w = save_weights(model)
S = np.zeros((n_spikes, 10))

for s in range(n_spikes):
    qi = ritz_vecs[:, s]
    restore_weights(model, current_w)
    apply_perturbation(model, (eps_probe * qi).astype(np.float32))
    acc_pos = per_class_accuracy(model, x_sens, y_sens)
    restore_weights(model, current_w)
    apply_perturbation(model, (-eps_probe * qi).astype(np.float32))
    acc_neg = per_class_accuracy(model, x_sens, y_sens)
    S[s] = (acc_pos - acc_neg) / (2 * eps_probe)
    print(f"    spike {s+1}/9 done")

restore_weights(model, current_w)

# Sauvegarde
os.makedirs("results/spectral/sensitivity_matrix", exist_ok=True)
np.savez("results/spectral/sensitivity_matrix/sensitivity_data.npz",
         S=S, eigenvalues=ritz_vals[:n_spikes])

# Affichage texte
print("\n  Matrice de sensibilité S (×100):")
header = "         " + "  ".join(f"{n:>6s}" for n in CIFAR10_CLASSES)
print(header)
for s in range(n_spikes):
    row = f"  spike {s+1}: " + "  ".join(f"{S[s,c]*100:>6.1f}" for c in range(10))
    print(row)

# ── Heatmap ──────────────────────────────────────────────────────────
print("\n[5] Génération du heatmap ...")
fig, ax = plt.subplots(figsize=(10, 5))
vmax = np.abs(S * 100).max()
im = ax.imshow(S * 100, cmap="RdBu_r", aspect="auto",
               vmin=-vmax, vmax=vmax)

ax.set_xticks(range(10))
ax.set_xticklabels(CIFAR10_CLASSES, fontsize=10)
ax.set_yticks(range(n_spikes))
ylabels = [f"spike {i+1}\n($\\lambda={ritz_vals[i]:.0f}$)" for i in range(n_spikes)]
ax.set_yticklabels(ylabels, fontsize=9)
ax.set_xlabel("Class", fontsize=11)
ax.set_ylabel("Spike eigenvector", fontsize=11)

# Annotations
for i in range(n_spikes):
    for j in range(10):
        val = S[i, j] * 100
        color = "white" if abs(val) > vmax * 0.6 else "black"
        ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                fontsize=8, color=color)

cbar = fig.colorbar(im, ax=ax, label="Sensitivity (%/unit)")
ax.set_title("Spike-Class Sensitivity Matrix $S$ (baseline ResNet-50 / CIFAR-10)",
             fontsize=12)

plt.tight_layout()
fig.savefig("results/spectral/sensitivity_matrix/sensitivity_heatmap.png", dpi=200)
fig.savefig("results/spectral/sensitivity_matrix/sensitivity_heatmap.pdf")
plt.close()
print("  Sauvegardé dans results/spectral/sensitivity_matrix/")
