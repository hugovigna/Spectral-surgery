"""
Calcule la matrice de sensibilité S sur le modèle CIFAR-10 baseline (pre-SS)
et reporte son rang effectif. Permet la comparaison directe ISIC-CB vs CIFAR-CE.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf

CIFAR10 = ["avion","auto","oiseau","chat","cerf","chien","grenouille","cheval","bateau","camion"]
N = 10
SEED = 0
N_HVP = 128
N_SPIKES = 9
EPS_PROBE = 0.01
LANCZOS_M = 10

rng = np.random.default_rng(SEED)

print("[1] CIFAR-10 ...")
(x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.cifar10.load_data()
y_tr, y_te = y_tr.flatten(), y_te.flatten()
x_tr = x_tr.astype("float32")/255.0
x_te = x_te.astype("float32")/255.0

idx = rng.permutation(len(x_te))
x_sens, y_sens = x_te[idx[:5000]], y_te[idx[:5000]]

hvp_idx = rng.choice(len(x_tr), N_HVP, replace=False)
x_hvp, y_hvp = x_tr[hvp_idx], y_tr[hvp_idx]

print("[2] Modèle baseline pre-SS ...")
model = tf.keras.models.load_model("resnet50_cifar10.keras", compile=False)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

def per_class_acc(m, x, y):
    p = m.predict(x, verbose=0, batch_size=64).argmax(axis=1)
    return np.array([(p[y==c]==c).mean() if (y==c).sum()>0 else 0.0 for c in range(N)])

def save_w(m): return [v.numpy().copy() for v in m.variables]
def restore_w(m, w):
    for v, w_ in zip(m.variables, w):
        v.assign(tf.constant(w_, dtype=v.dtype))
def apply_pert(m, d):
    i = 0
    for v in m.trainable_variables:
        s = int(np.prod(v.shape))
        v.assign_add(tf.constant(d[i:i+s].reshape(v.shape), dtype=v.dtype))
        i += s

print("[3] Lanczos top-9 (CPU) ...")
from spectral_tools import HessianVectorProduct, StochasticLanczosQuadrature

gpu = tf.config.list_physical_devices('GPU')
if gpu:
    with tf.device('/CPU:0'):
        cpu_model = tf.keras.models.clone_model(model)
        cpu_model.set_weights(model.get_weights())
    hvp_model = cpu_model
else:
    hvp_model = model

hvp = HessianVectorProduct(model=hvp_model, loss_fn=loss_fn,
                           data_x=x_hvp, data_y=y_hvp, batch_size=32)
slq = StochasticLanczosQuadrature(hvp=hvp, n_params=hvp.n_params, m=20, k=1, sigma2=1e-5)
ritz_vals, ritz_vecs = slq.estimate_top_eigenvalues(m_lanczos=LANCZOS_M, verbose=False)
print(f"    λ top-9 : {ritz_vals[:N_SPIKES].round(1).tolist()}")

print("[4] Matrice S (sensibilité) ...")
n_sp = min(N_SPIKES, ritz_vecs.shape[1])
current_w = save_w(model)
S = np.zeros((n_sp, N))
for s in range(n_sp):
    delta = (EPS_PROBE * ritz_vecs[:, s]).astype(np.float32)
    apply_pert(model, delta)
    acc_pos = per_class_acc(model, x_sens, y_sens)
    apply_pert(model, -2*delta)
    acc_neg = per_class_acc(model, x_sens, y_sens)
    restore_w(model, current_w)
    S[s] = (acc_pos - acc_neg) / (2*EPS_PROBE)
    print(f"    spike {s+1}/{n_sp} done  λ={ritz_vals[s]:.1f}")

print("\nMatrice S (CIFAR-10 baseline pre-SS) :")
header = "  spike  " + "  ".join(f"{c[:5]:>5s}" for c in CIFAR10)
print(header)
for s in range(n_sp):
    row = "  ".join(f"{S[s,c]:>+5.2f}" for c in range(N))
    print(f"  q{s+1:02d}    {row}")

sv = np.linalg.svd(S, compute_uv=False)
stable = (sv**2).sum() / sv[0]**2
p = sv**2 / (sv**2).sum()
H = -np.sum(p * np.log(p + 1e-15))
eff_H = np.exp(H)

print(f"\nValeurs singulières : {sv.round(2).tolist()}")
print(f"  ‖S‖_F            = {np.linalg.norm(S, 'fro'):.2f}")
print(f"  σ_max            = {sv[0]:.2f}")
print(f"  σ_max/σ_2        = {sv[0]/sv[1]:.2f}")
print(f"  rang stable      = {stable:.3f}")
print(f"  rang effectif H  = {eff_H:.3f}")
print(f"  énergie top-2    = {(p[0]+p[1])*100:.1f}%")
print(f"  énergie top-3    = {(p[0]+p[1]+p[2])*100:.1f}%")

np.savez("results/cifar10/baseline_S.npz", S=S, ritz_vals=ritz_vals[:n_sp], sv=sv)
print("\nSauvegardé → results/cifar10/baseline_S.npz")
