"""
compare_posthoc_cifar10.py
--------------------------
Compare τ-normalization et logit adjustment à Hessian Surgery sur CIFAR-10.

Références :
  τ-norm      : Kang et al. "Decoupling Representation and Classifier
                for Long-Tailed Recognition" (ICLR 2020)
  Logit Adj.  : Menon et al. "Long-tail learning via logistic adjustment"
                (ICLR 2021)

Usage :
    python3.12 compare_posthoc_cifar10.py 2>&1 | tee results/cifar10/compare_posthoc.txt
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
import pandas as pd
import time

CLASSES = ["avion", "auto", "oiseau", "chat", "cerf",
           "chien", "grenouille", "cheval", "bateau", "camion"]
C       = len(CLASSES)
SEED    = 0

# SS résultats de référence (depuis results/cifar10/ss/)
SS_RESULTS = {
    "baseline": {
        "global": 0.8406, "std": 0.0917,
        "per_class": [0.884, 0.915, 0.806, 0.638, 0.874,
                      0.708, 0.867, 0.901, 0.931, 0.897],
    },
    "SS (10 iter, γ-decay)": {
        "global": 0.8468, "std": 0.0552,
        "per_class": [0.859, 0.924, 0.832, 0.761, 0.817,
                      0.749, 0.892, 0.840, 0.900, 0.884],
    },
}

# ════════════════════════════════════════════════════════════════════════════

def per_class_accuracy(preds, y):
    return np.array([(preds[y == c] == c).mean() for c in range(C)])

def print_row(name, global_acc, std, per_class, baseline_per_class=None, elapsed=None):
    gains = ""
    if baseline_per_class is not None:
        g = [(per_class[c] - baseline_per_class[c]) * 100 for c in range(C)]
        worst  = min(g)
        best   = max(g)
        gains  = f"  worst={worst:+.1f}pp  best={best:+.1f}pp"
    t_str  = f"  {elapsed:.1f}s" if elapsed is not None else ""
    cls_str = "  ".join(f"{per_class[c]*100:5.1f}" for c in range(C))
    print(f"  {name:32s}  glob={global_acc*100:.2f}%  σ={std*100:.2f}%  "
          f"[{cls_str}]{gains}{t_str}")


# ════════════════════════════════════════════════════════════════════════════
# 1. Chargement données + modèle
# ════════════════════════════════════════════════════════════════════════════

print("[1] Chargement CIFAR-10 ...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_train = y_train.flatten().astype(np.int32)
y_test  = y_test.flatten().astype(np.int32)
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0

rng = np.random.default_rng(SEED)
idx_test   = rng.permutation(len(x_test))
x_eval, y_eval = x_test[idx_test[5000:]], y_test[idx_test[5000:]]
print(f"    Held-out eval : {len(x_eval)} images")

# Fréquences d'entraînement par classe (pour logit adjustment)
n_train    = len(y_train)
pi_train   = np.array([(y_train == c).sum() / n_train for c in range(C)])
print(f"    Fréquences train : {pi_train.round(3).tolist()}")

print("\n[2] Chargement modèle ...")
model    = tf.keras.models.load_model("resnet50_cifar10.keras", compile=False)
logits_m = tf.keras.Model(inputs=model.input,
                          outputs=model.layers[-1].output)  # avant softmax si dispo
# Vérifier si la dernière couche est Dense (pas softmax)
last_layer = model.layers[-1]
print(f"    Dernière couche : {last_layer.name}  ({last_layer.__class__.__name__})")

# Obtenir les logits : si le modèle sort des proba (softmax final),
# on intercepte la couche Dense qui précède.
dense_layer = None
for layer in reversed(model.layers):
    if isinstance(layer, tf.keras.layers.Dense):
        dense_layer = layer
        break
print(f"    Couche Dense    : {dense_layer.name}  "
      f"shape W={dense_layer.kernel.shape}")

# Modèle jusqu'à la couche Dense (logits bruts)
logit_model = tf.keras.Model(inputs=model.input, outputs=dense_layer.output)

# ════════════════════════════════════════════════════════════════════════════
# Helper : évaluer depuis les logits bruts
# ════════════════════════════════════════════════════════════════════════════

def get_logits(x, batch_size=256):
    logits_list = []
    for i in range(0, len(x), batch_size):
        logits_list.append(logit_model(x[i:i+batch_size], training=False).numpy())
    return np.concatenate(logits_list, axis=0)

def eval_from_logits(logits, y):
    preds      = logits.argmax(axis=1)
    global_acc = (preds == y).mean()
    per_class  = per_class_accuracy(preds, y)
    return global_acc, per_class

# ════════════════════════════════════════════════════════════════════════════
# 2. Baseline
# ════════════════════════════════════════════════════════════════════════════

print("\n[3] Logits baseline ...")
t0 = time.time()
logits_eval = get_logits(x_eval)
acc0, pc0   = eval_from_logits(logits_eval, y_eval)
std0        = float(np.std(pc0))
t_baseline  = time.time() - t0

header = "  " + "  ".join(f"{n[:5]:>5s}" for n in CLASSES)
print(f"\n{header}")
print("─" * 100)
print_row("Baseline", acc0, std0, pc0, elapsed=t_baseline)

results = [{"method": "Baseline", "tau": None, "elapsed_s": round(t_baseline, 2),
            "global": acc0, "std": std0,
            **{c: pc0[i] for i, c in enumerate(CLASSES)}}]

# ════════════════════════════════════════════════════════════════════════════
# 3. τ-normalization
# ════════════════════════════════════════════════════════════════════════════
# W[:, c] ← W[:, c] / ||W[:, c]||^τ
# Appliqué uniquement sur la couche Dense finale, backbone inchangé.
# ════════════════════════════════════════════════════════════════════════════

print("\n[4] τ-normalization ...")
W_orig, b_orig = dense_layer.get_weights()
col_norms      = np.linalg.norm(W_orig, axis=0)   # shape (C,)
print(f"    Normes colonnes W (par classe) :")
for i, (n, norm) in enumerate(zip(CLASSES, col_norms)):
    print(f"      {n:12s} : {norm:.4f}")

tau_values = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
best_tau_norm = {"std": np.inf}

for tau in tau_values:
    t0 = time.time()
    W_new = W_orig / (col_norms[np.newaxis, :] ** tau)
    dense_layer.set_weights([W_new, b_orig])
    logits_tau  = get_logits(x_eval)
    acc_t, pc_t = eval_from_logits(logits_tau, y_eval)
    std_t       = float(np.std(pc_t))
    elapsed_t   = time.time() - t0
    print_row(f"τ-norm τ={tau}", acc_t, std_t, pc_t, pc0, elapsed=elapsed_t)
    row = {"method": "tau-norm", "tau": tau, "elapsed_s": round(elapsed_t, 2),
           "global": acc_t, "std": std_t,
           **{c: pc_t[i] for i, c in enumerate(CLASSES)}}
    results.append(row)
    if std_t < best_tau_norm["std"]:
        best_tau_norm = {**row}

# Restaurer les poids originaux
dense_layer.set_weights([W_orig, b_orig])

# ════════════════════════════════════════════════════════════════════════════
# 4. Logit Adjustment
# ════════════════════════════════════════════════════════════════════════════
# logit_c(x) -= τ · log(π_c)
# Sur CIFAR-10 (équilibré), π_c = 1/10 ∀c → ajustement constant → pas d'effet.
# On le montre explicitement et on note la borne théorique.
# ════════════════════════════════════════════════════════════════════════════

print("\n[5] Logit Adjustment ...")
log_pi = np.log(pi_train)   # shape (C,)
print(f"    log(π_c) : {log_pi.round(4).tolist()}")
print(f"    Variation max log(π) : {log_pi.max() - log_pi.min():.6f}")
print("    → CIFAR-10 est équilibré (π_c = 1/C ∀c) : l'ajustement est constant")
print("      sur toutes les classes et annulé par le softmax. Effet nul par construction.\n")

tau_la_values = [0.5, 1.0, 2.0]
for tau in tau_la_values:
    t0           = time.time()
    adj          = tau * log_pi
    logits_adj   = logits_eval - adj[np.newaxis, :]
    acc_a, pc_a  = eval_from_logits(logits_adj, y_eval)
    std_a        = float(np.std(pc_a))
    elapsed_a    = time.time() - t0
    print_row(f"Logit Adj. τ={tau}", acc_a, std_a, pc_a, pc0, elapsed=elapsed_a)
    results.append({"method": "logit-adj", "tau": tau, "elapsed_s": round(elapsed_a, 3),
                    "global": acc_a, "std": std_a,
                    **{c: pc_a[i] for i, c in enumerate(CLASSES)}})

# ════════════════════════════════════════════════════════════════════════════
# 5. Résumé comparatif
# ════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*90}")
print("  RÉSUMÉ COMPARATIF (held-out 5000 images, seed=0)")
print(f"{'='*90}")
print(f"{header}")
print("─" * 90)

# Baseline
print_row("Baseline", acc0, std0, pc0)

# SS (depuis CSV)
ss_pc = SS_RESULTS["SS (10 iter, γ-decay)"]["per_class"]
print_row("SS 10 iter γ-decay [CSV]",
          SS_RESULTS["SS (10 iter, γ-decay)"]["global"],
          SS_RESULTS["SS (10 iter, γ-decay)"]["std"],
          ss_pc, pc0)

# Meilleur τ-norm
best_pc = [best_tau_norm[c] for c in CLASSES]
print_row(f"τ-norm best (τ={best_tau_norm['tau']})",
          best_tau_norm["global"], best_tau_norm["std"], best_pc, pc0)

# Logit adj est trivial — on l'indique
print(f"  {'Logit Adj. (tous τ)':30s}  → identique baseline (CIFAR-10 équilibré)")

print(f"{'='*90}")
print(f"\n  Note : SS Adam 15 iter en cours (résultats à lire dans")
print(f"         results/cifar10/ss/eval_heldout.csv quand terminé)")

# ════════════════════════════════════════════════════════════════════════════
# 6. Sauvegarde
# ════════════════════════════════════════════════════════════════════════════

os.makedirs("results/cifar10", exist_ok=True)
df = pd.DataFrame(results)
df.to_csv("results/cifar10/compare_posthoc.csv", index=False)
print(f"\n  CSV → results/cifar10/compare_posthoc.csv")

# Table LaTeX prête à coller
print("\n  Table LaTeX :")
print("  \\begin{tabular}{lccc}")
print("  \\toprule")
print("  Method & Global (\\%) & $\\sigma$ (\\%) & Chat (\\%) \\\\ \\midrule")
print(f"  Baseline & {acc0*100:.2f} & {std0*100:.2f} & {pc0[3]*100:.1f} \\\\")
print(f"  $\\tau$-norm ($\\tau$={best_tau_norm['tau']}) "
      f"& {best_tau_norm['global']*100:.2f} "
      f"& {best_tau_norm['std']*100:.2f} "
      f"& {best_tau_norm['chat']*100:.1f} \\\\")
print(f"  Logit Adj. & \\multicolumn{{3}}{{c}}{{trivial (dataset équilibré)}} \\\\")
ss = SS_RESULTS["SS (10 iter, γ-decay)"]
print(f"  Hessian Surgery & {ss['global']*100:.2f} & {ss['std']*100:.2f} "
      f"& {ss['per_class'][3]*100:.1f} \\\\")
print("  \\bottomrule")
print("  \\end{tabular}")
