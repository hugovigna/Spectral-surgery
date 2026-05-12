"""
compare_posthoc_isic2019.py
----------------------------
Compare sur ISIC-2019 held-out (test_ss.npz, 1900 imgs) :
  - τ-normalization   sur le modèle baseline (cross-entropy)
  - Logit Adjustment  sur le modèle baseline
  - τ-normalization   sur le modèle FL (focal loss pré-entraîné)
  - Logit Adjustment  sur le modèle FL
  - SS seule + FL+SS  (chargés depuis les CSV existants)

Références :
  τ-norm     : Kang et al. ICLR 2020
  Logit Adj. : Menon et al. ICLR 2021

Usage :
    python3.12 -u compare_posthoc_isic2019.py 2>&1 | tee results/isic2019/isic2019_log.txt
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
import pandas as pd
import time

CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
C       = len(CLASSES)

MODEL_BASELINE = "resnet50_isic2019.keras"
MODEL_FL       = "results/isic2019/focal_loss/model_focal.keras"
CACHE_TEST     = "data/isic2019_cache/test_ss.npz"
CACHE_TRAIN    = "data/isic2019_cache/train.npz"
OUT_DIR        = "results/isic2019"

# Résultats SS existants (test set, depuis eval_heldout.csv / article)
SS_EXISTING = {
    "Baseline (CE)": {
        "global": 0.6978, "bal_acc": 0.3752, "std": 0.2360,
        "per_class": [0.472, 0.859, 0.744, 0.262, 0.365, 0.333, 0.105, 0.319],
    },
    "FL seule": {
        "global": 0.7247, "bal_acc": 0.4920, "std": 0.2081,
        "per_class": [0.584, 0.887, 0.720, 0.354, 0.467, 0.278, 0.263, 0.383],
    },
    "SS seule": {
        "global": 0.6920, "bal_acc": 0.5010, "std": 0.1720,
        "per_class": [0.510, 0.844, 0.675, 0.369, 0.528, 0.444, 0.289, 0.351],
    },
    "FL + SS": {
        "global": 0.6980, "bal_acc": 0.5150, "std": 0.1640,
        "per_class": [0.581, 0.828, 0.652, 0.415, 0.543, 0.389, 0.263, 0.447],
    },
}

TAU_VALUES    = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
TAU_LA_VALUES = [0.25, 0.5, 1.0, 1.5, 2.0]

# ════════════════════════════════════════════════════════════════════════════

def per_class_accuracy(preds, y):
    return np.array(
        [(preds[y == c] == c).mean() if (y == c).sum() > 0 else 0.0
         for c in range(C)]
    )

def eval_from_logits(logits, y):
    preds      = logits.argmax(axis=1)
    global_acc = (preds == y).mean()
    pc         = per_class_accuracy(preds, y)
    return global_acc, pc

def get_logits(logit_model, x, batch_size=64):
    out = []
    for i in range(0, len(x), batch_size):
        out.append(logit_model(x[i:i+batch_size], training=False).numpy())
    return np.concatenate(out, axis=0)

def get_dense_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Dense):
            return layer
    raise RuntimeError("Pas de couche Dense trouvée")

def print_row(name, global_acc, bal_acc, std, per_class, ref_pc=None, elapsed=None):
    t_str = f"  {elapsed:.2f}s" if elapsed is not None else ""
    cls_str = "  ".join(f"{per_class[c]*100:5.1f}" for c in range(C))
    if ref_pc is not None:
        g = [(per_class[c] - ref_pc[c]) * 100 for c in range(C)]
        gain_str = f"  Δbest={max(g):+.1f}pp  Δworst={min(g):+.1f}pp"
    else:
        gain_str = ""
    print(f"  {name:35s}  glob={global_acc*100:.1f}%  "
          f"bal={bal_acc*100:.1f}%  σ={std*100:.1f}%  "
          f"[{cls_str}]{gain_str}{t_str}")

def apply_tau_norm(dense_layer, tau):
    W, b = dense_layer.get_weights()
    col_norms = np.linalg.norm(W, axis=0)
    W_new = W / (col_norms[np.newaxis, :] ** tau)
    dense_layer.set_weights([W_new, b])

def restore_dense(dense_layer, W_orig, b_orig):
    dense_layer.set_weights([W_orig, b_orig])

# ════════════════════════════════════════════════════════════════════════════
# 1. Chargement
# ════════════════════════════════════════════════════════════════════════════

print("[1] Chargement données ...")
test_data  = np.load(CACHE_TEST)
x_test     = test_data["imgs"].astype(np.float32)
y_test     = test_data["labels"].astype(np.int32)
print(f"    Test : {len(y_test)} imgs  "
      + ", ".join(f"{c}:{(y_test==i).sum()}" for i,c in enumerate(CLASSES)))

train_data = np.load(CACHE_TRAIN)
y_train    = train_data["labels"].astype(np.int32)
n_train    = len(y_train)
pi_train   = np.array([(y_train == c).sum() / n_train for c in range(C)])
log_pi     = np.log(pi_train)
print(f"\n    Fréquences train (π_c) :")
for i, (n, pi) in enumerate(zip(CLASSES, pi_train)):
    print(f"      {n:6s}: {pi*100:.1f}%  log(π)={log_pi[i]:.3f}")

results = []
header  = "  " + "  ".join(f"{n:>5s}" for n in CLASSES)

# ════════════════════════════════════════════════════════════════════════════
# Fonction principale : évaluer un modèle avec tous les post-hoc
# ════════════════════════════════════════════════════════════════════════════

def run_posthoc(model_path, model_label, ref_label):
    print(f"\n{'='*90}")
    print(f"  Modèle : {model_label}  ({model_path})")
    print(f"{'='*90}")

    model = tf.keras.models.load_model(model_path, compile=False)
    dense = get_dense_layer(model)
    print(f"  Dense : {dense.name}  W={dense.kernel.shape}")

    logit_model = tf.keras.Model(inputs=model.input, outputs=dense.output)

    # ── Baseline logits ──────────────────────────────────────────────────
    print("\n  [a] Logits baseline ...")
    t0          = time.time()
    logits      = get_logits(logit_model, x_test)
    acc0, pc0   = eval_from_logits(logits, y_test)
    std0        = float(np.std(pc0))
    bal0        = float(pc0.mean())
    t_base      = time.time() - t0

    ref_pc = np.array(SS_EXISTING[ref_label]["per_class"])

    print(f"\n{header}")
    print("─" * 110)
    print_row(f"{model_label} (baseline)",
              acc0, bal0, std0, pc0, ref_pc, t_base)
    results.append({"method": f"{model_label}_baseline", "tau": None,
                    "elapsed_s": round(t_base, 2),
                    "global": acc0, "bal_acc": bal0, "std": std0,
                    **{c: pc0[i] for i, c in enumerate(CLASSES)}})

    W_orig, b_orig = dense.get_weights()
    col_norms      = np.linalg.norm(W_orig, axis=0)
    print(f"\n  Normes colonnes W :")
    for i, (n, norm) in enumerate(zip(CLASSES, col_norms)):
        print(f"    {n:6s}: {norm:.4f}")

    # ── τ-normalization ──────────────────────────────────────────────────
    print("\n  [b] τ-normalization ...")
    best_tau = {"std": np.inf}
    for tau in TAU_VALUES:
        t0 = time.time()
        apply_tau_norm(dense, tau)
        logits_tau  = get_logits(logit_model, x_test)
        acc_t, pc_t = eval_from_logits(logits_tau, y_test)
        std_t       = float(np.std(pc_t))
        bal_t       = float(pc_t.mean())
        elapsed_t   = time.time() - t0
        restore_dense(dense, W_orig, b_orig)

        print_row(f"  τ-norm τ={tau}", acc_t, bal_t, std_t, pc_t, ref_pc, elapsed_t)
        row = {"method": f"{model_label}_tau-norm", "tau": tau,
               "elapsed_s": round(elapsed_t, 2),
               "global": acc_t, "bal_acc": bal_t, "std": std_t,
               **{c: pc_t[i] for i, c in enumerate(CLASSES)}}
        results.append(row)
        if std_t < best_tau["std"]:
            best_tau = {**row}

    print(f"\n  → Best τ-norm : τ={best_tau['tau']}  "
          f"σ={best_tau['std']*100:.1f}%  bal={best_tau['bal_acc']*100:.1f}%")

    # ── Logit Adjustment ─────────────────────────────────────────────────
    print("\n  [c] Logit Adjustment ...")
    best_la = {"std": np.inf}
    for tau in TAU_LA_VALUES:
        t0           = time.time()
        adj          = tau * log_pi          # shape (C,)
        logits_adj   = logits - adj[np.newaxis, :]
        acc_a, pc_a  = eval_from_logits(logits_adj, y_test)
        std_a        = float(np.std(pc_a))
        bal_a        = float(pc_a.mean())
        elapsed_a    = time.time() - t0

        print_row(f"  Logit Adj. τ={tau}", acc_a, bal_a, std_a, pc_a, ref_pc, elapsed_a)
        row = {"method": f"{model_label}_logit-adj", "tau": tau,
               "elapsed_s": round(elapsed_a, 4),
               "global": acc_a, "bal_acc": bal_a, "std": std_a,
               **{c: pc_a[i] for i, c in enumerate(CLASSES)}}
        results.append(row)
        if std_a < best_la["std"]:
            best_la = {**row}

    print(f"\n  → Best Logit Adj. : τ={best_la['tau']}  "
          f"σ={best_la['std']*100:.1f}%  bal={best_la['bal_acc']*100:.1f}%")

    return best_tau, best_la

# ════════════════════════════════════════════════════════════════════════════
# Runs
# ════════════════════════════════════════════════════════════════════════════

best_taun_base, best_la_base = run_posthoc(
    MODEL_BASELINE, "Baseline (CE)", "Baseline (CE)"
)
best_taun_fl, best_la_fl = run_posthoc(
    MODEL_FL, "FL", "FL seule"
)

# ════════════════════════════════════════════════════════════════════════════
# Résumé final
# ════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'═'*110}")
print("  RÉSUMÉ COMPARATIF — ISIC-2019 test set (1900 imgs)")
print(f"{'═'*110}")
print(f"{header}")
print("─" * 110)

ref_base = np.array(SS_EXISTING["Baseline (CE)"]["per_class"])

for label, info in SS_EXISTING.items():
    pc = np.array(info["per_class"])
    print_row(label, info["global"], info["bal_acc"], info["std"], pc,
              ref_base if label != "Baseline (CE)" else None)

print("─" * 110)

for name, row in [
    (f"τ-norm CE  (τ={best_taun_base['tau']})", best_taun_base),
    (f"Logit Adj. CE (τ={best_la_base['tau']})", best_la_base),
    (f"τ-norm FL  (τ={best_taun_fl['tau']})", best_taun_fl),
    (f"Logit Adj. FL (τ={best_la_fl['tau']})", best_la_fl),
]:
    pc = np.array([row[c] for c in CLASSES])
    print_row(name, row["global"], row["bal_acc"], row["std"], pc, ref_base)

print(f"{'═'*110}")

# ════════════════════════════════════════════════════════════════════════════
# Sauvegarde
# ════════════════════════════════════════════════════════════════════════════

os.makedirs(OUT_DIR, exist_ok=True)

# Ajouter les résultats SS existants
for label, info in SS_EXISTING.items():
    results.insert(0, {
        "method": label, "tau": None, "elapsed_s": None,
        "global": info["global"], "bal_acc": info["bal_acc"], "std": info["std"],
        **{c: info["per_class"][i] for i, c in enumerate(CLASSES)},
    })

df = pd.DataFrame(results)
path = os.path.join(OUT_DIR, "isic2019_posthoc.csv")
df.to_csv(path, index=False)
print(f"\n  CSV → {path}")
