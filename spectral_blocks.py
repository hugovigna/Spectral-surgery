"""
spectral_blocks.py
------------------
Densité spectrale de la Hessienne par bloc diagonal — ResNet-50 ISIC-2019 v3.
6 groupes anatomiques (un par stage ResNet-50), SLQ pour la distribution.

Groupes :
  conv1  : conv1_conv                         ~9K   ( 0.0%)
  conv2  : conv2_block*                       ~170K ( 0.7%)
  conv3  : conv3_block*                       ~1.1M ( 4.5%)
  conv4  : conv4_block*                       ~7.1M (30.4%)
  conv5  : conv5_block*                       ~14M  (59.4%)
  head   : dense                              ~16K  ( 0.1%)

Chaque bloc : Lanczos m=20 (top λ) puis SLQ m=30 k=5 (densité).
Résultat comparable avec spectral_density_isic2019.py (densité globale).

Usage:
    python3.12 -u spectral_blocks.py 2>&1 | tee results/spectral/blocks_log.txt
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json, time

from spectral_tools import StochasticLanczosQuadrature

CONFIG = {
    "model_path"    : "resnet50_isic2019.keras",
    "cache_train"   : "data/isic2019_cache/train.npz",
    "n_hvp_samples" : 256,
    "lanczos_m_top" : 20,    # Lanczos pour top λ et t_range
    "slq_m"         : 30,    # Ordre quadrature SLQ
    "slq_k"         : 5,     # Nombre de vecteurs sondes
    "sigma2"        : 1e-5,  # Variance noyau gaussien (même que global)
    "seed"          : 42,
    "output_dir"    : "results/spectral/blocks",
}

GROUPS = {
    "conv2" : ["conv2_block"],
    "conv3" : ["conv3_block"],
    "conv4" : ["conv4_block"],
    "conv5" : ["conv5_block"],
    "head"  : ["dense"],
}

GROUP_COLORS = {
    "conv1" : "#BDBDBD",
    "conv2" : "#78909C",
    "conv3" : "#42A5F5",
    "conv4" : "#1565C0",
    "conv5" : "#EF5350",
    "head"  : "#66BB6A",
}


# ── HVP restreint à un sous-ensemble de variables ────────────────────────────

class RestrictedHVP:
    """
    Duck-type de HessianVectorProduct restreint aux variables d'un bloc.
    Compatible avec StochasticLanczosQuadrature (interface .compute + .n_params).
    """
    def __init__(self, model, loss_fn, data_x, data_y, var_list):
        self.model    = model
        self.loss_fn  = loss_fn
        self.data_x   = data_x   # tf.constant déjà créé à l'extérieur
        self.data_y   = data_y
        self.var_list = var_list
        self.n_params = sum(int(np.prod(v.shape)) for v in var_list)
        # Formes pour la reconstruction flat ↔ vars
        self._shapes = [v.shape for v in var_list]
        self._sizes  = [int(np.prod(v.shape)) for v in var_list]

    @tf.function(reduce_retracing=True)
    def _hvp_tf(self, v_flat):
        splits, idx = [], 0
        for shape, size in zip(self._shapes, self._sizes):
            splits.append(tf.reshape(v_flat[idx:idx + size], shape))
            idx += size
        with tf.GradientTape() as t2:
            with tf.GradientTape() as t1:
                preds = self.model(self.data_x, training=False)
                loss  = self.loss_fn(self.data_y, preds)
            grads = t1.gradient(loss, self.var_list)
            gv = tf.add_n([
                tf.reduce_sum(g * s)
                for g, s in zip(grads, splits)
                if g is not None
            ])
        hvp_parts = t2.gradient(gv, self.var_list)
        return tf.concat([tf.reshape(h, [-1]) for h in hvp_parts], axis=0)

    def compute(self, v: np.ndarray) -> np.ndarray:
        return self._hvp_tf(tf.constant(v, dtype=tf.float32)).numpy()


# ── Sélection des variables par pattern ─────────────────────────────────────

def get_vars(model, patterns):
    result = []
    def collect(layer):
        if any(p in layer.name for p in patterns):
            result.extend(layer.trainable_variables)
        else:
            for sub in getattr(layer, "layers", []):
                collect(sub)
    collect(model)
    seen, out = set(), []
    for v in result:
        if id(v) not in seen:
            seen.add(id(v))
            out.append(v)
    return out


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    print("[1] Chargement données ...")
    train_data = np.load(CONFIG["cache_train"])
    rng = np.random.default_rng(CONFIG["seed"])
    idx = rng.choice(len(train_data["imgs"]), CONFIG["n_hvp_samples"], replace=False)
    x_hvp = tf.constant(train_data["imgs"][idx].astype(np.float32))
    y_hvp = tf.constant(train_data["labels"][idx].astype(np.int32))
    print(f"    HVP batch : {CONFIG['n_hvp_samples']} images")

    print("[2] Chargement modèle ...")
    model = tf.keras.models.load_model(CONFIG["model_path"], compile=False)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    total_params = sum(int(np.prod(v.shape)) for v in model.trainable_variables)
    print(f"    Total params : {total_params:,}")

    results = {}

    for group_name, patterns in GROUPS.items():
        print(f"\n{'='*65}")
        print(f"  Bloc : {group_name}  patterns={patterns}")
        var_list = get_vars(model, patterns)
        n_params = sum(int(np.prod(v.shape)) for v in var_list)
        pct = 100 * n_params / total_params
        print(f"    {len(var_list)} variables — {n_params:,} params ({pct:.1f}% du total)")

        if n_params == 0:
            print("    SKIP — aucune variable trouvée")
            continue

        hvp = RestrictedHVP(model, loss_fn, x_hvp, y_hvp, var_list)

        # ── Étape 1 : top eigenvalues via Lanczos ────────────────────────────
        t0 = time.time()
        print(f"    [a] Top eigenvalues (Lanczos m={CONFIG['lanczos_m_top']}) ...")
        slq = StochasticLanczosQuadrature(
            hvp=hvp, n_params=n_params,
            m=CONFIG["lanczos_m_top"], k=1, sigma2=CONFIG["sigma2"],
        )
        ritz_vals, _ = slq.estimate_top_eigenvalues(
            m_lanczos=CONFIG["lanczos_m_top"], verbose=False
        )
        ritz_vals = np.sort(ritz_vals)[::-1]
        lambda_max = float(ritz_vals[0])
        lambda_min = float(ritz_vals[-1])
        print(f"    λ_max={lambda_max:.3f}  λ_min={lambda_min:.3f}")
        print(f"    Top-5 λ : {ritz_vals[:5].round(4).tolist()}")

        # Détection spikes : λ > 3× médiane des valeurs propres positives
        pos = ritz_vals[ritz_vals > 0]
        threshold = float(np.median(pos)) * 3 if len(pos) > 0 else 0.0
        n_spikes  = int(np.sum(ritz_vals > threshold))
        print(f"    Spikes  : {n_spikes}  (λ > {threshold:.4f}, seuil 3× médiane+)")

        # ── Étape 2 : densité SLQ ────────────────────────────────────────────
        t1 = time.time()
        print(f"    [b] SLQ density (m={CONFIG['slq_m']}, k={CONFIG['slq_k']}) ...")
        slq_density = StochasticLanczosQuadrature(
            hvp=hvp, n_params=n_params,
            m=CONFIG["slq_m"], k=CONFIG["slq_k"], sigma2=CONFIG["sigma2"],
        )

        # Plage adaptée à la dynamique du bloc
        t_max  = max(lambda_max * 1.2, 0.1)
        t_min  = min(lambda_min * 1.1, -0.05)
        t_full = np.linspace(t_min, t_max, 2000)
        # Zoom bulk : [-0.5, 5% de t_max] pour voir la masse principale
        t_bulk = np.linspace(t_min, max(t_max * 0.05, 0.5), 1000)

        density_full = slq_density.estimate_density(t_full, verbose=True)
        density_bulk = slq_density.estimate_density(t_bulk, verbose=False)
        dt = time.time() - t0
        print(f"    Durée totale : {dt:.0f}s")

        results[group_name] = {
            "ritz_vals"    : ritz_vals.tolist(),
            "density_full" : density_full.tolist(),
            "density_bulk" : density_bulk.tolist(),
            "t_full"       : t_full.tolist(),
            "t_bulk"       : t_bulk.tolist(),
            "n_params"     : n_params,
            "pct_total"    : round(pct, 2),
            "n_spikes"     : n_spikes,
            "lambda_max"   : lambda_max,
            "lambda_min"   : lambda_min,
            "threshold"    : threshold,
            "duration_s"   : round(dt, 1),
        }

        # Sauvegarde intermédiaire après chaque bloc (run long)
        npz_path = os.path.join(CONFIG["output_dir"], f"{group_name}_density.npz")
        np.savez(npz_path,
                 t_full=t_full, density_full=density_full,
                 t_bulk=t_bulk, density_bulk=density_bulk,
                 ritz_vals=ritz_vals)
        print(f"    Sauvegardé : {npz_path}")

    # ── JSON global ──────────────────────────────────────────────────────────
    json_path = os.path.join(CONFIG["output_dir"], "all_blocks.json")
    json_results = {k: {kk: vv for kk, vv in v.items()
                        if kk not in ("density_full", "density_bulk", "t_full", "t_bulk")}
                    for k, v in results.items()}
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\n[3] JSON sauvegardé : {json_path}")

    # ── Plot : 2 colonnes × 3 rangées (densité pleine + zoom bulk) ──────────
    print("[4] Plot ...")
    n_groups = len(results)
    fig, axes = plt.subplots(n_groups, 2, figsize=(12, 3.5 * n_groups))
    fig.suptitle(
        "Densité spectrale Hessienne par bloc diagonal — ResNet-50 ISIC-2019 v3\n"
        f"SLQ m={CONFIG['slq_m']}, k={CONFIG['slq_k']}, {CONFIG['n_hvp_samples']} samples",
        fontsize=12, fontweight="bold"
    )

    for row, (name, res) in enumerate(results.items()):
        color  = GROUP_COLORS.get(name, "#555")
        t_full = np.array(res["t_full"])
        t_bulk = np.array(res["t_bulk"])
        d_full = np.array(res["density_full"])
        d_bulk = np.array(res["density_bulk"])
        ritz   = np.array(res["ritz_vals"])

        title_info = (f"{name}  —  {res['n_params']:,} params ({res['pct_total']:.1f}%)  "
                      f"| {res['n_spikes']} spike(s)  λ_max={res['lambda_max']:.2f}")

        # Spectre complet
        ax_full = axes[row, 0]
        ax_full.plot(t_full, d_full, color=color, lw=1.2)
        ax_full.fill_between(t_full, d_full, alpha=0.25, color=color)
        for i, lam in enumerate(ritz[:min(res["n_spikes"] + 1, 5)]):
            if lam > 0:
                ax_full.axvline(lam, color="crimson", lw=1.0, ls="--", alpha=0.8,
                                label="spikes" if i == 0 else None)
        if res["threshold"] > 0:
            ax_full.axvline(res["threshold"], color="orange", lw=0.8, ls=":",
                            label=f"seuil {res['threshold']:.3f}")
        ax_full.set_title(title_info, fontsize=8.5)
        ax_full.set_xlabel(r"$\lambda$", fontsize=8)
        ax_full.set_ylabel(r"$\phi(\lambda)$", fontsize=8)
        ax_full.legend(fontsize=7)
        ax_full.grid(alpha=0.3)

        # Zoom bulk
        ax_bulk = axes[row, 1]
        ax_bulk.fill_between(t_bulk, d_bulk, alpha=0.4, color=color)
        ax_bulk.plot(t_bulk, d_bulk, color=color, lw=1.5)
        ax_bulk.set_title(f"{name} — zoom bulk", fontsize=8.5)
        ax_bulk.set_xlabel(r"$\lambda$", fontsize=8)
        ax_bulk.grid(alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(CONFIG["output_dir"], "spectra_blocks.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"    Plot sauvegardé : {plot_path}")

    # ── Résumé ───────────────────────────────────────────────────────────────
    print("\n[5] Résumé :")
    print(f"  {'Bloc':8s}  {'params':>10s}  {'%':>5s}  {'λ_max':>8s}  "
          f"{'spikes':>7s}  {'durée':>7s}")
    for name, res in results.items():
        print(f"  {name:8s}  {res['n_params']:>10,}  {res['pct_total']:>4.1f}%  "
              f"{res['lambda_max']:>8.3f}  {res['n_spikes']:>7d}  {res['duration_s']:>6.0f}s")
    print(f"\n  NPZ  → {CONFIG['output_dir']}/<bloc>_density.npz")
    print(f"  JSON → {json_path}")
    print(f"  Plot → {plot_path}")
