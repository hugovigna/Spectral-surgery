"""
spike_optimizer_cifar100.py
----------------------------
Hessian Surgery sur CIFAR-100 : même pipeline que spike_optimizer.py
mais adapté pour 100 classes.

Différences avec CIFAR-10 :
  - 100 classes → théoriquement ~99 spikes
  - On utilise n_spikes=30 (top 30 vecteurs propres, compromis coût/couverture)
  - lanczos_m=40 (pour capturer assez de spikes)
  - eps_probe plus petit (0.005) car plus de classes = plus sensible
  - Matrice de sensibilité 30×100 (vs 9×10)
  - alpha_max_init plus petit (0.01) — espace plus complexe

Usage :
    python spike_optimizer_cifar100.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import time
from scipy.optimize import minimize as scipy_minimize

# ════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════
CONFIG = {
    # ── Modèle et données ──────────────────────────────────────────────────
    "model_path"   : "resnet50_cifar100.keras",
    "n_hvp_samples": 128,
    "seed"         : 0,

    # ── Analyse spectrale ──────────────────────────────────────────────────
    "lanczos_m"    : 40,        # plus grand que CIFAR-10 pour capturer plus de spikes
    "n_spikes"     : 30,        # top 30 sur les ~99 théoriques

    # ── Sondage de sensibilité ─────────────────────────────────────────────
    "eps_probe"    : 0.005,     # plus petit (100 classes = plus sensible)

    # ── Optimisation des coefficients α ───────────────────────────────────
    "alpha_max_init": 0.01,     # plus prudent qu'avec 10 classes
    "alpha_min"    : 0.001,

    # ── Decay adaptatif (EMA) ─────────────────────────────────────────────
    "decay_factor" : 0.7,
    "beta_ema"     : 0.7,

    # ── Sauvegarde ─────────────────────────────────────────────────────────
    "save_model"   : True,
    "model_out"    : "resnet50_cifar100_spiked.keras",

    # ── Boucle principale ──────────────────────────────────────────────────
    "n_iter"       : 10,

    # ── Sortie ────────────────────────────────────────────────────────────
    "output_dir"   : "results/cifar100/spike_optimizer",
}

# Noms des 100 classes CIFAR-100 (fine labels, ordre officiel)
CIFAR100_CLASSES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver",
    "bed", "bee", "beetle", "bicycle", "bottle",
    "bowl", "boy", "bridge", "bus", "butterfly",
    "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach",
    "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox",
    "girl", "hamster", "house", "kangaroo", "keyboard",
    "lamp", "lawn_mower", "leopard", "lion", "lizard",
    "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid",
    "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
    "plain", "plate", "poppy", "porcupine", "possum",
    "rabbit", "raccoon", "ray", "road", "rocket",
    "rose", "sea", "seal", "shark", "shrew",
    "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table",
    "tank", "telephone", "television", "tiger", "tractor",
    "train", "trout", "tulip", "turtle", "wardrobe",
    "whale", "willow_tree", "wolf", "woman", "worm",
]

N_CLASSES = 100

# ════════════════════════════════════════════════════════════════════════════
# Fonctions utilitaires
# ════════════════════════════════════════════════════════════════════════════

def per_class_accuracy(model, x, y, n_classes=N_CLASSES):
    """Retourne un array (n_classes,) avec l'accuracy par classe."""
    preds = model.predict(x, verbose=0, batch_size=256).argmax(axis=1)
    accs = []
    for c in range(n_classes):
        mask = y == c
        if mask.sum() == 0:
            accs.append(0.0)
        else:
            accs.append((preds[mask] == c).mean())
    return np.array(accs)

def save_weights(model):
    return [v.numpy().copy() for v in model.variables]

def restore_weights(model, weights):
    for var, w in zip(model.variables, weights):
        var.assign(tf.constant(w, dtype=var.dtype))

def apply_perturbation(model, delta_flat):
    idx = 0
    for var in model.trainable_variables:
        size = np.prod(var.shape)
        var.assign_add(tf.constant(
            delta_flat[idx:idx+size].reshape(var.shape), dtype=var.dtype
        ))
        idx += size

def compute_eigenvectors(model, loss_fn, x_hvp, y_hvp, lanczos_m, n_spikes):
    from spectral_tools import HessianVectorProduct, StochasticLanczosQuadrature
    hvp = HessianVectorProduct(
        model=model, loss_fn=loss_fn,
        data_x=x_hvp, data_y=y_hvp, batch_size=None,
    )
    slq = StochasticLanczosQuadrature(
        hvp=hvp, n_params=hvp.n_params, m=20, k=1, sigma2=1e-5,
    )
    return slq.estimate_top_eigenvalues(m_lanczos=lanczos_m, verbose=False)

def compute_sensitivity(model, ritz_vecs, n_spikes, eps_probe, x_test, y_test):
    """Matrice de sensibilité S[spike, classe] de taille n_spikes × 100."""
    current_w = save_weights(model)
    S = np.zeros((n_spikes, N_CLASSES))
    for s in range(n_spikes):
        qi = ritz_vecs[:, s]
        restore_weights(model, current_w)
        apply_perturbation(model, (eps_probe * qi).astype(np.float32))
        acc_pos = per_class_accuracy(model, x_test, y_test)
        restore_weights(model, current_w)
        apply_perturbation(model, (-eps_probe * qi).astype(np.float32))
        acc_neg = per_class_accuracy(model, x_test, y_test)
        S[s] = (acc_pos - acc_neg) / (2 * eps_probe)
        if (s + 1) % 5 == 0:
            print(f"    sensibilité spike {s+1}/{n_spikes} calculée")
    restore_weights(model, current_w)
    return S

def optimize_alpha(S, acc_current, alpha_max):
    """
    Même logique que CIFAR-10 : poids ∝ écart relatif, contrainte de norme
    et de non-dégradation des classes fortes.
    """
    acc_best = acc_current.max()
    acc_min  = acc_current.min()
    weights  = (acc_best - acc_current) / max(acc_best - acc_min, 1e-12)
    weights /= weights.sum()

    def objective(alpha):
        return -np.dot(weights, S.T @ alpha)

    def constraint_norm(alpha):
        return alpha_max - np.linalg.norm(alpha)

    def constraint_no_degrade(alpha):
        predicted_delta = S.T @ alpha
        strong = acc_current > 0.85
        if strong.sum() == 0:
            return 1.0
        return np.min(predicted_delta[strong]) + 0.01

    result = scipy_minimize(
        objective, x0=np.zeros(len(S)),
        method='SLSQP',
        constraints=[
            {'type': 'ineq', 'fun': constraint_norm},
            {'type': 'ineq', 'fun': constraint_no_degrade},
        ],
    )
    return result.x


# ════════════════════════════════════════════════════════════════════════════
# Classe principale
# ════════════════════════════════════════════════════════════════════════════

class SpikeOptimizerCIFAR100:
    """
    Hessian Surgery pour CIFAR-100.
    Même pipeline que CIFAR-10, adapté pour 100 classes.
    """

    def __init__(self, model, loss_fn, x_test, y_test, x_hvp, y_hvp, cfg):
        self.model   = model
        self.loss_fn = loss_fn
        self.x_test  = x_test
        self.y_test  = y_test
        self.x_hvp   = x_hvp
        self.y_hvp   = y_hvp
        self.cfg     = cfg
        os.makedirs(cfg["output_dir"], exist_ok=True)

    def run(self):
        cfg = self.cfg
        alpha_max    = cfg["alpha_max_init"]
        beta         = cfg["beta_ema"]
        log          = []

        # Baseline
        acc_baseline = per_class_accuracy(self.model, self.x_test, self.y_test)
        acc_global_0 = self.model.evaluate(
            self.x_test, self.y_test, verbose=0, batch_size=256)[1]
        self._print_baseline(acc_baseline, acc_global_0)

        acc_current = acc_baseline.copy()

        # EMA sur la std
        std_ema      = float(np.std(acc_baseline))
        best_std_ema = std_ema

        print(f"\n{'='*72}")
        print(f"  ITÉRATIONS — {cfg['n_iter']} chocs  "
              f"α_init={alpha_max}  decay={cfg['decay_factor']}  "
              f"β_ema={beta}  n_spikes={cfg['n_spikes']}")
        print(f"{'='*72}")

        for it in range(1, cfg["n_iter"] + 1):
            t0 = time.time()
            print(f"\n  --- Itération {it}/{cfg['n_iter']}  ‖α‖≤{alpha_max:.4f} ---")

            # Eigenvectors
            ritz_vals, ritz_vecs = compute_eigenvectors(
                self.model, self.loss_fn, self.x_hvp, self.y_hvp,
                cfg["lanczos_m"], cfg["n_spikes"],
            )
            n_sp = min(cfg["n_spikes"], ritz_vecs.shape[1])
            print(f"  λ top-5 : {ritz_vals[:5].round(1).tolist()}")

            # Sensibilité (30×100 — plus long que CIFAR-10)
            print(f"  Calcul sensibilité ({n_sp} spikes × {N_CLASSES} classes) ...")
            S = compute_sensitivity(
                self.model, ritz_vecs, n_sp,
                cfg["eps_probe"], self.x_test, self.y_test,
            )

            # Optimiser α
            alpha = optimize_alpha(S, acc_current, alpha_max)

            # Sauvegarder poids avant choc
            weights_before = save_weights(self.model)
            std_before     = float(np.std(acc_current))

            # Appliquer choc
            delta = np.zeros(ritz_vecs.shape[0], dtype=np.float64)
            for s in range(n_sp):
                delta += alpha[s] * ritz_vecs[:, s]
            apply_perturbation(self.model, delta.astype(np.float32))

            # Mesurer
            acc_new    = per_class_accuracy(self.model, self.x_test, self.y_test)
            acc_global = self.model.evaluate(
                self.x_test, self.y_test, verbose=0, batch_size=256)[1]
            delta_acc  = acc_new - acc_current
            cur_std    = float(np.std(acc_new))

            # Rollback si dégradation forte
            rolled_back = False
            if cur_std > std_before + 0.005:
                std_measured = cur_std
                restore_weights(self.model, weights_before)
                cur_std   = std_before
                acc_new   = acc_current.copy()
                delta_acc = np.zeros_like(acc_current)
                acc_global = self.model.evaluate(
                    self.x_test, self.y_test, verbose=0, batch_size=256)[1]
                rolled_back = True
                print(f"  [ROLLBACK] std {std_before:.4f}→{std_measured:.4f}")

            else:
                acc_current = acc_new.copy()

            # EMA + decay
            std_ema = beta * std_ema + (1.0 - beta) * cur_std
            if rolled_back:
                alpha_max = max(alpha_max * cfg["decay_factor"], cfg["alpha_min"])
                best_std_ema = std_ema
                print(f"  [decay] rollback → α_max={alpha_max:.4f}")
            elif std_ema < best_std_ema - 1e-4:
                best_std_ema = std_ema
            else:
                alpha_max = max(alpha_max * cfg["decay_factor"], cfg["alpha_min"])
                print(f"  [decay] std_ema={std_ema:.4f} → α_max={alpha_max:.4f}")

            elapsed = time.time() - t0

            # Affichage résumé
            rb_tag = " ↩ROLLBACK" if rolled_back else ""
            print(f"  global={acc_global:.4f}  std={cur_std:.4f}  "
                  f"ema={std_ema:.4f}  ({elapsed:.0f}s){rb_tag}")

            # Top 5 pires et meilleures classes
            worst5 = np.argsort(acc_new)[:5]
            best5  = np.argsort(acc_new)[-5:][::-1]
            print(f"  Pires 5 : " + "  ".join(
                f"{CIFAR100_CLASSES[c][:8]}={acc_new[c]*100:.0f}%" for c in worst5))
            print(f"  Mieux 5 : " + "  ".join(
                f"{CIFAR100_CLASSES[c][:8]}={acc_new[c]*100:.0f}%" for c in best5))

            # Classes avec plus gros changements
            big_delta = np.argsort(np.abs(delta_acc))[-5:][::-1]
            if not rolled_back:
                print(f"  Δ max   : " + "  ".join(
                    f"{CIFAR100_CLASSES[c][:8]}={delta_acc[c]*100:+.1f}%"
                    for c in big_delta if abs(delta_acc[c]) > 0.005))

            log.append({
                "iteration" : it,
                "acc_global": float(acc_global),
                "std"       : cur_std,
                "std_ema"   : std_ema,
                "alpha_max" : alpha_max,
                "lambda_max": float(ritz_vals[0]),
                "alpha_norm": float(np.linalg.norm(alpha)),
                "rolled_back": rolled_back,
                "elapsed_s" : elapsed,
                **{CIFAR100_CLASSES[c]: float(acc_new[c]) for c in range(N_CLASSES)},
            })

        self._print_summary(acc_baseline, acc_global_0, log)
        self._save(acc_baseline, acc_global_0, log)
        self._plot(acc_baseline, log)

        if cfg.get("save_model"):
            self.model.save(cfg["model_out"])
            print(f"\n  Modèle sauvegardé : {cfg['model_out']}")

        return log

    def _print_baseline(self, acc, acc_global):
        print(f"\n  Baseline  acc_global={acc_global:.4f}  std={np.std(acc):.4f}")
        print(f"  Min : {CIFAR100_CLASSES[np.argmin(acc)]} = {acc.min()*100:.1f}%")
        print(f"  Max : {CIFAR100_CLASSES[np.argmax(acc)]} = {acc.max()*100:.1f}%")
        worst10 = np.argsort(acc)[:10]
        print(f"  10 pires classes :")
        for c in worst10:
            print(f"    {CIFAR100_CLASSES[c]:16s} : {acc[c]*100:.1f}%")

    def _print_summary(self, acc_baseline, acc_global_0, log):
        acc_final = np.array([log[-1][n] for n in CIFAR100_CLASSES])
        acc_gf    = log[-1]["acc_global"]
        delta     = acc_final - acc_baseline

        print(f"\n{'='*60}")
        print(f"  RÉSUMÉ FINAL — CIFAR-100")
        print(f"{'='*60}")
        print(f"  Global : {acc_global_0*100:.1f}% → {acc_gf*100:.1f}%  "
              f"(Δ={( acc_gf-acc_global_0)*100:+.1f}%)")
        print(f"  Std    : {np.std(acc_baseline)*100:.2f}% → "
              f"{np.std(acc_final)*100:.2f}%  "
              f"(Δ={(np.std(acc_final)-np.std(acc_baseline))*100:+.2f}%)")

        # Top 10 améliorations
        top_improve = np.argsort(delta)[-10:][::-1]
        print(f"\n  Top 10 améliorations :")
        for c in top_improve:
            print(f"    {CIFAR100_CLASSES[c]:16s} : "
                  f"{acc_baseline[c]*100:.1f}% → {acc_final[c]*100:.1f}%  "
                  f"({delta[c]*100:+.1f}%)")

        # Top 10 dégradations
        top_degrade = np.argsort(delta)[:10]
        print(f"\n  Top 10 dégradations :")
        for c in top_degrade:
            print(f"    {CIFAR100_CLASSES[c]:16s} : "
                  f"{acc_baseline[c]*100:.1f}% → {acc_final[c]*100:.1f}%  "
                  f"({delta[c]*100:+.1f}%)")

        print(f"{'='*60}")

    def _save(self, acc_baseline, acc_global_0, log):
        out = self.cfg["output_dir"]
        pd.DataFrame(log).to_csv(os.path.join(out, "iteration_log.csv"), index=False)
        print(f"\n  CSV sauvegardé dans {out}/")

    def _plot(self, acc_baseline, log):
        out   = self.cfg["output_dir"]
        iters = [r["iteration"] for r in log]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Hessian Surgery — CIFAR-100", fontsize=13)

        # Panel 1 : distribution des accuracies (histogramme)
        ax = axes[0]
        acc_final = np.array([log[-1][n] for n in CIFAR100_CLASSES])
        ax.hist(acc_baseline * 100, bins=20, alpha=0.5, label="baseline", color="gray")
        ax.hist(acc_final * 100, bins=20, alpha=0.5, label="après surgery", color="steelblue")
        ax.set_xlabel("Accuracy (%)")
        ax.set_ylabel("Nombre de classes")
        ax.set_title("Distribution des accuracies par classe")
        ax.legend()
        ax.grid(alpha=0.3)

        # Panel 2 : std inter-classes
        ax = axes[1]
        stds = [r["std"] * 100 for r in log]
        ax.plot(iters, stds, "s-", color="crimson", lw=2, ms=5)
        ax.axhline(np.std(acc_baseline) * 100, color="crimson",
                   ls=":", lw=1, alpha=0.6, label="baseline")
        ax.set_xlabel("Itération")
        ax.set_ylabel("Std inter-classes (%)")
        ax.set_title("Variance inter-classes")
        ax.legend()
        ax.grid(alpha=0.3)

        # Panel 3 : alpha_max
        ax = axes[2]
        alphas = [r["alpha_max"] for r in log]
        ax.step(iters, alphas, where="post", color="darkorange", lw=2)
        ax.set_xlabel("Itération")
        ax.set_ylabel("α_max")
        ax.set_title("Amplitude du choc")
        ax.grid(alpha=0.3)

        plt.tight_layout()
        path = os.path.join(out, "evolution_cifar100.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Figure : {path}")


# ════════════════════════════════════════════════════════════════════════════
# Point d'entrée
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("[1] Chargement de CIFAR-100 ...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    y_train, y_test = y_train.flatten(), y_test.flatten()
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0

    rng     = np.random.default_rng(CONFIG["seed"])
    hvp_idx = rng.choice(len(x_train), CONFIG["n_hvp_samples"], replace=False)
    x_hvp   = x_train[hvp_idx]
    y_hvp   = y_train[hvp_idx]

    print("[2] Chargement du modèle ...")
    model   = tf.keras.models.load_model(CONFIG["model_path"])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    print(f"    Paramètres : "
          f"{sum(np.prod(v.shape) for v in model.trainable_variables):,}")

    optimizer = SpikeOptimizerCIFAR100(
        model=model, loss_fn=loss_fn,
        x_test=x_test, y_test=y_test,
        x_hvp=x_hvp, y_hvp=y_hvp,
        cfg=CONFIG,
    )
    optimizer.run()
