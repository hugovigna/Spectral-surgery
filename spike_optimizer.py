"""
spike_optimizer.py
------------------
Optimisation itérative des performances par classe via chocs composés
dans le sous-espace spike de la Hessienne.

Idée :
  À chaque itération, on mesure la sensibilité de l'accuracy par classe
  à chaque vecteur propre (spike) du Hessien, on construit une combinaison
  linéaire optimale δθ = Σ αᵢ qᵢ qui améliore les classes faibles, et on
  l'applique directement (sans fine-tuning).

  Un mécanisme de decay adaptatif réduit l'amplitude des chocs quand la
  variance inter-classes cesse de diminuer — analogue à un learning rate
  qui décroît près d'un minimum.

Usage rapide :
    python spike_optimizer.py

Paramètres configurables dans CONFIG ci-dessous.
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
# CONFIG — modifie ces valeurs pour paramétrer le run
# ════════════════════════════════════════════════════════════════════════════
CONFIG = {
    # ── Modèle et données ──────────────────────────────────────────────────
    "model_path"   : "results/cifar10/focal_loss/model_fl.keras",
    "n_hvp_samples": 128,       # images pour le calcul HVP/Lanczos
    "seed"         : 0,

    # ── Analyse spectrale ──────────────────────────────────────────────────
    "lanczos_m"    : 10,        # pas de Lanczos
    "n_spikes"     : 9,         # nombre de vecteurs propres calculés

    # ── Sondage de sensibilité ─────────────────────────────────────────────
    "eps_probe"    : 0.01,      # amplitude ±ε pour mesurer dAcc/dqᵢ

    # ── Optimisation des coefficients α ───────────────────────────────────
    "alpha_max_init": 0.02,     # amplitude initiale du choc
    "alpha_min"    : 0.002,     # plancher
    "omega_p"      : 1.0,       # exposant des poids : 0.5=sqrt, 1=linear, 2=square

    # ── Contrôle adaptatif de alpha_max (Adam sur le signal de progression) ─
    "adam_beta1"   : 0.9,       # momentum du signal moyen
    "adam_beta2"   : 0.999,     # momentum du signal carré (mémoire longue)
    "adam_eps"     : 1e-8,

    # ── Rollback ──────────────────────────────────────────────────────────
    "rollback_std_tol"  : 0.005,   # Δstd absolu max avant rollback
    "rollback_drop_tol" : 0.07,    # drop max par classe avant rollback

    # ── Sauvegarde du modèle post-spike ──────────────────────────────────────
    "save_model"   : True,
    "model_out"    : "resnet50_cifar10_fl_ss.keras",

    # ── Boucle principale ──────────────────────────────────────────────────
    "n_iter"       : 15,

    # ── Sortie ────────────────────────────────────────────────────────────
    "output_dir"   : "results/cifar10/fl_ss",
}

CIFAR10_CLASSES = [
    "avion", "auto", "oiseau", "chat", "cerf",
    "chien", "grenouille", "cheval", "bateau", "camion"
]

# ════════════════════════════════════════════════════════════════════════════
# Fonctions utilitaires
# ════════════════════════════════════════════════════════════════════════════

def per_class_accuracy(model, x, y, n_classes=10):
    """Retourne un array (n_classes,) avec l'accuracy par classe."""
    preds = model.predict(x, verbose=0, batch_size=256).argmax(axis=1)
    return np.array([(preds[y == c] == c).mean() for c in range(n_classes)])

def save_weights(model):
    return [v.numpy().copy() for v in model.variables]

def restore_weights(model, weights):
    for var, w in zip(model.variables, weights):
        var.assign(tf.constant(w, dtype=var.dtype))

def apply_perturbation(model, delta_flat):
    """Applique δθ aux trainable_variables (delta_flat est un vecteur plat)."""
    idx = 0
    for var in model.trainable_variables:
        size = np.prod(var.shape)
        var.assign_add(tf.constant(
            delta_flat[idx:idx+size].reshape(var.shape), dtype=var.dtype
        ))
        idx += size

def compute_eigenvectors(model, loss_fn, x_hvp, y_hvp, lanczos_m, n_spikes):
    """Calcule les top-n_spikes vecteurs propres via Lanczos."""
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
    """
    Matrice de sensibilité S[spike, classe].
    S[i,j] = (acc(θ + ε·qᵢ) - acc(θ - ε·qᵢ)) / (2ε)
    Interprétation : S[i,j] > 0 → aller dans la direction +qᵢ améliore la classe j.
    """
    current_w = save_weights(model)
    S = np.zeros((n_spikes, 10))
    for s in range(n_spikes):
        qi = ritz_vecs[:, s]
        restore_weights(model, current_w)
        apply_perturbation(model, (eps_probe * qi).astype(np.float32))
        acc_pos = per_class_accuracy(model, x_test, y_test)
        restore_weights(model, current_w)
        apply_perturbation(model, (-eps_probe * qi).astype(np.float32))
        acc_neg = per_class_accuracy(model, x_test, y_test)
        S[s] = (acc_pos - acc_neg) / (2 * eps_probe)
    restore_weights(model, current_w)
    return S

def optimize_alpha(S, acc_current, alpha_max, ritz_vals):
    """
    Trouve α ∈ R^n_spikes qui maximise l'amélioration pondérée par classe.

    Poids : proportionnels à l'écart relatif à la meilleure classe.
      w_j = (acc_best - acc_j) / (acc_best - acc_min)
    → w=0 pour la meilleure classe, w=1 pour la pire.
    → redistribue automatiquement la pression au fil des itérations.

    Contraintes :
      |α_i| ≤ alpha_max * sqrt(λ_min / λ_i)  : budget par spike
        (coût quadratique ≈ constant α_i² λ_i ≤ alpha_max² λ_min)
      Δacc_prédit ≥ -1%  : ne pas dégrader les classes fortes (>85%)
    """
    acc_best = acc_current.max()
    acc_min  = acc_current.min()
    weights  = (acc_best - acc_current) / max(acc_best - acc_min, 1e-12)
    weights /= weights.sum()

    lambda_ref    = float(ritz_vals.min())
    alpha_budgets = alpha_max * np.sqrt(lambda_ref / ritz_vals)

    def objective(alpha):
        return -np.dot(weights, S.T @ alpha)

    def constraint_no_degrade(alpha):
        # Δacc prédit ≥ -0.01 pour les classes fortes (>85%)
        predicted_delta = S.T @ alpha
        strong = acc_current > 0.85
        if strong.sum() == 0:
            return 1.0
        return np.min(predicted_delta[strong]) + 0.01

    result = scipy_minimize(
        objective, x0=np.zeros(len(S)),
        method='SLSQP',
        constraints=[{'type': 'ineq', 'fun': constraint_no_degrade}],
        bounds=[(-b, b) for b in alpha_budgets],
    )
    return result.x

# ════════════════════════════════════════════════════════════════════════════
# Classe principale
# ════════════════════════════════════════════════════════════════════════════

class SpikeOptimizer:
    """
    Optimiseur itératif basé sur les chocs composés dans le sous-espace spike.

    Paramètres via CONFIG (voir haut de fichier).

    x_sens / y_sens : 5000 images pour l'estimation de la sensibilité et le pilotage itératif.
    x_eval / y_eval : 5000 images réservées à l'évaluation finale (non contaminées).
    """

    def __init__(self, model, loss_fn, x_sens, y_sens, x_eval, y_eval,
                 x_hvp, y_hvp, cfg):
        self.model   = model
        self.loss_fn = loss_fn
        self.x_sens  = x_sens
        self.y_sens  = y_sens
        self.x_eval  = x_eval
        self.y_eval  = y_eval
        self.x_hvp   = x_hvp
        self.y_hvp   = y_hvp
        self.cfg     = cfg
        os.makedirs(cfg["output_dir"], exist_ok=True)

    def run(self):
        cfg = self.cfg
        alpha_max = cfg["alpha_max_init"]
        log       = []

        # EMA windows scaled to n_iter: τ1 = n_iter/4 (mean, reactive),
        # τ2 = n_iter (variance, full-horizon). Override via explicit
        # adam_beta1/adam_beta2 in cfg if needed.
        n_it = max(int(cfg["n_iter"]), 2)
        β1 = cfg.get("adam_beta1_override", 1.0 - 4.0 / n_it)
        β2 = cfg.get("adam_beta2_override", 1.0 - 1.0 / n_it)
        ε  = cfg["adam_eps"]
        adam_m, adam_v, adam_t = 0.0, 0.0, 0
        best_std = None

        # Baseline (mesuré sur le set de sensibilité — le set d'éval est réservé à la fin)
        acc_baseline = per_class_accuracy(self.model, self.x_sens, self.y_sens)
        acc_global_0 = self.model.evaluate(
            self.x_sens, self.y_sens, verbose=0, batch_size=256)[1]
        self._print_baseline(acc_baseline, acc_global_0)

        acc_current = acc_baseline.copy()
        best_std    = float(np.std(acc_baseline))
        lin_errors  = []   # historique des écarts de linéarisation

        print(f"\n{'='*72}")
        print(f"  ITÉRATIONS — {cfg['n_iter']} chocs  "
              f"α∈[{cfg['alpha_min']}, {cfg['alpha_max_init']}]  "
              f"Adam β1={β1} β2={β2}")
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
            print(f"  λ top-3 : {ritz_vals[:3].round(1).tolist()}")

            # Sensibilité (sur le set de sensibilité, pas l'éval)
            S = compute_sensitivity(
                self.model, ritz_vecs, n_sp,
                cfg["eps_probe"], self.x_sens, self.y_sens,
            )

            # Optimiser α
            alpha = optimize_alpha(S, acc_current, alpha_max, ritz_vals[:n_sp])

            # Sauvegarder poids avant choc (pour rollback éventuel)
            weights_before = save_weights(self.model)
            std_before     = float(np.std(acc_current))

            # Appliquer choc
            delta = np.zeros(ritz_vecs.shape[0], dtype=np.float64)
            for s in range(n_sp):
                delta += alpha[s] * ritz_vecs[:, s]
            apply_perturbation(self.model, delta.astype(np.float32))

            # Mesurer (sur le set de sensibilité pour le pilotage)
            acc_new    = per_class_accuracy(self.model, self.x_sens, self.y_sens)
            acc_global = self.model.evaluate(
                self.x_sens, self.y_sens, verbose=0, batch_size=256)[1]
            delta_acc  = acc_new - acc_current
            cur_std    = float(np.std(acc_new))

            # ── Écart de linéarisation (diagnostic) ──────────────────────────
            predicted_delta = S.T @ alpha          # ce que le modèle linéaire prédit
            observed_delta  = delta_acc            # ce qui s'est réellement passé
            lin_err_abs     = float(np.linalg.norm(predicted_delta - observed_delta))
            pred_norm       = float(np.linalg.norm(predicted_delta))
            lin_err_rel     = lin_err_abs / pred_norm if pred_norm > 1e-12 else 0.0
            lin_errors.append(lin_err_rel)
            trend = ""
            if len(lin_errors) >= 2:
                trend = " ↑" if lin_errors[-1] > lin_errors[-2] else " ↓"
            print(f"  [LIN] ‖pred−obs‖/‖pred‖ = {lin_err_rel:.3f}{trend}")

            # ── Rollback ──────────────────────────────────────────────────────
            max_drop    = float(np.max(acc_current - acc_new))
            rolled_back = (cur_std > std_before + cfg["rollback_std_tol"]
                           or max_drop > cfg["rollback_drop_tol"])
            if rolled_back:
                std_measured = cur_std
                restore_weights(self.model, weights_before)
                cur_std    = std_before
                acc_new    = acc_current.copy()
                delta_acc  = np.zeros_like(acc_current)
                acc_global = self.model.evaluate(
                    self.x_sens, self.y_sens, verbose=0, batch_size=256)[1]
                print(f"  [ROLLBACK] std {std_before:.4f}→{std_measured:.4f}  "
                      f"max_drop={max_drop*100:.1f}%")
            else:
                acc_current = acc_new.copy()

            # ── Adam sur alpha_max ─────────────────────────────────────────────
            if rolled_back:
                g = -max_drop
            elif cur_std < best_std - 1e-4:
                g = std_before - cur_std   # amélioration positive
                best_std = cur_std
            else:
                g = 0.0

            adam_t += 1
            adam_m  = β1 * adam_m + (1 - β1) * g
            adam_v  = β2 * adam_v + (1 - β2) * g ** 2
            snr     = (adam_m / (1 - β1**adam_t)) / (
                        np.sqrt(adam_v / (1 - β2**adam_t)) + ε)
            α_frac  = 0.5 + 0.5 * float(np.tanh(snr * 5.0))
            alpha_max = float(np.clip(
                cfg["alpha_min"] + α_frac * (cfg["alpha_max_init"] - cfg["alpha_min"]),
                cfg["alpha_min"], cfg["alpha_max_init"],
            ))
            print(f"  [Adam] SNR={snr:.2f}  α_frac={α_frac:.2f}  α_max={alpha_max:.4f}")

            elapsed = time.time() - t0

            # ── Affichage toutes classes ──────────────────────────────────────
            rb_tag = " ↩ROLLBACK" if rolled_back else ""
            print(f"  global={acc_global:.4f}  std={cur_std:.4f}  ({elapsed:.0f}s){rb_tag}")
            header = "  " + "  ".join(f"{n[:4]:>5s}" for n in CIFAR10_CLASSES)
            vals   = "  " + "  ".join(f"{acc_new[c]*100:>5.1f}" for c in range(10))
            delts  = "  " + "  ".join(
                f"{delta_acc[c]*100:>+5.1f}" for c in range(10))
            print(header)
            print(vals)
            print(delts)

            log.append({
                "iteration" : it,
                "acc_global": float(acc_global),
                "std"       : cur_std,
                "alpha_max" : alpha_max,
                "lambda_max": float(ritz_vals[0]),
                "alpha_norm": float(np.linalg.norm(alpha)),
                "rolled_back": rolled_back,
                "lin_err_rel": lin_err_rel,
                "elapsed_s" : elapsed,
                **{CIFAR10_CLASSES[c]: float(acc_new[c])         for c in range(10)},
                **{f"d_{CIFAR10_CLASSES[c]}": float(delta_acc[c]) for c in range(10)},
            })

        self._print_summary(acc_baseline, acc_global_0, log)
        self._save(acc_baseline, acc_global_0, log)
        self._plot(acc_baseline, log)

        # ── Évaluation finale sur le set held-out (non contaminé) ─────────
        self._final_evaluation(log)

        # Sauvegarde du modèle post-spike pour le bulk FT
        if cfg.get("save_model"):
            out_path = cfg["model_out"]
            self.model.save(out_path)
            print(f"\n  Modèle post-spike sauvegardé : {out_path}")

        return log

    # ── Affichage ────────────────────────────────────────────────────────────

    def _print_baseline(self, acc, acc_global):
        print(f"\n  Baseline  acc_global={acc_global:.4f}  std={np.std(acc):.4f}")
        for c, name in enumerate(CIFAR10_CLASSES):
            print(f"    {name:12s} : {acc[c]*100:.1f}%")

    def _print_summary(self, acc_baseline, acc_global_0, log):
        acc_final = np.array([log[-1][n] for n in CIFAR10_CLASSES])
        acc_gf    = log[-1]["acc_global"]
        print(f"\n{'='*60}")
        print(f"  RÉSUMÉ FINAL")
        print(f"{'='*60}")
        print(f"  {'Classe':12s} {'baseline':>9s} {'final':>9s} {'Δ':>8s}")
        print(f"  " + "─" * 42)
        for c, name in enumerate(CIFAR10_CLASSES):
            print(f"  {name:12s} {acc_baseline[c]*100:>8.1f}% "
                  f"{acc_final[c]*100:>8.1f}% "
                  f"{(acc_final[c]-acc_baseline[c])*100:>+7.1f}%")
        print(f"  " + "─" * 42)
        print(f"  {'GLOBAL':12s} {acc_global_0*100:>8.1f}% "
              f"{acc_gf*100:>8.1f}% "
              f"{(acc_gf-acc_global_0)*100:>+7.1f}%")
        print(f"  std  :  {np.std(acc_baseline)*100:.2f}% → {np.std(acc_final)*100:.2f}%  "
              f"(Δ={( np.std(acc_final)-np.std(acc_baseline))*100:+.2f}%)")
        print(f"{'='*60}")

    # ── Évaluation finale (held-out, non contaminé) ─────────────────────────

    def _final_evaluation(self, log):
        """Évalue le modèle final sur le set d'évaluation held-out (5000 images)."""
        print(f"\n{'='*60}")
        print(f"  ÉVALUATION FINALE — set held-out (5000 images, non contaminé)")
        print(f"{'='*60}")
        acc_eval = per_class_accuracy(self.model, self.x_eval, self.y_eval)
        results_eval = self.model.evaluate(
            self.x_eval, self.y_eval, verbose=0, batch_size=256)
        acc_global_eval = results_eval[1]
        loss_eval = results_eval[0]

        print(f"  Accuracy globale : {acc_global_eval*100:.2f}%")
        print(f"  Loss             : {loss_eval:.4f}")
        print(f"  Std inter-classes : {np.std(acc_eval)*100:.2f}%")
        print(f"\n  {'Classe':12s} {'accuracy':>9s}")
        print(f"  " + "─" * 24)
        for c, name in enumerate(CIFAR10_CLASSES):
            print(f"  {name:12s} {acc_eval[c]*100:>8.1f}%")

        # Sauvegarder les résultats held-out
        out = self.cfg["output_dir"]
        eval_summary = {
            "acc_global_eval": float(acc_global_eval),
            "loss_eval": float(loss_eval),
            "std_eval": float(np.std(acc_eval)),
            **{f"eval_{n}": float(acc_eval[c])
               for c, n in enumerate(CIFAR10_CLASSES)},
        }
        pd.DataFrame([eval_summary]).to_csv(
            os.path.join(out, "eval_heldout.csv"), index=False)
        print(f"\n  Résultats held-out sauvegardés : {out}/eval_heldout.csv")

    # ── Sauvegarde ───────────────────────────────────────────────────────────

    def _save(self, acc_baseline, acc_global_0, log):
        out = self.cfg["output_dir"]
        df = pd.DataFrame(log)
        df.to_csv(os.path.join(out, "iteration_log.csv"), index=False)

        acc_final = np.array([log[-1][n] for n in CIFAR10_CLASSES])
        summary = {
            "n_iter"       : len(log),
            "acc_global_0" : acc_global_0,
            "acc_global_f" : log[-1]["acc_global"],
            "std_0"        : float(np.std(acc_baseline)),
            "std_f"        : log[-1]["std"],
            **{f"baseline_{n}": float(acc_baseline[c])
               for c, n in enumerate(CIFAR10_CLASSES)},
            **{f"final_{n}": float(acc_final[c])
               for c, n in enumerate(CIFAR10_CLASSES)},
        }
        pd.DataFrame([summary]).to_csv(
            os.path.join(out, "summary.csv"), index=False)
        print(f"\n  CSV sauvegardés dans {out}/")

    # ── Plots ────────────────────────────────────────────────────────────────

    def _plot(self, acc_baseline, log):
        out  = self.cfg["output_dir"]
        iters = [r["iteration"] for r in log]

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle("Spike Optimizer — évolution par itération", fontsize=13)

        # ── Panel 1 : accuracy par classe ────────────────────────────────────
        ax = axes[0, 0]
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        for c, name in enumerate(CIFAR10_CLASSES):
            vals = [r[name] * 100 for r in log]
            ax.plot(iters, vals, "o-", color=colors[c], label=name, lw=1.5, ms=4)
            ax.axhline(acc_baseline[c] * 100, color=colors[c],
                       ls=":", lw=0.8, alpha=0.5)
        ax.set_xlabel("Itération")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Accuracy par classe (pointillés = baseline)")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)

        # ── Panel 2 : classes faibles zoom ───────────────────────────────────
        ax = axes[0, 1]
        weak = ["chat", "chien", "oiseau"]
        for c, name in enumerate(CIFAR10_CLASSES):
            if name in weak:
                vals = [r[name] * 100 for r in log]
                ax.plot(iters, vals, "o-", color=colors[c], label=name, lw=2, ms=5)
                ax.axhline(acc_baseline[c] * 100, color=colors[c],
                           ls=":", lw=1, alpha=0.6)
        ax.set_xlabel("Itération")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Classes faibles (zoom)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # ── Panel 3 : std inter-classes ───────────────────────────────────────
        ax = axes[1, 0]
        stds = [r["std"] * 100 for r in log]
        ax.plot(iters, stds, "s-", color="crimson", lw=2, ms=5)
        ax.axhline(np.std(acc_baseline) * 100, color="crimson",
                   ls=":", lw=1, alpha=0.6, label="baseline std")
        ax.set_xlabel("Itération")
        ax.set_ylabel("Std inter-classes (%)")
        ax.set_title("Variance inter-classes (↓ = plus équilibré)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # ── Panel 4 : alpha_max adaptatif ────────────────────────────────────
        ax = axes[1, 1]
        alphas = [r["alpha_max"] for r in log]
        ax.step(iters, alphas, where="post", color="darkorange", lw=2)
        ax.set_xlabel("Itération")
        ax.set_ylabel("α_max")
        ax.set_title("Amplitude du choc (decay adaptatif)")
        ax.grid(alpha=0.3)

        plt.tight_layout()
        path = os.path.join(out, "evolution.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Figure sauvegardée : {path}")


# ════════════════════════════════════════════════════════════════════════════
# Point d'entrée
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("[1] Chargement de CIFAR-10 ...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train, y_test = y_train.flatten(), y_test.flatten()
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0

    # ── Split test set : 5000 sensibilité + 5000 évaluation held-out ──────
    # Shuffle reproductible pour garantir un split stratifié
    rng = np.random.default_rng(CONFIG["seed"])
    idx_test = rng.permutation(len(x_test))
    x_sens, y_sens = x_test[idx_test[:5000]], y_test[idx_test[:5000]]
    x_eval, y_eval = x_test[idx_test[5000:]], y_test[idx_test[5000:]]
    print(f"    Test split : {len(x_sens)} sensibilité + {len(x_eval)} évaluation held-out")

    # ── Sous-ensemble HVP (depuis le train set) ──────────────────────────
    hvp_idx = rng.choice(len(x_train), CONFIG["n_hvp_samples"], replace=False)
    x_hvp   = x_train[hvp_idx]
    y_hvp   = y_train[hvp_idx]

    print("[2] Chargement du modèle ...")
    model   = tf.keras.models.load_model(CONFIG["model_path"], compile=False)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
    print(f"    Paramètres : "
          f"{sum(np.prod(v.shape) for v in model.trainable_variables):,}")

    optimizer = SpikeOptimizer(
        model=model, loss_fn=loss_fn,
        x_sens=x_sens, y_sens=y_sens,
        x_eval=x_eval, y_eval=y_eval,
        x_hvp=x_hvp, y_hvp=y_hvp,
        cfg=CONFIG,
    )
    optimizer.run()
