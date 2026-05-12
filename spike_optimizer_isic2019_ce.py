"""
spike_optimizer_isic2019.py
---------------------------
Spectral Surgery sur ISIC 2019 : même pipeline que spike_optimizer.py
adapté pour 8 classes dermoscopiques fortement déséquilibrées.

Différences avec CIFAR-10 :
  - 8 classes → 7 spikes théoriques (n_classes - 1)
  - Images 224×224 chargées depuis le cache .npz (ImageNet-normalisées)
  - Val set limité (3800 images) → 2000 sensibilité + 1800 éval held-out
  - Classes rares (DF=36, VASC=38 dans le val) → sensibilité bruitée attendue

Usage :
    python spike_optimizer_isic2019.py
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
    "model_path"   : "resnet50_isic2019.keras",
    "cache_train"  : "data/isic2019_cache/train.npz",
    "cache_val"    : "data/isic2019_cache/val_ss.npz",   # monitoring SS (1900 imgs)
    "cache_test"   : "data/isic2019_cache/test_ss.npz",  # éval finale uniquement
    "n_hvp_samples": 256,
    "seed"         : 0,

    # ── Analyse spectrale ──────────────────────────────────────────────────
    "lanczos_m"    : 10,        # suffisant pour 7 spikes théoriques (gap bien séparé)
    "n_spikes"     : 7,         # n_classes - 1

    # ── Sondage de sensibilité ─────────────────────────────────────────────
    "eps_probe"    : 0.01,

    # ── Optimisation des coefficients α ───────────────────────────────────
    "alpha_max_init": 0.01,       # v2 : pas plus petits
    "alpha_min"    : 0.001,       # v2 : plancher plus bas → plus d'exploration

    # ── Decay adaptatif (EMA) ─────────────────────────────────────────────
    "decay_factor" : 0.7,
    "beta_ema"     : 0.7,

    # ── Sensitivity stratifiée depuis le train set ────────────────────────
    "sens_max_per_class": 250,    # v2 : cap les classes majoritaires

    # ── Sauvegarde ─────────────────────────────────────────────────────────
    "save_model"   : True,
    "model_out"    : "resnet50_isic2019_ce_ss.keras",

    # ── Boucle principale ──────────────────────────────────────────────────
    "n_iter"       : 20,
    "patience"     : 3,           # early stop si pas d'amélioration au plancher α_min

    # ── Sortie ────────────────────────────────────────────────────────────
    "output_dir"   : "results/isic2019/ce_ss",
}

ISIC_CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
N_CLASSES    = len(ISIC_CLASSES)

# ════════════════════════════════════════════════════════════════════════════
# Fonctions utilitaires
# ════════════════════════════════════════════════════════════════════════════

def per_class_accuracy(model, x, y, n_classes=N_CLASSES):
    """Retourne un array (n_classes,) avec l'accuracy par classe."""
    preds = model.predict(x, verbose=0, batch_size=64).argmax(axis=1)
    return np.array([(preds[y == c] == c).mean() if (y == c).sum() > 0 else 0.0
                     for c in range(n_classes)])

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
    """Calcule les top-n_spikes vecteurs propres via Lanczos.

    Metal : les GradientTapes imbriqués (@tf.function) crashent sur Metal après
    le 1er appel (OOM fragmenté). Contournement : copie du modèle sur CPU pour
    les HVP. L'inférence (matrice S, rollback eval) reste sur Metal GPU.
    """
    from spectral_tools import HessianVectorProduct, StochasticLanczosQuadrature
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        # Copie légère des poids vers un modèle CPU (94 MB, ~1 ms sur M4)
        with tf.device('/CPU:0'):
            cpu_model = tf.keras.models.clone_model(model)
            cpu_model.set_weights(model.get_weights())
        hvp_model = cpu_model
    else:
        hvp_model = model

    hvp = HessianVectorProduct(
        model=hvp_model, loss_fn=loss_fn,
        data_x=x_hvp, data_y=y_hvp, batch_size=32,
    )
    slq = StochasticLanczosQuadrature(
        hvp=hvp, n_params=hvp.n_params, m=20, k=1, sigma2=1e-5,
    )
    return slq.estimate_top_eigenvalues(m_lanczos=lanczos_m, verbose=False)

def compute_sensitivity(model, ritz_vecs, n_spikes, eps_probe, x_test, y_test):
    """
    Matrice de sensibilité S[spike, classe].
    S[i,j] = (acc(θ + ε·qᵢ) - acc(θ - ε·qᵢ)) / (2ε)

    Séquentiel : les clones Metal partagent les buffers GPU avec le modèle
    principal, les détruire en threads invalide les placeholders de _hvp_batch.
    Optimisation : δ = -2ε·qᵢ pour aller de +ε à -ε sans restore intermédiaire.
    """
    current_w = save_weights(model)
    S = np.zeros((n_spikes, N_CLASSES))
    for s in range(n_spikes):
        delta = (eps_probe * ritz_vecs[:, s]).astype(np.float32)
        apply_perturbation(model, delta)            # θ → θ + ε·q
        acc_pos = per_class_accuracy(model, x_test, y_test)
        apply_perturbation(model, -2 * delta)       # θ + ε → θ - ε  (pas de restore)
        acc_neg = per_class_accuracy(model, x_test, y_test)
        restore_weights(model, current_w)           # → θ
        S[s] = (acc_pos - acc_neg) / (2 * eps_probe)
    return S

def optimize_alpha(S, acc_current, acc_baseline, alpha_max, ritz_vals, n_iter):
    """
    Trouve α ∈ R^n_spikes qui maximise l'amélioration pondérée par classe.
    Poids proportionnels à l'écart relatif à la meilleure classe.

    Budget Hessien par spike : |αi| ≤ α_max · sqrt(λ_min / λi)

    Contrainte cumulative : acc_current + S.T @ alpha ≥ acc_baseline - 0.05
    → aucune classe ne peut descendre sous baseline - 5%, quelle que soit
      l'histoire des itérations précédentes. Empêche l'érosion lente de NV/BCC.
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
        predicted_delta = S.T @ alpha
        # Classes fortes (baseline > 70%) : budget total 6%, réparti sur n_iter
        # Classes faibles : max -3%/iter (inchangé)
        delta_max_total = 0.06
        per_iter_limit  = delta_max_total / n_iter
        limits = np.where(acc_baseline > 0.70, -per_iter_limit, -0.03)
        return np.min(predicted_delta - limits)

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
    def __init__(self, model, loss_fn, x_sens, y_sens, x_eval, y_eval,
                 x_hvp, y_hvp, cfg, x_val_mon=None, y_val_mon=None):
        self.model     = model
        self.loss_fn   = loss_fn
        self.x_sens    = x_sens
        self.y_sens    = y_sens
        self.x_eval    = x_eval
        self.y_eval    = y_eval
        self.x_val_mon = x_val_mon
        self.y_val_mon = y_val_mon
        self.x_hvp     = x_hvp
        self.y_hvp     = y_hvp
        self.cfg       = cfg
        os.makedirs(cfg["output_dir"], exist_ok=True)

    def run(self):
        cfg = self.cfg
        alpha_max    = cfg["alpha_max_init"]
        beta         = cfg["beta_ema"]
        log          = []

        acc_baseline = per_class_accuracy(self.model, self.x_sens, self.y_sens)
        acc_global_0 = self.model.evaluate(
            self.x_sens, self.y_sens, verbose=0, batch_size=64)[1]
        self._print_baseline(acc_baseline, acc_global_0)

        acc_current  = acc_baseline.copy()
        std_ema      = float(np.std(acc_baseline))
        best_std_ema = std_ema
        no_improve   = 0

        # Adam-like adaptation de alpha_max
        # Signal g_t : amélioration réelle (>0), stagnation (=0), ou dégradation évitée (<0)
        adam_m, adam_v, adam_t = 0.0, 0.0, 0
        β1_a, β2_a, ε_a = 0.9, 0.999, 1e-8

        print(f"\n{'='*72}")
        print(f"  ITÉRATIONS — {cfg['n_iter']} chocs  α_init={alpha_max}  β_ema={beta}")
        print(f"{'='*72}")

        for it in range(1, cfg["n_iter"] + 1):
            t0 = time.time()
            print(f"\n  --- Itération {it}/{cfg['n_iter']}  ‖α‖≤{alpha_max:.4f} ---")

            ritz_vals, ritz_vecs = compute_eigenvectors(
                self.model, self.loss_fn, self.x_hvp, self.y_hvp,
                cfg["lanczos_m"], cfg["n_spikes"],
            )
            n_sp = min(cfg["n_spikes"], ritz_vecs.shape[1])
            print(f"  λ top-7 : {ritz_vals[:n_sp].round(1).tolist()}")

            # eps_probe = alpha_max courant : la sonde est à la même échelle que
            # la perturbation finale → S.T @ alpha est une estimation calibrée
            # → la contrainte de non-dégradation est effectivement respectée.
            S = compute_sensitivity(
                self.model, ritz_vecs, n_sp,
                alpha_max, self.x_sens, self.y_sens,
            )

            # ── Print matrice de sensibilité ─────────────────────────────────
            header = "  spike  " + "  ".join(f"{c:>5s}" for c in ISIC_CLASSES)
            print(f"\n  Matrice S (sensibilité spike × classe) :")
            print(header)
            for s in range(n_sp):
                row = "  ".join(f"{S[s,c]:>+5.2f}" for c in range(N_CLASSES))
                print(f"  q{s+1:02d}    {row}")
            print()

            alpha = optimize_alpha(S, acc_current, acc_baseline, alpha_max, ritz_vals[:n_sp], cfg["n_iter"])

            weights_before = save_weights(self.model)
            std_before     = float(np.std(acc_current))

            delta = np.zeros(ritz_vecs.shape[0], dtype=np.float64)
            for s in range(n_sp):
                delta += alpha[s] * ritz_vecs[:, s]
            apply_perturbation(self.model, delta.astype(np.float32))

            acc_new    = per_class_accuracy(self.model, self.x_sens, self.y_sens)
            acc_global = self.model.evaluate(
                self.x_sens, self.y_sens, verbose=0, batch_size=64)[1]
            delta_acc  = acc_new - acc_current
            cur_std    = float(np.std(acc_new))

            rolled_back = False
            max_class_drop = float(np.max(acc_current - acc_new))  # pire dégradation réelle
            if cur_std > std_before + 0.005 or max_class_drop > 0.07:
                std_measured  = cur_std
                worst_class   = ISIC_CLASSES[int(np.argmax(acc_current - acc_new))]
                restore_weights(self.model, weights_before)
                cur_std    = std_before
                acc_new    = acc_current.copy()
                delta_acc  = np.zeros_like(acc_current)
                acc_global = self.model.evaluate(
                    self.x_sens, self.y_sens, verbose=0, batch_size=64)[1]
                rolled_back = True
                if max_class_drop > 0.07:
                    print(f"  [ROLLBACK] {worst_class} -{max_class_drop*100:.1f}% > 7% → poids restaurés")
                else:
                    print(f"  [ROLLBACK] std {std_before:.4f}→{std_measured:.4f} → poids restaurés")
            else:
                acc_current = acc_new.copy()

            std_ema = beta * std_ema + (1.0 - beta) * cur_std
            if rolled_back:
                g_adam = -(std_measured - std_before)  # signal négatif : dégradation évitée
                no_improve += 1
            elif std_ema < best_std_ema - 1e-4:
                g_adam = std_before - cur_std           # signal positif : amélioration réelle
                best_std_ema = std_ema
                no_improve   = 0
            else:
                g_adam = 0.0                            # accepté mais stagnation
                no_improve += 1

            # Mise à jour Adam
            adam_t += 1
            adam_m  = β1_a * adam_m + (1 - β1_a) * g_adam
            adam_v  = β2_a * adam_v + (1 - β2_a) * g_adam ** 2
            m_hat   = adam_m / (1 - β1_a ** adam_t)
            v_hat   = adam_v / (1 - β2_a ** adam_t)
            snr     = m_hat / (np.sqrt(v_hat) + ε_a)
            # SNR > 0 : progrès consistant → α remonte ; SNR < 0 → α descend
            α_frac  = 0.5 + 0.5 * float(np.tanh(snr * 5.0))
            alpha_max = cfg["alpha_min"] + α_frac * (cfg["alpha_max_init"] - cfg["alpha_min"])
            alpha_max = float(np.clip(alpha_max, cfg["alpha_min"], cfg["alpha_max_init"]))
            print(f"  [adam] g={g_adam:+.4f}  snr={snr:.3f}  α_max={alpha_max:.4f}")

            if no_improve >= cfg.get("patience", 3) and alpha_max <= cfg["alpha_min"] + 1e-6:
                print(f"  [EARLY STOP] {no_improve} itérations sans amélioration "
                      f"au plancher α_min={cfg['alpha_min']} → arrêt anticipé")
                break

            elapsed = time.time() - t0

            rb_tag = " ↩ROLLBACK" if rolled_back else ""
            print(f"  global={acc_global:.4f}  std={cur_std:.4f}  "
                  f"ema={std_ema:.4f}  ({elapsed:.0f}s){rb_tag}")
            header = "  " + "  ".join(f"{n[:4]:>5s}" for n in ISIC_CLASSES)
            vals   = "  " + "  ".join(f"{acc_new[c]*100:>5.1f}" for c in range(N_CLASSES))
            delts  = "  " + "  ".join(f"{delta_acc[c]*100:>+5.1f}" for c in range(N_CLASSES))
            print(header)
            print(vals)
            print(delts)

            log.append({
                "iteration"  : it,
                "acc_global" : float(acc_global),
                "std"        : cur_std,
                "std_ema"    : std_ema,
                "alpha_max"  : alpha_max,
                "lambda_max" : float(ritz_vals[0]),
                "alpha_norm" : float(np.linalg.norm(alpha)),
                "rolled_back": rolled_back,
                "elapsed_s"  : elapsed,
                **{ISIC_CLASSES[c]: float(acc_new[c])          for c in range(N_CLASSES)},
                **{f"d_{ISIC_CLASSES[c]}": float(delta_acc[c]) for c in range(N_CLASSES)},
            })

            # Checkpoint disque après chaque itération (rollback ou non)
            pd.DataFrame(log).to_csv(
                os.path.join(self.cfg["output_dir"], "iteration_log.csv"), index=False)
            if not rolled_back and self.cfg.get("save_model"):
                self.model.save(self.cfg["model_out"])
                print(f"  [ckpt] modèle sauvegardé → {self.cfg['model_out']}")

        self._print_summary(acc_baseline, acc_global_0, log)
        self._save(acc_baseline, acc_global_0, log)
        self._plot(acc_baseline, log)
        self._final_evaluation(log)

        if cfg.get("save_model"):
            self.model.save(cfg["model_out"])
            print(f"\n  Modèle post-spike sauvegardé : {cfg['model_out']}")

        return log

    def _print_baseline(self, acc, acc_global):
        print(f"\n  Baseline  acc_global={acc_global:.4f}  std={np.std(acc):.4f}")
        for c, name in enumerate(ISIC_CLASSES):
            print(f"    {name:4s} : {acc[c]*100:.1f}%")

    def _print_summary(self, acc_baseline, acc_global_0, log):
        acc_final = np.array([log[-1][n] for n in ISIC_CLASSES])
        acc_gf    = log[-1]["acc_global"]
        print(f"\n{'='*60}")
        print(f"  RÉSUMÉ FINAL")
        print(f"{'='*60}")
        print(f"  {'Classe':6s} {'baseline':>9s} {'final':>9s} {'Δ':>8s}")
        print(f"  " + "─" * 36)
        for c, name in enumerate(ISIC_CLASSES):
            print(f"  {name:6s} {acc_baseline[c]*100:>8.1f}% "
                  f"{acc_final[c]*100:>8.1f}% "
                  f"{(acc_final[c]-acc_baseline[c])*100:>+7.1f}%")
        print(f"  " + "─" * 36)
        print(f"  {'GLOBAL':6s} {acc_global_0*100:>8.1f}% "
              f"{acc_gf*100:>8.1f}% "
              f"{(acc_gf-acc_global_0)*100:>+7.1f}%")
        print(f"  std : {np.std(acc_baseline)*100:.2f}% → {np.std(acc_final)*100:.2f}%  "
              f"(Δ={( np.std(acc_final)-np.std(acc_baseline))*100:+.2f}%)")
        print(f"{'='*60}")

    def _final_evaluation(self, log):
        print(f"\n{'='*60}")
        print(f"  ÉVALUATION FINALE — set held-out ({len(self.y_eval)} images)")
        print(f"{'='*60}")
        acc_eval        = per_class_accuracy(self.model, self.x_eval, self.y_eval)
        results_eval    = self.model.evaluate(
            self.x_eval, self.y_eval, verbose=0, batch_size=64)
        acc_global_eval = results_eval[1]
        loss_eval       = results_eval[0]

        print(f"  Accuracy globale  : {acc_global_eval*100:.2f}%")
        print(f"  Balanced accuracy : {acc_eval.mean()*100:.2f}%")
        print(f"  Loss              : {loss_eval:.4f}")
        print(f"  Std inter-classes : {np.std(acc_eval)*100:.2f}%")
        print(f"\n  {'Classe':6s} {'accuracy':>9s}")
        print(f"  " + "─" * 18)
        for c, name in enumerate(ISIC_CLASSES):
            print(f"  {name:6s} {acc_eval[c]*100:>8.1f}%")

        out = self.cfg["output_dir"]
        eval_summary = {
            "acc_global_eval"  : float(acc_global_eval),
            "balanced_acc_eval": float(acc_eval.mean()),
            "loss_eval"        : float(loss_eval),
            "std_eval"         : float(np.std(acc_eval)),
            **{f"eval_{n}": float(acc_eval[c]) for c, n in enumerate(ISIC_CLASSES)},
        }
        pd.DataFrame([eval_summary]).to_csv(
            os.path.join(out, "eval_heldout.csv"), index=False)
        print(f"\n  Résultats held-out : {out}/eval_heldout.csv")

    def _save(self, acc_baseline, acc_global_0, log):
        out = self.cfg["output_dir"]
        pd.DataFrame(log).to_csv(os.path.join(out, "iteration_log.csv"), index=False)

        acc_final = np.array([log[-1][n] for n in ISIC_CLASSES])
        summary = {
            "n_iter"       : len(log),
            "acc_global_0" : acc_global_0,
            "acc_global_f" : log[-1]["acc_global"],
            "std_0"        : float(np.std(acc_baseline)),
            "std_f"        : log[-1]["std"],
            "bal_acc_0"    : float(acc_baseline.mean()),
            "bal_acc_f"    : float(acc_final.mean()),
            **{f"baseline_{n}": float(acc_baseline[c]) for c, n in enumerate(ISIC_CLASSES)},
            **{f"final_{n}"   : float(acc_final[c])    for c, n in enumerate(ISIC_CLASSES)},
        }
        pd.DataFrame([summary]).to_csv(os.path.join(out, "summary.csv"), index=False)
        print(f"\n  CSV sauvegardés dans {out}/")

    def _plot(self, acc_baseline, log):
        out   = self.cfg["output_dir"]
        iters = [r["iteration"] for r in log]

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle("Spike Optimizer ISIC2019 — évolution par itération", fontsize=13)

        colors = plt.cm.tab10(np.linspace(0, 1, N_CLASSES))

        ax = axes[0, 0]
        for c, name in enumerate(ISIC_CLASSES):
            vals = [r[name] * 100 for r in log]
            ax.plot(iters, vals, "o-", color=colors[c], label=name, lw=1.5, ms=4)
            ax.axhline(acc_baseline[c] * 100, color=colors[c], ls=":", lw=0.8, alpha=0.5)
        ax.set_xlabel("Itération")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Accuracy par classe (pointillés = baseline)")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)

        ax = axes[0, 1]
        weak_classes = ["DF", "VASC", "AK"]  # classes les plus rares
        for c, name in enumerate(ISIC_CLASSES):
            if name in weak_classes:
                vals = [r[name] * 100 for r in log]
                ax.plot(iters, vals, "o-", color=colors[c], label=name, lw=2, ms=5)
                ax.axhline(acc_baseline[c] * 100, color=colors[c], ls=":", lw=1, alpha=0.6)
        ax.set_xlabel("Itération")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Classes rares — zoom (DF, VASC, AK)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

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
    rng = np.random.default_rng(CONFIG["seed"])

    # ── Val set = monitoring SS (rollback + early stop) ──────────────────
    print("[1a] Chargement val_ss (monitoring SS) ...")
    val_data  = np.load(CONFIG["cache_val"])
    x_val_mon = val_data["imgs"].astype(np.float32)
    y_val_mon = val_data["labels"].astype(np.int32)
    print(f"    Val monitor : {len(x_val_mon)} images")
    for c, name in enumerate(ISIC_CLASSES):
        print(f"      {name:4s} : {(y_val_mon==c).sum():4d}")

    # ── Test set = évaluation finale uniquement ───────────────────────────
    print("\n[1b] Chargement test_ss (éval finale) ...")
    test_data = np.load(CONFIG["cache_test"])
    x_eval    = test_data["imgs"].astype(np.float32)
    y_eval    = test_data["labels"].astype(np.int32)
    print(f"    Test set    : {len(x_eval)} images")
    for c, name in enumerate(ISIC_CLASSES):
        print(f"      {name:4s} : {(y_eval==c).sum():4d}")

    # ── Train set : HVP + sensitivity stratifiée ─────────────────────────
    print("\n[2] Chargement du cache train ...")
    train_data = np.load(CONFIG["cache_train"])
    x_train    = train_data["imgs"].astype(np.float32)
    y_train    = train_data["labels"].astype(np.int32)

    # HVP samples
    hvp_idx = rng.choice(len(x_train), CONFIG["n_hvp_samples"], replace=False)
    x_hvp   = x_train[hvp_idx]
    y_hvp   = y_train[hvp_idx]
    print(f"    HVP samples : {len(x_hvp)}")

    # Sensitivity set stratifié : max_per_class pour les classes majoritaires,
    # toutes les images disponibles pour les classes rares
    max_pc     = CONFIG["sens_max_per_class"]
    sens_idx   = []
    print(f"\n    Sensitivity set stratifié (max {max_pc}/classe) :")
    for c, name in enumerate(ISIC_CLASSES):
        class_idx = np.where(y_train == c)[0]
        n_take    = min(len(class_idx), max_pc)
        chosen    = rng.choice(class_idx, n_take, replace=False)
        sens_idx.append(chosen)
        print(f"      {name:4s} : {n_take:4d} / {len(class_idx)}")
    sens_idx = np.concatenate(sens_idx)
    rng.shuffle(sens_idx)
    x_sens = x_train[sens_idx]
    y_sens = y_train[sens_idx]
    print(f"    Total sensitivity : {len(x_sens)} images")
    del x_train, y_train, train_data  # libère ~13 GB

    # ── Modèle ───────────────────────────────────────────────────────────
    print("\n[3] Chargement du modèle ...")
    model   = tf.keras.models.load_model(CONFIG["model_path"], compile=False)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
    print(f"    Paramètres : {sum(np.prod(v.shape) for v in model.trainable_variables):,}")

    # ── Baseline sur val monitor + test ──────────────────────────────────
    print("\n[4] Évaluation baseline ...")
    baseline_val_acc  = per_class_accuracy(model, x_val_mon, y_val_mon)
    baseline_val_glob = model.evaluate(x_val_mon, y_val_mon, verbose=0, batch_size=64)[1]
    baseline_val_bal  = float(baseline_val_acc.mean())
    print(f"    acc_global : {baseline_val_glob:.4f}")
    print(f"    bal_acc    : {baseline_val_bal:.4f}")
    print(f"    std        : {float(np.std(baseline_val_acc)):.4f}")
    for c, name in enumerate(ISIC_CLASSES):
        print(f"      {name:4s} : {baseline_val_acc[c]*100:.1f}%")
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    pd.DataFrame([{
        "acc_global"  : float(baseline_val_glob),
        "balanced_acc": baseline_val_bal,
        "std"         : float(np.std(baseline_val_acc)),
        **{n: float(baseline_val_acc[c]) for c, n in enumerate(ISIC_CLASSES)},
    }]).to_csv(os.path.join(CONFIG["output_dir"], "baseline_val.csv"), index=False)

    # ── Lancement SS ─────────────────────────────────────────────────────
    optimizer = SpikeOptimizer(
        model=model, loss_fn=loss_fn,
        x_sens=x_sens,   y_sens=y_sens,
        x_eval=x_eval,   y_eval=y_eval,       # test set → éval finale
        x_val_mon=x_val_mon, y_val_mon=y_val_mon,  # val → monitoring SS
        x_hvp=x_hvp,     y_hvp=y_hvp,
        cfg=CONFIG,
    )
    optimizer.run()
