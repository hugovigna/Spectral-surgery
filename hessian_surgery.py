"""
hessian_surgery.py
-------------------
Runner Hessian Surgery configurable — générique pour tout modèle Keras.

Hyperparamètres principaux :
  n_iter           : nombre max d'itérations
  max_degrade_total: dégradation cumulée max autorisée par classe (ex. 0.06 = 6%)
  omega_mode       : distribution des poids de l'objectif selon l'accuracy courante
                     "homogeneous" : ω_c = 1/C             (toutes classes égales)
                     "linear"      : ω_c ∝ (1 - acc_c)     (priorité proportionnelle)
                     "square"      : ω_c ∝ (1 - acc_c)²    (accent fort sur classes faibles)
                     "sqrt"        : ω_c ∝ √(1 - acc_c)    (priorité lissée)

Usage :
    python3.12 -u hessian_surgery.py 2>&1 | tee results/my_run/log.txt
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

from spectral_tools import HessianVectorProduct, StochasticLanczosQuadrature

# ════════════════════════════════════════════════════════════════════════════
# CONFIG — tous les hyperparamètres ici
# ════════════════════════════════════════════════════════════════════════════

CONFIG = {
    # ── Modèle et données ─────────────────────────────────────────────────
    "model_path"        : "results/isic2019/focal_loss/model_focal.keras",
    "cache_sens"        : "data/isic2019_cache/train.npz",   # sensitivity (train stratifié)
    "cache_val"         : "data/isic2019_cache/val_ss.npz",  # monitoring val (jamais utilisé pour décisions)
    "cache_test"        : "data/isic2019_cache/test_ss.npz", # éval finale uniquement
    "class_names"       : ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"],
    "sens_max_per_class": 250,    # cap du sensitivity set par classe
    "n_hvp_samples"     : 64,

    # ── Spectral ──────────────────────────────────────────────────────────
    "lanczos_m"         : 10,
    "n_spikes"          : 7,      # n_classes - 1

    # ── Boucle principale ─────────────────────────────────────────────────
    "n_iter"            : 10,
    "patience"          : 3,      # early stop si stagnation à alpha_min

    # ── Objectif omega ────────────────────────────────────────────────────
    "omega_mode"        : "linear",   # homogeneous | linear | square | sqrt

    # ── Contrainte de non-dégradation ────────────────────────────────────
    "max_degrade_total" : 0.06,   # dégradation cumulée max par classe sur tout le run
    "max_degrade_iter"  : 0.03,   # dégradation max par itération pour classes faibles

    # ── Amplitude des chocs ───────────────────────────────────────────────
    "alpha_max_init"    : 0.01,
    "alpha_min"         : 0.001,
    "beta_ema"          : 0.7,    # EMA pour le suivi de std

    # ── Sauvegarde ────────────────────────────────────────────────────────
    "output_dir"        : "results/isic2019/hessian_surgery",
    "model_out"         : "resnet50_isic2019_ss.keras",
    "save_model"        : True,
    "seed"              : 0,
}

# ════════════════════════════════════════════════════════════════════════════
# Utilitaires
# ════════════════════════════════════════════════════════════════════════════

def per_class_accuracy(model, x, y, n_classes=None):
    if n_classes is None:
        n_classes = len(CONFIG["class_names"])
    preds = model.predict(x, verbose=0, batch_size=64).argmax(axis=1)
    return np.array([(preds[y == c] == c).mean() if (y == c).sum() > 0 else 0.0
                     for c in range(n_classes)])

def save_weights(model):
    return [v.numpy().copy() for v in model.variables]

def restore_weights(model, weights):
    for var, w in zip(model.variables, weights):
        var.assign(tf.constant(w, dtype=var.dtype))

def apply_perturbation(model, delta_flat):
    idx = 0
    for var in model.trainable_variables:
        size = int(np.prod(var.shape))
        var.assign_add(tf.constant(
            delta_flat[idx:idx+size].reshape(var.shape), dtype=var.dtype
        ))
        idx += size

def progress_bar(it, n_iter, bal_acc, std, alpha_max, status, width=20):
    """Barre de progression — compatible terminal et Jupyter."""
    frac   = it / n_iter
    filled = int(frac * width)
    bar    = "█" * filled + "░" * (width - filled)
    line   = (f"[Iter {it:>2d}/{n_iter}] {bar} {frac*100:4.0f}%  "
              f"bal={bal_acc*100:.1f}%  std={std*100:.1f}%  α={alpha_max:.4f}  {status}")
    try:
        from IPython.display import clear_output
        clear_output(wait=True)
        print(line, flush=True)
    except ImportError:
        print(f"\r{line}", end="", flush=True)

# ════════════════════════════════════════════════════════════════════════════
# Omega — distribution des poids de l'objectif
# ════════════════════════════════════════════════════════════════════════════

def compute_omega(acc_current, mode):
    """
    Calcule les poids ω_c ∈ [0,1] (sum=1) selon le mode choisi.
    Les classes faibles (acc basse → erreur haute) reçoivent un poids plus élevé.

    homogeneous : priorité égale — équivalent à balanced accuracy classique
    linear      : ω ∝ (1 - acc)     — priorité proportionnelle à l'erreur
    square      : ω ∝ (1 - acc)²    — accent fort sur les classes les plus faibles
    sqrt        : ω ∝ √(1 - acc)    — priorité lissée, moins agressive que linear
    """
    e = 1.0 - np.array(acc_current, dtype=np.float64)   # taux d'erreur par classe
    e = np.maximum(e, 0.0)

    if mode == "homogeneous":
        w = np.ones(len(e))
    elif mode == "linear":
        w = e
    elif mode == "square":
        w = e ** 2
    elif mode == "sqrt":
        w = np.sqrt(e)
    else:
        raise ValueError(f"omega_mode inconnu : {mode!r}. "
                         "Choix : homogeneous | linear | square | sqrt")

    s = w.sum()
    return w / s if s > 1e-12 else np.ones(len(e)) / len(e)

# ════════════════════════════════════════════════════════════════════════════
# Analyse spectrale
# ════════════════════════════════════════════════════════════════════════════

def compute_eigenvectors(model, loss_fn, x_hvp, y_hvp, lanczos_m, n_spikes):
    """Top-n_spikes vecteurs propres via Lanczos. HVP sur CPU pour éviter les
    crashs Metal avec GradientTapes imbriqués (@tf.function OOM fragmenté)."""
    if tf.config.list_physical_devices('GPU'):
        with tf.device('/CPU:0'):
            cpu_model = tf.keras.models.clone_model(model)
            cpu_model.set_weights(model.get_weights())
        hvp_model = cpu_model
    else:
        hvp_model = model

    hvp = HessianVectorProduct(
        model=hvp_model, loss_fn=loss_fn,
        data_x=x_hvp, data_y=y_hvp, batch_size=None,
    )
    slq = StochasticLanczosQuadrature(
        hvp=hvp, n_params=hvp.n_params, m=20, k=1, sigma2=1e-5,
    )
    return slq.estimate_top_eigenvalues(m_lanczos=lanczos_m, verbose=False)

def compute_sensitivity(model, ritz_vecs, n_spikes, eps_probe, x_sens, y_sens,
                        n_classes=None):
    """S[spike, classe] = sensibilité directionnelle par différence finie centrée."""
    if n_classes is None:
        n_classes = len(CONFIG["class_names"])
    current_w = save_weights(model)
    S = np.zeros((n_spikes, n_classes))
    for s in range(n_spikes):
        delta = (eps_probe * ritz_vecs[:, s]).astype(np.float32)
        apply_perturbation(model, delta)
        acc_pos = per_class_accuracy(model, x_sens, y_sens, n_classes)
        apply_perturbation(model, -2 * delta)
        acc_neg = per_class_accuracy(model, x_sens, y_sens, n_classes)
        restore_weights(model, current_w)
        S[s] = (acc_pos - acc_neg) / (2 * eps_probe)
    return S

# ════════════════════════════════════════════════════════════════════════════
# Optimisation des coefficients α
# ════════════════════════════════════════════════════════════════════════════

def optimize_alpha(S, acc_current, acc_baseline, alpha_max, ritz_vals,
                   n_iter, max_degrade_total, max_degrade_iter, omega_mode,
                   per_spike_budget=True):
    """
    Maximise ω · (S^T α) sous contrainte de non-dégradation.

    ω = compute_omega(acc_current, omega_mode)
    per_spike_budget=True  : |αᵢ| ≤ alpha_max · √(λ_min / λᵢ)  (bounds par spike)
    per_spike_budget=False : ‖α‖ ≤ alpha_max                    (contrainte L2 globale)
    Contrainte : acc_current + S^T α ≥ acc_baseline - max_degrade_total
    """
    omega        = compute_omega(acc_current, omega_mode)
    per_iter_lim = max_degrade_total / n_iter

    def objective(alpha):
        return -np.dot(omega, S.T @ alpha)

    def constraint_no_degrade(alpha):
        delta  = S.T @ alpha
        limits = np.where(acc_baseline > 0.70, -per_iter_lim, -max_degrade_iter)
        return np.min(delta - limits)

    if per_spike_budget:
        lambda_ref    = float(ritz_vals.min())
        alpha_budgets = alpha_max * np.sqrt(lambda_ref / ritz_vals)
        bounds = [(-b, b) for b in alpha_budgets]
        constraints = [{"type": "ineq", "fun": constraint_no_degrade}]
    else:
        bounds = None
        def constraint_norm(alpha):
            return alpha_max - np.linalg.norm(alpha)
        constraints = [
            {"type": "ineq", "fun": constraint_no_degrade},
            {"type": "ineq", "fun": constraint_norm},
        ]

    result = scipy_minimize(
        objective, x0=np.zeros(len(S)),
        method="SLSQP",
        constraints=constraints,
        bounds=bounds,
    )
    return result.x

# ════════════════════════════════════════════════════════════════════════════
# Runner principal
# ════════════════════════════════════════════════════════════════════════════

class HessianSurgery:
    def __init__(self, model, loss_fn, x_sens, y_sens, x_val, y_val,
                 x_test, y_test, x_hvp, y_hvp, cfg):
        self.model   = model
        self.loss_fn = loss_fn
        self.x_sens  = x_sens;  self.y_sens  = y_sens
        self.x_val   = x_val;   self.y_val   = y_val
        self.x_test  = x_test;  self.y_test  = y_test
        self.x_hvp   = x_hvp;   self.y_hvp   = y_hvp
        self.cfg     = cfg
        self.classes = cfg["class_names"]
        self.C       = len(self.classes)
        os.makedirs(cfg["output_dir"], exist_ok=True)

    def run(self,
            n_iter            = None,
            omega_mode        = None,
            max_degrade_total = None,
            max_degrade_iter  = None,
            rollback_std_tol  = None,   # seuil rollback sur Δstd (défaut 0.005)
            rollback_drop_tol = None,   # seuil rollback sur max drop/classe (défaut 0.07)
            alpha_max_init    = None,
            alpha_min         = None,
            patience          = None,
            lanczos_m         = None,
            n_spikes          = None,
            output_dir        = None,
            model_out         = None,
            save_model        = None,
        ):
        """
        Lance la Hessian Surgery. Les kwargs surchargent CONFIG sans le modifier.

        Exemple Jupyter :
            runner.run(n_iter=15, omega_mode="sqrt", max_degrade_total=0.04,
                       rollback_std_tol=0.003, rollback_drop_tol=0.05)
        """
        cfg = {**self.cfg}   # copie locale — self.cfg inchangé
        overrides = dict(
            n_iter=n_iter, omega_mode=omega_mode,
            max_degrade_total=max_degrade_total, max_degrade_iter=max_degrade_iter,
            rollback_std_tol=rollback_std_tol, rollback_drop_tol=rollback_drop_tol,
            alpha_max_init=alpha_max_init, alpha_min=alpha_min,
            patience=patience, lanczos_m=lanczos_m, n_spikes=n_spikes,
            output_dir=output_dir, model_out=model_out, save_model=save_model,
        )
        cfg.update({k: v for k, v in overrides.items() if v is not None})
        cfg.setdefault("rollback_std_tol",  0.005)
        cfg.setdefault("rollback_drop_tol", 0.07)
        os.makedirs(cfg["output_dir"], exist_ok=True)

        alpha_max = cfg["alpha_max_init"]
        beta      = cfg["beta_ema"]
        log_sens  = []   # métriques sur sensitivity set (train stratifié)
        log_val   = []   # métriques sur val set (monitoring propre)

        # ── Baseline ─────────────────────────────────────────────────────
        n_cls = len(cfg["class_names"])
        acc_baseline  = per_class_accuracy(self.model, self.x_sens, self.y_sens, n_cls)
        acc_val_base  = per_class_accuracy(self.model, self.x_val,  self.y_val,  n_cls)
        self._print_baseline(acc_baseline, acc_val_base)

        acc_current = acc_baseline.copy()
        std_ema     = float(np.std(acc_baseline))
        best_std    = std_ema
        no_improve  = 0

        # Adam adaptatif sur alpha_max
        adam_m, adam_v, adam_t = 0.0, 0.0, 0
        β1, β2, ε = 0.9, 0.999, 1e-8

        print(f"\n{'='*72}")
        print(f"  SS — {cfg['n_iter']} iters  "
              f"omega={cfg['omega_mode']}  "
              f"degrade_max={cfg['max_degrade_total']*100:.0f}%  "
              f"α∈[{cfg['alpha_min']},{cfg['alpha_max_init']}]")
        print(f"{'='*72}\n")

        for it in range(1, cfg["n_iter"] + 1):
            t0 = time.time()

            # ── Spectre ──────────────────────────────────────────────────
            ritz_vals, ritz_vecs = compute_eigenvectors(
                self.model, self.loss_fn, self.x_hvp, self.y_hvp,
                cfg["lanczos_m"], cfg["n_spikes"],
            )
            n_sp = min(cfg["n_spikes"], ritz_vecs.shape[1])

            # ── Sensibilité ───────────────────────────────────────────────
            S = compute_sensitivity(
                self.model, ritz_vecs, n_sp,
                alpha_max, self.x_sens, self.y_sens, n_cls,
            )
            # ── Optimisation α ────────────────────────────────────────────
            alpha = optimize_alpha(
                S, acc_current, acc_baseline, alpha_max,
                ritz_vals[:n_sp], cfg["n_iter"],
                cfg["max_degrade_total"], cfg["max_degrade_iter"],
                cfg["omega_mode"],
                per_spike_budget=cfg.get("per_spike_budget", True),
            )

            # ── Perturbation ──────────────────────────────────────────────
            weights_before = save_weights(self.model)
            std_before     = float(np.std(acc_current))
            delta = sum(alpha[s] * ritz_vecs[:, s] for s in range(n_sp))
            apply_perturbation(self.model, delta.astype(np.float32))

            # ── Évaluation post-perturbation ──────────────────────────────
            acc_new     = per_class_accuracy(self.model, self.x_sens, self.y_sens, n_cls)
            acc_val_new = per_class_accuracy(self.model, self.x_val,  self.y_val,  n_cls)
            delta_acc   = acc_new - acc_current
            cur_std     = float(np.std(acc_new))

            # ── Rollback ──────────────────────────────────────────────────
            max_drop  = float(np.max(acc_current - acc_new))
            rolled_back = (cur_std > std_before + cfg["rollback_std_tol"]
                           or max_drop > cfg["rollback_drop_tol"])
            if rolled_back:
                restore_weights(self.model, weights_before)
                acc_new     = acc_current.copy()
                acc_val_new = per_class_accuracy(self.model, self.x_val, self.y_val, n_cls)
                delta_acc   = np.zeros(self.C)
                cur_std     = std_before
                status      = "ROLLBACK"
            else:
                acc_current = acc_new.copy()
                status      = "ACCEPT"

            # ── Adam sur alpha_max ────────────────────────────────────────
            std_ema = beta * std_ema + (1 - beta) * cur_std
            if rolled_back:
                g = -(max_drop)
                no_improve += 1
            elif std_ema < best_std - 1e-4:
                g = std_before - cur_std
                best_std   = std_ema
                no_improve = 0
            else:
                g = 0.0
                no_improve += 1

            adam_t += 1
            adam_m  = β1 * adam_m + (1 - β1) * g
            adam_v  = β2 * adam_v + (1 - β2) * g ** 2
            snr     = (adam_m / (1 - β1**adam_t)) / (np.sqrt(adam_v / (1 - β2**adam_t)) + ε)
            α_frac  = 0.5 + 0.5 * float(np.tanh(snr * 5.0))
            alpha_max = float(np.clip(
                cfg["alpha_min"] + α_frac * (cfg["alpha_max_init"] - cfg["alpha_min"]),
                cfg["alpha_min"], cfg["alpha_max_init"]
            ))

            elapsed = time.time() - t0

            # ── Progress bar ──────────────────────────────────────────────
            progress_bar(it, cfg["n_iter"],
                         bal_acc=acc_new.mean(), std=cur_std,
                         alpha_max=alpha_max, status=status)
            print()  # newline après la barre

            # ── Détail par classe ─────────────────────────────────────────
            header = "  " + "  ".join(f"{n[:4]:>5s}" for n in self.classes)
            sens_v = "  " + "  ".join(f"{acc_new[c]*100:>5.1f}" for c in range(self.C))
            val_v  = "  " + "  ".join(f"{acc_val_new[c]*100:>5.1f}" for c in range(self.C))
            delt_v = "  " + "  ".join(f"{delta_acc[c]*100:>+5.1f}" for c in range(self.C))
            print(f"  {'sens':6s}{sens_v}")
            print(f"  {'val':6s}{val_v}")
            print(f"  {'Δ':6s}{delt_v}")
            print(f"  ‖α‖={np.linalg.norm(alpha):.4f}  ({elapsed:.0f}s)")

            # ── Log ───────────────────────────────────────────────────────
            base_row = {
                "iteration"  : it,
                "bal_acc"    : float(acc_new.mean()),
                "std"        : cur_std,
                "std_ema"    : std_ema,
                "alpha_max"  : alpha_max,
                "lambda_max" : float(ritz_vals[0]),
                "rolled_back": rolled_back,
                "elapsed_s"  : round(elapsed, 1),
                "omega_mode" : cfg["omega_mode"],
                **{n: float(acc_new[c])      for c, n in enumerate(self.classes)},
                **{f"d_{n}": float(delta_acc[c]) for c, n in enumerate(self.classes)},
            }
            log_sens.append(base_row)
            log_val.append({**base_row,
                            **{n: float(acc_val_new[c]) for c, n in enumerate(self.classes)},
                            "bal_acc": float(acc_val_new.mean()),
                            "std"    : float(np.std(acc_val_new))})

            # Checkpoint CSV après chaque iter
            pd.DataFrame(log_sens).to_csv(
                os.path.join(cfg["output_dir"], "log_sens.csv"), index=False)
            pd.DataFrame(log_val).to_csv(
                os.path.join(cfg["output_dir"], "log_val.csv"), index=False)
            if not rolled_back and cfg.get("save_model"):
                self.model.save(cfg["model_out"])

            # ── Early stop ────────────────────────────────────────────────
            if no_improve >= cfg.get("patience", 3) and alpha_max <= cfg["alpha_min"] + 1e-6:
                print(f"\n  [EARLY STOP] {no_improve} iters sans amélioration → arrêt")
                break

        self._final_eval(cfg)
        self._plot(acc_baseline, acc_val_base, log_sens, log_val, cfg)
        if cfg.get("save_model"):
            self.model.save(cfg["model_out"])
            print(f"\n  Modèle sauvegardé → {cfg['model_out']}")
        return log_sens, log_val

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _print_baseline(self, acc_sens, acc_val):
        print(f"\n  {'Classe':6s}  {'sens':>7s}  {'val':>7s}")
        print("  " + "─" * 24)
        for c, n in enumerate(self.classes):
            print(f"  {n:6s}  {acc_sens[c]*100:>6.1f}%  {acc_val[c]*100:>6.1f}%")
        print("  " + "─" * 24)
        print(f"  {'bal':6s}  {acc_sens.mean()*100:>6.1f}%  {acc_val.mean()*100:>6.1f}%")
        print(f"  {'std':6s}  {np.std(acc_sens)*100:>6.1f}%  {np.std(acc_val)*100:>6.1f}%\n")

    def _print_sensitivity(self, S, n_sp):
        print(f"\n  Matrice S  (omega={self.cfg['omega_mode']}):")
        header = "         " + "  ".join(f"{n[:4]:>5s}" for n in self.classes)
        print(header)
        for s in range(n_sp):
            row = "  ".join(f"{S[s,c]:>+5.2f}" for c in range(self.C))
            print(f"  q{s+1:02d}    {row}")

    def _final_eval(self, cfg):
        acc  = per_class_accuracy(self.model, self.x_test, self.y_test, self.C)
        print(f"\n{'='*60}  ÉVAL FINALE (test held-out, {len(self.y_test)} imgs)")
        print(f"  bal_acc={acc.mean()*100:.2f}%  std={np.std(acc)*100:.2f}%")
        for c, n in enumerate(self.classes):
            print(f"  {n:6s} {acc[c]*100:.1f}%")
        pd.DataFrame([{
            "bal_acc": float(acc.mean()), "std": float(np.std(acc)),
            **{n: float(acc[c]) for c, n in enumerate(self.classes)},
        }]).to_csv(os.path.join(cfg["output_dir"], "eval_test.csv"), index=False)

    def _plot(self, acc_sens_base, acc_val_base, log_sens, log_val, cfg):
        out    = cfg["output_dir"]
        iters  = [r["iteration"] for r in log_sens]
        colors = plt.cm.tab10(np.linspace(0, 1, self.C))

        fig, axes = plt.subplots(2, 3, figsize=(18, 9))
        fig.suptitle(
            f"Hessian Surgery — omega={self.cfg['omega_mode']}  "
            f"degrade_max={self.cfg['max_degrade_total']*100:.0f}%",
            fontsize=13, fontweight="bold"
        )

        # [0,0] Accuracy par classe — sens set
        ax = axes[0, 0]
        for c, n in enumerate(self.classes):
            ax.plot(iters, [r[n]*100 for r in log_sens], "o-",
                    color=colors[c], label=n, lw=1.5, ms=3)
            ax.axhline(acc_sens_base[c]*100, color=colors[c], ls=":", lw=0.8, alpha=0.4)
        ax.set_title("Accuracy / classe (sensitivity set)"); ax.set_xlabel("Iter")
        ax.set_ylabel("%"); ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

        # [0,1] Accuracy par classe — val set
        ax = axes[0, 1]
        for c, n in enumerate(self.classes):
            ax.plot(iters, [r[n]*100 for r in log_val], "o-",
                    color=colors[c], label=n, lw=1.5, ms=3)
            ax.axhline(acc_val_base[c]*100, color=colors[c], ls=":", lw=0.8, alpha=0.4)
        ax.set_title("Accuracy / classe (val set)"); ax.set_xlabel("Iter")
        ax.set_ylabel("%"); ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

        # [0,2] bal_acc sens vs val
        ax = axes[0, 2]
        ax.plot(iters, [r["bal_acc"]*100 for r in log_sens], "o-",
                color="steelblue", label="sens", lw=2, ms=4)
        ax.plot(iters, [r["bal_acc"]*100 for r in log_val], "s--",
                color="darkorange", label="val", lw=2, ms=4)
        ax.axhline(acc_sens_base.mean()*100, color="steelblue", ls=":", alpha=0.5)
        ax.axhline(acc_val_base.mean()*100,  color="darkorange", ls=":", alpha=0.5)
        ax.set_title("Balanced accuracy sens vs val"); ax.set_xlabel("Iter")
        ax.set_ylabel("%"); ax.legend(); ax.grid(alpha=0.3)

        # [1,0] Std inter-classes
        ax = axes[1, 0]
        ax.plot(iters, [r["std"]*100 for r in log_sens], "o-",
                color="steelblue", label="sens", lw=2, ms=4)
        ax.plot(iters, [r["std"]*100 for r in log_val], "s--",
                color="darkorange", label="val", lw=2, ms=4)
        ax.axhline(np.std(acc_sens_base)*100, color="steelblue", ls=":", alpha=0.5)
        ax.axhline(np.std(acc_val_base)*100,  color="darkorange", ls=":", alpha=0.5)
        ax.set_title("Std inter-classes (↓ = mieux)"); ax.set_xlabel("Iter")
        ax.set_ylabel("%"); ax.legend(); ax.grid(alpha=0.3)

        # [1,1] Δ par classe (delta cumulé)
        ax = axes[1, 1]
        for c, n in enumerate(self.classes):
            cum = np.cumsum([r[f"d_{n}"]*100 for r in log_sens])
            ax.plot(iters, cum, "o-", color=colors[c], label=n, lw=1.5, ms=3)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title("Δ cumulé par classe (sens set)"); ax.set_xlabel("Iter")
        ax.set_ylabel("Δ acc (pp)"); ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

        # [1,2] alpha_max + rollbacks
        ax = axes[1, 2]
        ax.step(iters, [r["alpha_max"] for r in log_sens],
                where="post", color="darkorange", lw=2, label="α_max")
        rb_iters = [r["iteration"] for r in log_sens if r["rolled_back"]]
        if rb_iters:
            rb_alpha = [r["alpha_max"] for r in log_sens if r["rolled_back"]]
            ax.scatter(rb_iters, rb_alpha, color="crimson", zorder=5,
                       s=60, label="rollback", marker="x")
        ax.set_title("Amplitude α (decay adaptatif)"); ax.set_xlabel("Iter")
        ax.legend(); ax.grid(alpha=0.3)

        plt.tight_layout()
        path = os.path.join(out, "evolution.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Plot → {path}")


# ════════════════════════════════════════════════════════════════════════════
# Point d'entrée
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cfg = CONFIG
    rng = np.random.default_rng(cfg["seed"])
    os.makedirs(cfg["output_dir"], exist_ok=True)

    # ── Sensitivity set (train stratifié) ────────────────────────────────
    print("[1] Chargement données ...")
    train_data = np.load(cfg["cache_sens"])
    x_tr, y_tr = train_data["imgs"].astype(np.float32), train_data["labels"].astype(np.int32)
    cap = cfg["sens_max_per_class"]
    idx = np.concatenate([
        rng.choice(np.where(y_tr == c)[0],
                   min(cap, (y_tr == c).sum()), replace=False)
        for c in range(len(cfg["class_names"]))
    ])
    x_sens, y_sens = x_tr[idx], y_tr[idx]
    print(f"    Sensitivity set : {len(x_sens)} imgs  "
          f"({dict(zip(cfg['class_names'], [(y_sens==c).sum() for c in range(len(cfg['class_names']))]))})")

    val_data  = np.load(cfg["cache_val"])
    x_val, y_val = val_data["imgs"].astype(np.float32), val_data["labels"].astype(np.int32)

    test_data = np.load(cfg["cache_test"])
    x_test, y_test = test_data["imgs"].astype(np.float32), test_data["labels"].astype(np.int32)

    hvp_idx = rng.choice(len(x_tr), cfg["n_hvp_samples"], replace=False)
    x_hvp, y_hvp = x_tr[hvp_idx].astype(np.float32), y_tr[hvp_idx].astype(np.int32)
    print(f"    HVP batch : {len(x_hvp)} imgs  |  Val : {len(x_val)}  |  Test : {len(x_test)}")

    # ── Modèle ───────────────────────────────────────────────────────────
    print("\n[2] Chargement modèle ...")
    model   = tf.keras.models.load_model(cfg["model_path"], compile=False)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
    print(f"    {cfg['model_path']}  —  {sum(np.prod(v.shape) for v in model.trainable_variables):,} params")

    # ── Run ───────────────────────────────────────────────────────────────
    print(f"\n[3] Hessian Surgery  (omega={cfg['omega_mode']}) ...")
    runner = HessianSurgery(
        model, loss_fn,
        x_sens, y_sens,
        x_val, y_val,
        x_test, y_test,
        x_hvp, y_hvp,
        cfg,
    )
    runner.run()
