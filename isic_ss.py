"""
isic_ss.py
----------
Hessian Surgery — variante ISIC-2019 (8 classes dermoscopiques déséquilibrées).

Algorithme partagé entre les 3 expériences de l'article (CE, FL+SS, CB+SS).
Diffère du `HessianSurgery` canonique (CIFAR-10, cf. hessian_surgery.py) sur :
  - Pondération ω : proportionnelle à l'écart à la meilleure classe (au lieu
    de compute_omega paramétrique par omega_mode).
  - Contrainte de non-dégradation : seuil dur -3% pour classes faibles
    (acc baseline < 70%) ; budget total max_degrade_total réparti sur n_iter
    pour les classes fortes.
  - Adam-α optionnellement scalé sur n_iter (cfg["beta_adaptive"]=True).
  - Sauvegarde optionnelle "best-on-val" en parallèle (cfg["save_best_on_val"]).

Les scripts thin spike_optimizer_isic2019_{ce,fl,cb}.py instancient cette
classe avec leur config et leurs splits.
"""
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize as scipy_minimize


ISIC_CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
N_CLASSES    = len(ISIC_CLASSES)


# ════════════════════════════════════════════════════════════════════════════
# Helpers — bas niveau
# ════════════════════════════════════════════════════════════════════════════

def per_class_accuracy(model, x, y, n_classes=N_CLASSES):
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
        size = np.prod(var.shape)
        var.assign_add(tf.constant(
            delta_flat[idx:idx+size].reshape(var.shape), dtype=var.dtype
        ))
        idx += size


def compute_eigenvectors(model, loss_fn, x_hvp, y_hvp, lanczos_m, n_spikes,
                         hvp_batch_size=32):
    """Top-n_spikes via Lanczos. HVP sur CPU pour éviter OOM Metal."""
    from spectral_tools import HessianVectorProduct, StochasticLanczosQuadrature
    gpu_devices = tf.config.list_physical_devices("GPU")
    if gpu_devices:
        with tf.device("/CPU:0"):
            cpu_model = tf.keras.models.clone_model(model)
            cpu_model.set_weights(model.get_weights())
        hvp_model = cpu_model
    else:
        hvp_model = model
    hvp = HessianVectorProduct(
        model=hvp_model, loss_fn=loss_fn,
        data_x=x_hvp, data_y=y_hvp, batch_size=hvp_batch_size,
    )
    slq = StochasticLanczosQuadrature(
        hvp=hvp, n_params=hvp.n_params, m=20, k=1, sigma2=1e-5,
    )
    return slq.estimate_top_eigenvalues(m_lanczos=lanczos_m, verbose=False)


def compute_sensitivity(model, ritz_vecs, n_spikes, eps_probe, x_test, y_test):
    """S[i,j] = (acc(θ+ε·qᵢ) - acc(θ-ε·qᵢ)) / (2ε)"""
    current_w = save_weights(model)
    S = np.zeros((n_spikes, N_CLASSES))
    for s in range(n_spikes):
        delta = (eps_probe * ritz_vecs[:, s]).astype(np.float32)
        apply_perturbation(model, delta)
        acc_pos = per_class_accuracy(model, x_test, y_test)
        apply_perturbation(model, -2 * delta)
        acc_neg = per_class_accuracy(model, x_test, y_test)
        restore_weights(model, current_w)
        S[s] = (acc_pos - acc_neg) / (2 * eps_probe)
    return S


def optimize_alpha(S, acc_current, acc_baseline, alpha_max, ritz_vals, n_iter,
                   max_degrade_total=0.06, max_degrade_iter=0.03,
                   strong_class_threshold=0.70):
    """α* = argmax ω·(Sᵀα) sous contraintes.
       ω ∝ (acc_best - acc_c) ; |αᵢ| ≤ α_max·√(λ_min/λᵢ)
       Classes fortes (acc_baseline > τ) : -max_degrade_total/n_iter par iter
       Classes faibles : -max_degrade_iter par iter
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
        per_iter_limit  = max_degrade_total / n_iter
        limits = np.where(acc_baseline > strong_class_threshold,
                          -per_iter_limit, -max_degrade_iter)
        return np.min(predicted_delta - limits)

    result = scipy_minimize(
        objective, x0=np.zeros(len(S)),
        method="SLSQP",
        constraints=[{"type": "ineq", "fun": constraint_no_degrade}],
        bounds=[(-b, b) for b in alpha_budgets],
    )
    return result.x


# ════════════════════════════════════════════════════════════════════════════
# Runner ISIC
# ════════════════════════════════════════════════════════════════════════════

class SpikeOptimizerISIC:
    """Hessian Surgery pour ISIC-2019.

    cfg attendus (clés obligatoires) :
        model_path, model_out, output_dir, save_model,
        n_iter, patience,
        alpha_max_init, alpha_min, beta_ema,
        lanczos_m, n_spikes, eps_probe,

    cfg optionnels (avec défauts) :
        max_degrade_total      = 0.06
        max_degrade_iter       = 0.03
        rollback_std_tol       = 0.005
        rollback_drop_tol      = 0.07
        hvp_batch_size         = 32
        beta_adaptive          = False   # CB : β1=1-4/n_iter, β2=1-1/n_iter
        save_best_on_val       = False   # CB : save ckpt val_std minimal
    """

    def __init__(self, model, loss_fn, x_sens, y_sens, x_eval, y_eval,
                 x_hvp, y_hvp, cfg, x_val_mon=None, y_val_mon=None):
        self.model     = model
        self.loss_fn   = loss_fn
        self.x_sens    = x_sens
        self.y_sens    = y_sens
        self.x_eval    = x_eval
        self.y_eval    = y_eval
        self.x_val_mon = x_val_mon if x_val_mon is not None else x_eval
        self.y_val_mon = y_val_mon if y_val_mon is not None else y_eval
        self.x_hvp     = x_hvp
        self.y_hvp     = y_hvp
        self.cfg       = cfg
        os.makedirs(cfg["output_dir"], exist_ok=True)

    def run(self):
        cfg   = self.cfg
        alpha_max = cfg["alpha_max_init"]
        beta      = cfg["beta_ema"]
        log       = []

        max_degrade_total = cfg.get("max_degrade_total", 0.06)
        max_degrade_iter  = cfg.get("max_degrade_iter", 0.03)
        rollback_std_tol  = cfg.get("rollback_std_tol", 0.005)
        rollback_drop_tol = cfg.get("rollback_drop_tol", 0.07)
        hvp_batch_size    = cfg.get("hvp_batch_size", 32)
        beta_adaptive     = cfg.get("beta_adaptive", False)
        save_best_on_val  = cfg.get("save_best_on_val", False)

        # Baseline
        acc_baseline = per_class_accuracy(self.model, self.x_sens, self.y_sens)
        acc_global_0 = self.model.evaluate(
            self.x_sens, self.y_sens, verbose=0, batch_size=64)[1]
        acc_val_base = per_class_accuracy(self.model, self.x_val_mon, self.y_val_mon)
        val_global_0 = self.model.evaluate(
            self.x_val_mon, self.y_val_mon, verbose=0, batch_size=64)[1]
        print(f"  [val_mon baseline] global={val_global_0:.4f}  "
              f"bal={acc_val_base.mean():.4f}  std={acc_val_base.std():.4f}")
        self._print_baseline(acc_baseline, acc_global_0)

        acc_current  = acc_baseline.copy()
        std_ema      = float(np.std(acc_baseline))
        best_std_ema = std_ema
        no_improve   = 0

        adam_m, adam_v, adam_t = 0.0, 0.0, 0
        if beta_adaptive:
            # β1 = 1 - 4/n_iter, β2 = 1 - 1/n_iter. Requires n_iter ≥ 5
            # to keep β1 ≥ 0.2 (and avoid β1^t = 1 in bias-correction).
            if cfg["n_iter"] < 5:
                raise ValueError(
                    f"beta_adaptive=True requires n_iter ≥ 5, got {cfg['n_iter']}. "
                    "Either disable beta_adaptive or raise n_iter."
                )
            β1_a = 1.0 - 4.0 / cfg["n_iter"]
            β2_a = 1.0 - 1.0 / cfg["n_iter"]
        else:
            β1_a, β2_a = 0.9, 0.999
        ε_a = 1e-8

        best_val_std  = float("inf")
        best_val_iter = 0

        print(f"\n{'='*72}")
        print(f"  ITÉRATIONS — {cfg['n_iter']} chocs  "
              f"α_init={alpha_max}  β_ema={beta}  "
              f"β_adam=({β1_a:.3f},{β2_a:.3f})")
        print(f"{'='*72}")

        for it in range(1, cfg["n_iter"] + 1):
            t0 = time.time()
            print(f"\n  --- Itération {it}/{cfg['n_iter']}  ‖α‖≤{alpha_max:.4f} ---")

            ritz_vals, ritz_vecs = compute_eigenvectors(
                self.model, self.loss_fn, self.x_hvp, self.y_hvp,
                cfg["lanczos_m"], cfg["n_spikes"],
                hvp_batch_size=hvp_batch_size,
            )
            n_sp = min(cfg["n_spikes"], ritz_vecs.shape[1])
            print(f"  λ top-{n_sp} : {ritz_vals[:n_sp].round(1).tolist()}")

            S = compute_sensitivity(
                self.model, ritz_vecs, n_sp, alpha_max,
                self.x_sens, self.y_sens,
            )
            header = "  spike  " + "  ".join(f"{c:>5s}" for c in ISIC_CLASSES)
            print(f"\n  Matrice S (spike × classe) :")
            print(header)
            for s in range(n_sp):
                row = "  ".join(f"{S[s,c]:>+5.2f}" for c in range(N_CLASSES))
                print(f"  q{s+1:02d}    {row}")
            print()

            alpha = optimize_alpha(
                S, acc_current, acc_baseline, alpha_max, ritz_vals[:n_sp],
                cfg["n_iter"], max_degrade_total, max_degrade_iter,
            )

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
            max_class_drop = float(np.max(acc_current - acc_new))
            std_measured   = cur_std
            if cur_std > std_before + rollback_std_tol or max_class_drop > rollback_drop_tol:
                worst_class = ISIC_CLASSES[int(np.argmax(acc_current - acc_new))]
                restore_weights(self.model, weights_before)
                cur_std    = std_before
                acc_new    = acc_current.copy()
                delta_acc  = np.zeros_like(acc_current)
                acc_global = self.model.evaluate(
                    self.x_sens, self.y_sens, verbose=0, batch_size=64)[1]
                rolled_back = True
                if max_class_drop > rollback_drop_tol:
                    print(f"  [ROLLBACK] {worst_class} -{max_class_drop*100:.1f}% > "
                          f"{rollback_drop_tol*100:.0f}% → poids restaurés")
                else:
                    print(f"  [ROLLBACK] std {std_before:.4f}→{std_measured:.4f} → poids restaurés")
            else:
                acc_current = acc_new.copy()

            std_ema = beta * std_ema + (1.0 - beta) * cur_std
            if rolled_back:
                g_adam = -(std_measured - std_before)
                no_improve += 1
            elif std_ema < best_std_ema - 1e-4:
                g_adam = std_before - cur_std
                best_std_ema = std_ema
                no_improve   = 0
            else:
                g_adam = 0.0
                no_improve += 1

            adam_t += 1
            adam_m  = β1_a * adam_m + (1 - β1_a) * g_adam
            adam_v  = β2_a * adam_v + (1 - β2_a) * g_adam ** 2
            m_hat   = adam_m / (1 - β1_a ** adam_t)
            v_hat   = adam_v / (1 - β2_a ** adam_t)
            snr     = m_hat / (np.sqrt(v_hat) + ε_a)
            α_frac  = 0.5 + 0.5 * float(np.tanh(snr * 5.0))
            alpha_max = cfg["alpha_min"] + α_frac * (cfg["alpha_max_init"] - cfg["alpha_min"])
            alpha_max = float(np.clip(alpha_max, cfg["alpha_min"], cfg["alpha_max_init"]))
            print(f"  [adam] g={g_adam:+.4f}  snr={snr:.3f}  α_max={alpha_max:.4f}")

            if no_improve >= cfg.get("patience", 3) and alpha_max <= cfg["alpha_min"] + 1e-6:
                print(f"  [EARLY STOP] {no_improve} iters sans amélioration → arrêt")
                break

            val_acc_pc = per_class_accuracy(self.model, self.x_val_mon, self.y_val_mon)
            val_global = self.model.evaluate(
                self.x_val_mon, self.y_val_mon, verbose=0, batch_size=64)[1]
            val_bal = float(val_acc_pc.mean())
            val_std = float(val_acc_pc.std())

            elapsed = time.time() - t0
            rb_tag  = " ↩ROLLBACK" if rolled_back else ""
            print(f"  [TRAIN] global={acc_global:.4f}  std={cur_std:.4f}  "
                  f"ema={std_ema:.4f}  ({elapsed:.0f}s){rb_tag}")
            print(f"  [VAL]   global={val_global:.4f}  bal={val_bal:.4f}  "
                  f"std={val_std:.4f}")
            header = "  " + "  ".join(f"{n[:4]:>5s}" for n in ISIC_CLASSES)
            vals   = "  " + "  ".join(f"{acc_new[c]*100:>5.1f}" for c in range(N_CLASSES))
            delts  = "  " + "  ".join(f"{delta_acc[c]*100:>+5.1f}" for c in range(N_CLASSES))
            print(header); print(vals); print(delts)

            log.append({
                "iteration"  : it,
                "acc_global" : float(acc_global),
                "std"        : cur_std,
                "std_ema"    : std_ema,
                "val_global" : float(val_global),
                "val_bal"    : val_bal,
                "val_std"    : val_std,
                "alpha_max"  : alpha_max,
                "lambda_max" : float(ritz_vals[0]),
                "alpha_norm" : float(np.linalg.norm(alpha)),
                "rolled_back": rolled_back,
                "elapsed_s"  : elapsed,
                **{ISIC_CLASSES[c]: float(acc_new[c])          for c in range(N_CLASSES)},
                **{f"d_{ISIC_CLASSES[c]}": float(delta_acc[c]) for c in range(N_CLASSES)},
                **{f"val_{ISIC_CLASSES[c]}": float(val_acc_pc[c]) for c in range(N_CLASSES)},
            })

            pd.DataFrame(log).to_csv(
                os.path.join(cfg["output_dir"], "iteration_log.csv"), index=False)
            if not rolled_back and cfg.get("save_model"):
                self.model.save(cfg["model_out"])

            if (save_best_on_val and not rolled_back
                    and val_std < best_val_std and cfg.get("save_model")):
                best_val_std  = val_std
                best_val_iter = it
                best_val_path = cfg["model_out"].replace(".keras", "_bestval.keras")
                self.model.save(best_val_path)
                print(f"  [ckpt-val] best val_std={val_std:.4f} (iter {it}) → {best_val_path}")

        self._print_summary(acc_baseline, acc_global_0, log)
        self._save(acc_baseline, acc_global_0, log)
        self._plot(acc_baseline, log)
        self._final_evaluation(log)

        if cfg.get("save_model"):
            self.model.save(cfg["model_out"])
            print(f"\n  Modèle post-spike sauvegardé : {cfg['model_out']}")
        return log

    # ── prints / saves / plots ───────────────────────────────────────────

    def _print_baseline(self, acc, acc_global):
        print(f"\n  Baseline  acc_global={acc_global:.4f}  std={np.std(acc):.4f}")
        for c, name in enumerate(ISIC_CLASSES):
            print(f"    {name:4s} : {acc[c]*100:.1f}%")

    def _print_summary(self, acc_baseline, acc_global_0, log):
        acc_final = np.array([log[-1][n] for n in ISIC_CLASSES])
        acc_gf    = log[-1]["acc_global"]
        print(f"\n{'='*60}\n  RÉSUMÉ FINAL\n{'='*60}")
        print(f"  {'Classe':6s} {'baseline':>9s} {'final':>9s} {'Δ':>8s}")
        print("  " + "─" * 36)
        for c, name in enumerate(ISIC_CLASSES):
            print(f"  {name:6s} {acc_baseline[c]*100:>8.1f}% "
                  f"{acc_final[c]*100:>8.1f}% "
                  f"{(acc_final[c]-acc_baseline[c])*100:>+7.1f}%")
        print("  " + "─" * 36)
        print(f"  {'GLOBAL':6s} {acc_global_0*100:>8.1f}% "
              f"{acc_gf*100:>8.1f}% {(acc_gf-acc_global_0)*100:>+7.1f}%")
        print(f"  std : {np.std(acc_baseline)*100:.2f}% → "
              f"{np.std(acc_final)*100:.2f}%  "
              f"(Δ={(np.std(acc_final)-np.std(acc_baseline))*100:+.2f}%)")
        print(f"{'='*60}")

    def _final_evaluation(self, log):
        print(f"\n{'='*60}\n  ÉVAL FINALE — held-out ({len(self.y_eval)} images)\n{'='*60}")
        acc_eval     = per_class_accuracy(self.model, self.x_eval, self.y_eval)
        results_eval = self.model.evaluate(
            self.x_eval, self.y_eval, verbose=0, batch_size=64)
        acc_global_eval, loss_eval = results_eval[1], results_eval[0]
        print(f"  acc_global : {acc_global_eval*100:.2f}%")
        print(f"  bal_acc    : {acc_eval.mean()*100:.2f}%")
        print(f"  loss       : {loss_eval:.4f}")
        print(f"  std        : {np.std(acc_eval)*100:.2f}%")
        for c, name in enumerate(ISIC_CLASSES):
            print(f"    {name:4s} : {acc_eval[c]*100:.1f}%")
        out = self.cfg["output_dir"]
        pd.DataFrame([{
            "acc_global_eval"  : float(acc_global_eval),
            "balanced_acc_eval": float(acc_eval.mean()),
            "loss_eval"        : float(loss_eval),
            "std_eval"         : float(np.std(acc_eval)),
            **{f"eval_{n}": float(acc_eval[c]) for c, n in enumerate(ISIC_CLASSES)},
        }]).to_csv(os.path.join(out, "eval_heldout.csv"), index=False)
        print(f"\n  Held-out → {out}/eval_heldout.csv")

    def _save(self, acc_baseline, acc_global_0, log):
        out = self.cfg["output_dir"]
        acc_final = np.array([log[-1][n] for n in ISIC_CLASSES])
        pd.DataFrame([{
            "n_iter"      : len(log),
            "acc_global_0": acc_global_0,
            "acc_global_f": log[-1]["acc_global"],
            "std_0"       : float(np.std(acc_baseline)),
            "std_f"       : log[-1]["std"],
            "bal_acc_0"   : float(acc_baseline.mean()),
            "bal_acc_f"   : float(acc_final.mean()),
            **{f"baseline_{n}": float(acc_baseline[c]) for c, n in enumerate(ISIC_CLASSES)},
            **{f"final_{n}"   : float(acc_final[c])    for c, n in enumerate(ISIC_CLASSES)},
        }]).to_csv(os.path.join(out, "summary.csv"), index=False)
        print(f"  CSV → {out}/")

    def _plot(self, acc_baseline, log):
        out   = self.cfg["output_dir"]
        iters = [r["iteration"] for r in log]
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle("Spike Optimizer ISIC2019", fontsize=13)
        colors = plt.cm.tab10(np.linspace(0, 1, N_CLASSES))

        ax = axes[0, 0]
        for c, name in enumerate(ISIC_CLASSES):
            ax.plot(iters, [r[name]*100 for r in log], "o-",
                    color=colors[c], label=name, lw=1.5, ms=4)
            ax.axhline(acc_baseline[c]*100, color=colors[c], ls=":", lw=0.8, alpha=0.5)
        ax.set_xlabel("Itération"); ax.set_ylabel("Accuracy (%)")
        ax.set_title("Par classe (pointillés = baseline)")
        ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

        ax = axes[0, 1]
        for c, name in enumerate(ISIC_CLASSES):
            if name in ("DF", "VASC", "AK"):
                ax.plot(iters, [r[name]*100 for r in log], "o-",
                        color=colors[c], label=name, lw=2, ms=5)
                ax.axhline(acc_baseline[c]*100, color=colors[c], ls=":", lw=1, alpha=0.6)
        ax.set_xlabel("Itération"); ax.set_ylabel("Accuracy (%)")
        ax.set_title("Classes rares (DF, VASC, AK)")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

        ax = axes[1, 0]
        ax.plot(iters, [r["std"]*100 for r in log], "s-",
                color="crimson", lw=2, ms=5)
        ax.axhline(np.std(acc_baseline)*100, color="crimson",
                   ls=":", lw=1, alpha=0.6, label="baseline")
        ax.set_xlabel("Itération"); ax.set_ylabel("Std (%)")
        ax.set_title("Variance inter-classes")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

        ax = axes[1, 1]
        ax.step(iters, [r["alpha_max"] for r in log],
                where="post", color="darkorange", lw=2)
        ax.set_xlabel("Itération"); ax.set_ylabel("α_max")
        ax.set_title("Décay adaptatif")
        ax.grid(alpha=0.3)

        plt.tight_layout()
        path = os.path.join(out, "evolution.png")
        plt.savefig(path, dpi=150); plt.close()
        print(f"  Fig → {path}")


# ════════════════════════════════════════════════════════════════════════════
# Helpers data-loading (utilisés par les 3 scripts thin)
# ════════════════════════════════════════════════════════════════════════════

def load_isic_caches(cfg):
    """Charge val_mon, eval, train depuis les 3 caches .npz."""
    val_data  = np.load(cfg["cache_val"])
    test_data = np.load(cfg["cache_test"])
    train_data = np.load(cfg["cache_train"])
    return (
        val_data["imgs"].astype(np.float32),  val_data["labels"].astype(np.int32),
        test_data["imgs"].astype(np.float32), test_data["labels"].astype(np.int32),
        train_data["imgs"].astype(np.float32), train_data["labels"].astype(np.int32),
    )


def sample_hvp_random(x_train, y_train, n_samples, rng):
    """HVP uniformément aléatoire — loss naturelle (CE, FL)."""
    idx = rng.choice(len(x_train), n_samples, replace=False)
    return x_train[idx], y_train[idx]


def sample_hvp_balanced(x_train, y_train, n_samples, rng, n_classes=N_CLASSES):
    """HVP balancé par classe — estime la Hessienne de la loss balancée (CB)."""
    n_per_class = n_samples // n_classes
    idx_list = []
    for c in range(n_classes):
        cls_idx = np.where(y_train == c)[0]
        idx_list.append(rng.choice(cls_idx, n_per_class, replace=False))
    idx = np.concatenate(idx_list)
    rng.shuffle(idx)
    return x_train[idx], y_train[idx]


def sample_sensitivity_stratified(x_train, y_train, max_per_class, rng,
                                  n_classes=N_CLASSES, class_names=None):
    """Sensitivity set stratifié : cap les classes majoritaires."""
    class_names = class_names or ISIC_CLASSES
    idx_list = []
    print(f"  Sensitivity stratifié (max {max_per_class}/classe) :")
    for c, name in enumerate(class_names):
        cls_idx = np.where(y_train == c)[0]
        n_take  = min(len(cls_idx), max_per_class)
        idx_list.append(rng.choice(cls_idx, n_take, replace=False))
        print(f"    {name:4s} : {n_take:4d} / {len(cls_idx)}")
    idx = np.concatenate(idx_list)
    rng.shuffle(idx)
    return x_train[idx], y_train[idx]
