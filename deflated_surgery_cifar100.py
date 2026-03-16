"""
deflated_surgery_cifar100.py
-----------------------------
Sequential Deflated Spectral Surgery pour CIFAR-100.

Au lieu d'un seul Lanczos m=100 (impossible en RAM), on lance plusieurs phases
qui ciblent chacune une tranche du spectre spike via déflation :

  Phase 1 : spikes 1–30   (Lanczos standard, m=40)
  Phase 2 : spikes 31–60  (Lanczos sur HVP déflaté par Q₁)
  Phase 3 : spikes 61–90  (Lanczos sur HVP déflaté par [Q₁,Q₂])
  Phase 4 : spikes 91–99  (Lanczos m=15 sur HVP déflaté par [Q₁,Q₂,Q₃])

Chaque phase :
  - Calcule les eigenvectors via Lanczos sur l'oracle déflaté
  - Lance T itérations de Spectral Surgery (early stop si σ stagne 2 itérations)
  - Sauvegarde eigenvectors, modèle, log

Usage :
    python deflated_surgery_cifar100.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import time
from scipy.optimize import minimize as scipy_minimize

from spectral_tools import HessianVectorProduct, LanczosAlgorithm, StochasticLanczosQuadrature
from spike_optimizer_cifar100 import (
    per_class_accuracy, save_weights, restore_weights,
    apply_perturbation, CIFAR100_CLASSES, N_CLASSES,
)

# ════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════

OUTPUT_DIR = "results/deflated_surgery_cifar100"

PHASES = [
    {"name": "phase1", "n_spikes": 15, "lanczos_m": 25, "max_iter": 3, "eps_probe": 0.005},
    {"name": "phase2", "n_spikes": 15, "lanczos_m": 25, "max_iter": 3, "eps_probe": 0.05},
    {"name": "phase3", "n_spikes": 15, "lanczos_m": 25, "max_iter": 3, "eps_probe": 0.1},
    {"name": "phase4", "n_spikes": 15, "lanczos_m": 25, "max_iter": 3, "eps_probe": 0.1},
]

ALPHA_MAX_REF = 0.025   # α_max de référence (calibré pour λ_ref)
LAMBDA_REF    = 600.0   # λ_max de référence (phase 1)
ALPHA_MIN     = 0.003
DECAY_FACTOR  = 0.7
BETA_EMA      = 0.3
N_HVP_SAMPLES = 256
STALL_PATIENCE = 2   # passe à la phase suivante si σ stagne 2 itérations
N_SENS_SAMPLES = 2000  # sous-ensemble de x_test pour la sensibilité (au lieu de 10k)


# ════════════════════════════════════════════════════════════════════════════
# Deflated HVP Oracle
# ════════════════════════════════════════════════════════════════════════════

class DeflatedHVP:
    """
    Oracle HVP déflaté : projette v hors du sous-espace Q_prev,
    applique le HVP original, puis projette le résultat aussi.

    Cela fait que Lanczos ne retrouve pas les eigenvectors déjà traités
    et converge vers les suivants dans le spectre.
    """

    def __init__(self, original_hvp, Q_prev):
        """
        original_hvp : instance de HessianVectorProduct (méthode .compute(v))
        Q_prev       : np.ndarray shape (n_params, k), colonnes orthonormales
                        des eigenvectors des phases précédentes.
                        Stocké en float16, casté en float32/64 pour les projections.
        """
        self.original_hvp = original_hvp
        # Re-orthogonaliser Q_prev pour stabilité numérique
        if Q_prev is not None and Q_prev.shape[1] > 0:
            Q_f32 = Q_prev.astype(np.float32)
            self.Q, _ = np.linalg.qr(Q_f32)
        else:
            self.Q = None
        self.n_params = original_hvp.n_params

    def compute(self, v):
        """HVP déflaté : projette v et Hv hors du sous-espace précédent."""
        if self.Q is None:
            return self.original_hvp.compute(v)

        v = v.astype(np.float64)
        Q = self.Q.astype(np.float64)

        # Projeter v hors du sous-espace précédent
        v_def = v - Q @ (Q.T @ v)

        # Appliquer HVP original
        Hv = self.original_hvp.compute(v_def.astype(np.float32)).astype(np.float64)

        # Projeter le résultat aussi
        Hv_def = Hv - Q @ (Q.T @ Hv)

        return Hv_def.astype(np.float32)


# ════════════════════════════════════════════════════════════════════════════
# Fonctions utilitaires
# ════════════════════════════════════════════════════════════════════════════

def compute_eigenvectors_deflated(hvp_oracle, n_params, lanczos_m, n_spikes):
    """Lanczos sur un oracle (standard ou déflaté)."""
    lanczos = LanczosAlgorithm()
    v0 = np.random.randn(n_params).astype(np.float64)
    v0 /= np.linalg.norm(v0)

    alpha, beta, Q = lanczos.run(
        hvp_oracle.compute, v0, lanczos_m,
        store_vectors=True, verbose=False,
    )

    if len(alpha) == 1:
        return alpha, Q

    from scipy.linalg import eigh_tridiagonal
    ritz_vals, U = eigh_tridiagonal(alpha, beta)
    idx = np.argsort(ritz_vals)[::-1]
    ritz_vals = ritz_vals[idx]
    U = U[:, idx]
    ritz_vecs = Q @ U

    # Garder seulement les n_spikes premiers
    n_sp = min(n_spikes, ritz_vecs.shape[1])
    return ritz_vals[:n_sp], ritz_vecs[:, :n_sp]


def compute_sensitivity(model, ritz_vecs, n_spikes, eps_probe, x_sens, y_sens):
    """Matrice de sensibilité S[spike, classe] sur un sous-ensemble."""
    current_w = save_weights(model)
    S = np.zeros((n_spikes, N_CLASSES))
    for s in range(n_spikes):
        qi = ritz_vecs[:, s]
        restore_weights(model, current_w)
        apply_perturbation(model, (eps_probe * qi).astype(np.float32))
        acc_pos = per_class_accuracy(model, x_sens, y_sens)
        restore_weights(model, current_w)
        apply_perturbation(model, (-eps_probe * qi).astype(np.float32))
        acc_neg = per_class_accuracy(model, x_sens, y_sens)
        S[s] = (acc_pos - acc_neg) / (2 * eps_probe)
        if (s + 1) % 5 == 0:
            print(f"      sensibilité spike {s+1}/{n_spikes}")
    restore_weights(model, current_w)
    return S


def optimize_alpha(S, acc_current, acc_original, alpha_max):
    """
    SLSQP avec contrainte de non-dégradation cumulée.
    La contrainte tient compte de la dégradation accumulée depuis la baseline originale.
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
        # Dégradation cumulée depuis la baseline originale
        degradation = acc_current - acc_original
        predicted_delta = S.T @ alpha
        strong = acc_current > 0.85
        if strong.sum() == 0:
            return 1.0
        # La somme dégradation + predicted_delta ne doit pas dépasser -1%
        return np.min(predicted_delta[strong] + degradation[strong]) + 0.01

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
# Phase runner
# ════════════════════════════════════════════════════════════════════════════

def run_phase(phase_cfg, model, loss_fn, x_sens_pool, y_sens_pool, x_hvp, y_hvp,
              Q_all_prev, acc_original, alpha_max, phase_dir):
    """
    Lance une phase de Spectral Surgery.

    Retourne : (Q_new, log, alpha_max_final)
    """
    os.makedirs(phase_dir, exist_ok=True)
    n_spikes  = phase_cfg["n_spikes"]
    lanczos_m = phase_cfg["lanczos_m"]
    max_iter  = phase_cfg["max_iter"]

    # Sous-ensemble stratifié pour la sensibilité (N_SENS_SAMPLES images)
    # Tiré depuis x_sens_pool (5000 images), PAS x_eval
    sens_rng = np.random.default_rng(42)
    per_class = max(N_SENS_SAMPLES // N_CLASSES, 1)
    sens_idx = []
    for c in range(N_CLASSES):
        mask = np.where(y_sens_pool == c)[0]
        n_take = min(per_class, len(mask))
        sens_idx.extend(sens_rng.choice(mask, n_take, replace=False).tolist())
    sens_idx = np.array(sens_idx)
    x_sens = x_sens_pool[sens_idx]
    y_sens = y_sens_pool[sens_idx]
    print(f"    Sous-ensemble sensibilité : {len(sens_idx)} images ({per_class}/classe)")

    # Construire l'oracle HVP (standard ou déflaté)
    hvp = HessianVectorProduct(
        model=model, loss_fn=loss_fn,
        data_x=x_hvp, data_y=y_hvp, batch_size=None,
    )

    if Q_all_prev is not None and Q_all_prev.shape[1] > 0:
        oracle = DeflatedHVP(hvp, Q_all_prev)
        print(f"    Oracle déflaté (projection hors de {Q_all_prev.shape[1]} vecteurs)")
    else:
        oracle = hvp
        print(f"    Oracle standard (pas de déflation)")

    # État initial (sur le set de sensibilité, pas le held-out)
    acc_current = per_class_accuracy(model, x_sens_pool, y_sens_pool)
    std_current = float(np.std(acc_current))
    std_ema     = std_current
    best_std_ema = std_ema

    log = []
    stall_count = 0
    Q_phase = None  # eigenvectors trouvés dans cette phase

    for it in range(1, max_iter + 1):
        t0 = time.time()
        print(f"\n    --- Phase itération {it}/{max_iter}  ‖α‖≤{alpha_max:.4f} ---")

        # Eigenvectors via Lanczos (déflaté si applicable)
        ritz_vals, ritz_vecs = compute_eigenvectors_deflated(
            oracle, hvp.n_params, lanczos_m, n_spikes,
        )
        n_sp = min(n_spikes, ritz_vecs.shape[1])
        print(f"    λ top-5 : {ritz_vals[:5].round(1).tolist()}")

        # Sauvegarder les eigenvectors de la première itération
        # et adapter α_max ∝ 1/√λ_max pour homogénéiser l'impact entre phases
        if it == 1:
            Q_phase = ritz_vecs[:, :n_sp].copy()
            lambda_max_phase = float(ritz_vals[0])
            alpha_max = ALPHA_MAX_REF * np.sqrt(LAMBDA_REF / max(lambda_max_phase, 1.0))
            print(f"    α_max adapté : {alpha_max:.4f} (λ_max={lambda_max_phase:.1f}, "
                  f"ref={ALPHA_MAX_REF}×√({LAMBDA_REF:.0f}/{lambda_max_phase:.1f}))")

        # Sensibilité (eps_probe adapté par phase)
        eps_probe = phase_cfg.get("eps_probe", 0.005)
        print(f"    Sensibilité ({n_sp} spikes × {N_CLASSES} classes, ε={eps_probe}, "
              f"n_sens={len(x_sens)}) ...")
        S = compute_sensitivity(model, ritz_vecs, n_sp, eps_probe, x_sens, y_sens)

        # Optimiser α (avec contrainte cumulée)
        alpha = optimize_alpha(S, acc_current, acc_original, alpha_max)

        # Sauvegarder poids avant choc
        weights_before = save_weights(model)
        std_before = float(np.std(acc_current))

        # Appliquer choc
        delta = np.zeros(ritz_vecs.shape[0], dtype=np.float64)
        for s in range(n_sp):
            delta += alpha[s] * ritz_vecs[:, s]
        apply_perturbation(model, delta.astype(np.float32))

        # Mesurer (sur le set de sensibilité pour le pilotage)
        acc_new = per_class_accuracy(model, x_sens_pool, y_sens_pool)
        acc_global = model.evaluate(x_sens_pool, y_sens_pool, verbose=0, batch_size=256)[1]
        delta_acc = acc_new - acc_current
        cur_std = float(np.std(acc_new))

        # Rollback si dégradation : hausse de std OU chute de global acc > 2%
        rolled_back = False
        acc_global_before = float(np.mean(acc_current))
        global_drop = (acc_global < acc_global_before - 0.02)
        if cur_std > std_before + 0.005 or global_drop:
            std_measured = cur_std
            restore_weights(model, weights_before)
            cur_std = std_before
            acc_new = acc_current.copy()
            delta_acc = np.zeros_like(acc_current)
            acc_global = model.evaluate(x_sens_pool, y_sens_pool, verbose=0, batch_size=256)[1]
            rolled_back = True
            reason = "global_drop" if global_drop else "std_increase"
            print(f"    [ROLLBACK] {reason} — std {std_before:.4f}→{std_measured:.4f}, "
                  f"global {acc_global_before:.4f}→{acc_global:.4f}")
        else:
            acc_current = acc_new.copy()

        # EMA + decay
        std_ema = BETA_EMA * std_ema + (1.0 - BETA_EMA) * cur_std
        if rolled_back:
            alpha_max = max(alpha_max * DECAY_FACTOR, ALPHA_MIN)
            best_std_ema = std_ema
            stall_count += 1
            print(f"    [decay] rollback → α_max={alpha_max:.4f}")
        elif std_ema < best_std_ema - 1e-4:
            best_std_ema = std_ema
            stall_count = 0
        else:
            alpha_max = max(alpha_max * DECAY_FACTOR, ALPHA_MIN)
            stall_count += 1
            print(f"    [decay] std_ema stagne → α_max={alpha_max:.4f}")

        elapsed = time.time() - t0
        rb_tag = " ↩ROLLBACK" if rolled_back else ""
        print(f"    global={acc_global:.4f}  std={cur_std:.4f}  "
              f"ema={std_ema:.4f}  ({elapsed:.0f}s){rb_tag}")

        # Top classes affectées
        worst5 = np.argsort(acc_new)[:5]
        best5 = np.argsort(acc_new)[-5:][::-1]
        print(f"    Pires 5 : " + "  ".join(
            f"{CIFAR100_CLASSES[c][:8]}={acc_new[c]*100:.0f}%" for c in worst5))
        print(f"    Mieux 5 : " + "  ".join(
            f"{CIFAR100_CLASSES[c][:8]}={acc_new[c]*100:.0f}%" for c in best5))

        log.append({
            "iteration": it,
            "acc_global": float(acc_global),
            "std": cur_std,
            "std_ema": std_ema,
            "alpha_max": alpha_max,
            "lambda_max": float(ritz_vals[0]),
            "alpha_norm": float(np.linalg.norm(alpha)),
            "rolled_back": rolled_back,
            "elapsed_s": elapsed,
            **{CIFAR100_CLASSES[c]: float(acc_new[c]) for c in range(N_CLASSES)},
        })

        # Early stopping : si σ stagne pendant STALL_PATIENCE itérations
        if stall_count >= STALL_PATIENCE:
            print(f"    [EARLY STOP] σ stagne depuis {stall_count} itérations → phase suivante")
            break

    # Sauvegardes
    pd.DataFrame(log).to_csv(os.path.join(phase_dir, "iteration_log.csv"), index=False)
    np.save(os.path.join(phase_dir, "eigenvectors.npy"), Q_phase.astype(np.float16))
    model.save(os.path.join(phase_dir, "model.keras"))
    np.save(os.path.join(phase_dir, "acc_current.npy"), acc_current)

    return Q_phase, log, alpha_max


# ════════════════════════════════════════════════════════════════════════════
# Point d'entrée
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-phase", type=int, default=1,
                        help="Phase de départ (1-4). Charge le modèle et Q_all de la phase précédente.")
    args = parser.parse_args()
    START_PHASE = args.start_phase

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Données ──────────────────────────────────────────────────────────
    print("[1] Chargement de CIFAR-100 ...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    y_train, y_test = y_train.flatten(), y_test.flatten()
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0

    # ── Split test set : 5000 sensibilité/pilotage + 5000 évaluation held-out ──
    rng = np.random.default_rng(0)
    idx_test = rng.permutation(len(x_test))
    x_sens_pool, y_sens_pool = x_test[idx_test[:5000]], y_test[idx_test[:5000]]
    x_eval, y_eval = x_test[idx_test[5000:]], y_test[idx_test[5000:]]
    print(f"    Test split : {len(x_sens_pool)} sensibilité + {len(x_eval)} évaluation held-out")

    idx = rng.choice(len(x_train), N_HVP_SAMPLES, replace=False)
    x_hvp = x_train[idx]
    y_hvp = y_train[idx]

    # ── Modèle (depuis checkpoint si start_phase > 1) ────────────────────
    if START_PHASE > 1:
        prev_phase = PHASES[START_PHASE - 2]["name"]
        prev_dir = os.path.join(OUTPUT_DIR, prev_phase)
        model_path = os.path.join(prev_dir, "model.keras")
        print(f"[2] Chargement du modèle depuis {model_path} ...")
        model = tf.keras.models.load_model(model_path)

        # Charger Q_all accumulé des phases précédentes
        Q_all_parts = []
        for i in range(START_PHASE - 1):
            q_path = os.path.join(OUTPUT_DIR, PHASES[i]["name"], "eigenvectors.npy")
            Q_all_parts.append(np.load(q_path))
            print(f"    Chargé eigenvectors {PHASES[i]['name']} : {Q_all_parts[-1].shape}")
        Q_all = np.concatenate(Q_all_parts, axis=1).astype(np.float16)
        print(f"    Q_all total : {Q_all.shape[1]} vecteurs ({Q_all.nbytes / 1e9:.1f} GB)")
    else:
        print("[2] Chargement du modèle ...")
        model = tf.keras.models.load_model("resnet50_cifar100.keras")
        Q_all = None

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    n_params = sum(np.prod(v.shape) for v in model.trainable_variables)
    print(f"    Paramètres : {n_params:,}")

    # ── Baseline originale (sur le set de sensibilité) ────────────────────
    print("[3] Baseline originale ...")
    # Recalculer la baseline (on ne charge pas de cache pour éviter l'ancien split)
    acc_original = per_class_accuracy(model, x_sens_pool, y_sens_pool)
    np.save(os.path.join(OUTPUT_DIR, "acc_original.npy"), acc_original)
    acc_global_0 = float(np.mean(acc_original))
    std_0 = float(np.std(acc_original))
    print(f"    Global={acc_global_0*100:.1f}%  Std={std_0*100:.2f}%  (sur 5000 sens)")

    # État courant du modèle
    acc_current_model = per_class_accuracy(model, x_sens_pool, y_sens_pool)
    acc_global_now = float(np.mean(acc_current_model))
    std_now = float(np.std(acc_current_model))
    print(f"    État actuel : Global={acc_global_now*100:.1f}%  Std={std_now*100:.2f}%")

    # ── Boucle sur les phases ────────────────────────────────────────────
    all_logs = []

    for phase_idx, phase_cfg in enumerate(PHASES):
        if phase_idx + 1 < START_PHASE:
            continue
        phase_num = phase_idx + 1
        phase_dir = os.path.join(OUTPUT_DIR, phase_cfg["name"])

        spike_start = sum(PHASES[i]["n_spikes"] for i in range(phase_idx)) + 1
        spike_end   = spike_start + phase_cfg["n_spikes"] - 1

        print(f"\n{'='*70}")
        print(f"  PHASE {phase_num} — spikes {spike_start}–{spike_end}  "
              f"(m={phase_cfg['lanczos_m']}, max_iter={phase_cfg['max_iter']})")
        print(f"{'='*70}")

        # alpha_max sera adapté automatiquement dans run_phase via λ_max
        alpha_max = ALPHA_MAX_REF  # valeur initiale, sera recalibrée au 1er Lanczos

        Q_new, log, alpha_max = run_phase(
            phase_cfg, model, loss_fn, x_sens_pool, y_sens_pool, x_hvp, y_hvp,
            Q_all, acc_original, alpha_max, phase_dir,
        )

        # Accumuler les eigenvectors (en float16 pour la mémoire)
        if Q_all is None:
            Q_all = Q_new.astype(np.float16)
        else:
            Q_all = np.concatenate([Q_all, Q_new.astype(np.float16)], axis=1)

        print(f"\n    Q_all : {Q_all.shape[1]} eigenvectors accumulés "
              f"({Q_all.nbytes / 1e9:.1f} GB)")

        # Vérification d'orthogonalité
        if phase_num >= 2:
            n_prev = Q_all.shape[1] - Q_new.shape[1]
            Q_prev_f32 = Q_all[:, :n_prev].astype(np.float32)
            Q_new_f32  = Q_new.astype(np.float32)
            cross = Q_prev_f32.T @ Q_new_f32
            max_cross = float(np.abs(cross).max())
            mean_cross = float(np.abs(cross).mean())
            print(f"    Orthogonalité Q_prev ⊥ Q_new : "
                  f"max|cos|={max_cross:.6f}  mean|cos|={mean_cross:.6f}")
            if max_cross > 0.1:
                print(f"    ⚠ ATTENTION : déflation fuit (max cross > 0.1)")

        # Eigenvalues de cette phase vs précédentes
        if log:
            lambda_max_phase = log[0]["lambda_max"]
            print(f"    λ_max phase {phase_num} : {lambda_max_phase:.1f}")

        # Log global
        for entry in log:
            entry["phase"] = phase_num
        all_logs.extend(log)

        # Résumé intermédiaire (sur le set de sensibilité)
        acc_current = per_class_accuracy(model, x_sens_pool, y_sens_pool)
        acc_global = model.evaluate(x_sens_pool, y_sens_pool, verbose=0, batch_size=256)[1]
        cur_std = float(np.std(acc_current))
        print(f"\n    Après phase {phase_num} : "
              f"global={acc_global*100:.1f}%  std={cur_std*100:.2f}%  "
              f"(Δstd={(cur_std - std_0)*100:+.2f}%)")

    # ── Résumé sur le set de sensibilité ─────────────────────────────────
    acc_final_sens = per_class_accuracy(model, x_sens_pool, y_sens_pool)
    acc_global_f_sens = model.evaluate(x_sens_pool, y_sens_pool, verbose=0, batch_size=256)[1]
    std_f_sens = float(np.std(acc_final_sens))

    print(f"\n{'='*70}")
    print(f"  RÉSUMÉ (set sensibilité, 5000 images)")
    print(f"{'='*70}")
    print(f"  Phases : {len(PHASES)}  |  Spikes totaux : {Q_all.shape[1]}")
    print(f"  Global : {acc_global_0*100:.1f}% → {acc_global_f_sens*100:.1f}%  "
          f"(Δ={(acc_global_f_sens - acc_global_0)*100:+.1f}%)")
    print(f"  Std    : {std_0*100:.2f}% → {std_f_sens*100:.2f}%  "
          f"(Δ={(std_f_sens - std_0)*100:+.2f}%)")

    # ── Évaluation finale held-out (non contaminé) ────────────────────────
    print(f"\n{'='*70}")
    print(f"  ÉVALUATION FINALE — set held-out (5000 images, non contaminé)")
    print(f"{'='*70}")
    acc_final = per_class_accuracy(model, x_eval, y_eval)
    acc_global_f = model.evaluate(x_eval, y_eval, verbose=0, batch_size=256)[1]
    std_f = float(np.std(acc_final))

    # Baseline held-out pour comparaison
    # (on recharge le modèle original pour ça — coûteux mais correct)
    # Plutôt : on compare avec les moyennes connues
    print(f"  Global held-out : {acc_global_f*100:.1f}%")
    print(f"  Std held-out    : {std_f*100:.2f}%")

    # Sauvegarder résultats held-out
    eval_summary = {
        "acc_global_eval": float(acc_global_f),
        "std_eval": float(std_f),
        **{f"eval_{CIFAR100_CLASSES[c]}": float(acc_final[c]) for c in range(N_CLASSES)},
    }
    pd.DataFrame([eval_summary]).to_csv(
        os.path.join(OUTPUT_DIR, "eval_heldout.csv"), index=False)

    # Top améliorations / dégradations (vs baseline sensibilité)
    delta = acc_final_sens - acc_original
    top_improve = np.argsort(delta)[-10:][::-1]
    top_degrade = np.argsort(delta)[:10]

    print(f"\n  Top 10 améliorations (set sensibilité) :")
    for c in top_improve:
        print(f"    {CIFAR100_CLASSES[c]:16s} : "
              f"{acc_original[c]*100:.1f}% → {acc_final_sens[c]*100:.1f}%  "
              f"({delta[c]*100:+.1f}%)")

    print(f"\n  Top 10 dégradations (set sensibilité) :")
    for c in top_degrade:
        print(f"    {CIFAR100_CLASSES[c]:16s} : "
              f"{acc_original[c]*100:.1f}% → {acc_final_sens[c]*100:.1f}%  "
              f"({delta[c]*100:+.1f}%)")

    print(f"{'='*70}")

    # Sauvegardes globales
    pd.DataFrame(all_logs).to_csv(
        os.path.join(OUTPUT_DIR, "all_phases_log.csv"), index=False)
    np.save(os.path.join(OUTPUT_DIR, "Q_all.npy"), Q_all)
    np.save(os.path.join(OUTPUT_DIR, "acc_final_sens.npy"), acc_final_sens)
    np.save(os.path.join(OUTPUT_DIR, "acc_final_eval.npy"), acc_final)
    model.save(os.path.join(OUTPUT_DIR, "model_final.keras"))

    # ── Plot récapitulatif ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Deflated Spectral Surgery — CIFAR-100", fontsize=13)

    # Panel 1 : distribution des accuracies (set sensibilité)
    ax = axes[0]
    ax.hist(acc_original * 100, bins=20, alpha=0.5, label="baseline", color="gray")
    ax.hist(acc_final_sens * 100, bins=20, alpha=0.5, label="après surgery", color="steelblue")
    ax.set_xlabel("Accuracy (%)")
    ax.set_ylabel("Nombre de classes")
    ax.set_title("Distribution des accuracies par classe")
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 2 : std par itération globale (toutes phases)
    ax = axes[1]
    stds = [r["std"] * 100 for r in all_logs]
    iters = list(range(1, len(stds) + 1))
    phases = [r["phase"] for r in all_logs]
    colors_phase = {1: "tab:blue", 2: "tab:orange", 3: "tab:green", 4: "tab:red"}
    for i, (x, y, p) in enumerate(zip(iters, stds, phases)):
        ax.plot(x, y, "o", color=colors_phase.get(p, "gray"), ms=6)
    ax.plot(iters, stds, "-", color="gray", alpha=0.5, lw=1)
    ax.axhline(std_0 * 100, color="crimson", ls=":", lw=1, alpha=0.6, label="baseline")
    # Phase boundaries
    for p_name, p_color in colors_phase.items():
        ax.plot([], [], "o", color=p_color, label=f"Phase {p_name}")
    ax.set_xlabel("Itération globale")
    ax.set_ylabel("Std inter-classes (%)")
    ax.set_title("Variance inter-classes par phase")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 3 : eigenvalues par phase
    ax = axes[2]
    for phase_idx, phase_cfg in enumerate(PHASES):
        phase_logs = [r for r in all_logs if r["phase"] == phase_idx + 1]
        if phase_logs:
            ax.bar(phase_idx + 1, phase_logs[0]["lambda_max"],
                   color=colors_phase.get(phase_idx + 1, "gray"),
                   label=f"Phase {phase_idx+1}")
    ax.set_xlabel("Phase")
    ax.set_ylabel("λ_max")
    ax.set_title("Plus grande eigenvalue par phase")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "deflated_surgery_summary.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  Figure : {path}")
    print(f"  Résultats : {OUTPUT_DIR}/")
    print(f"\n  TERMINÉ.")
