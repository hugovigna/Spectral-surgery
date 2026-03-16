"""
bulk_finetune.py
----------------
Fine-tuning projeté dans le bulk : entraîne le modèle en projetant les
gradients orthogonalement aux spikes de la Hessienne.

Principe :
  Les spikes encodent la structure inter-classes (Papyan 2020). Après un
  rééquilibrage via spike_optimizer, on veut améliorer l'accuracy globale
  SANS altérer l'équilibre obtenu. Solution : fine-tuner dans le complément
  orthogonal aux spikes — le "bulk" — où les performances progressent
  uniformément sur toutes les classes.

  À chaque step de training :
    g = gradient de la loss
    g_bulk = g - Σᵢ (g·qᵢ) qᵢ     (projection ⊥ spikes)
    θ ← θ - lr · g_bulk

Usage :
    python bulk_finetune.py                   # lance le FT
    python bulk_finetune.py --dry-run         # vérifie sans lancer

Paramètres configurables dans CONFIG ci-dessous.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import numpy as np
import tensorflow as tf
import time

# ════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════
CONFIG = {
    # ── Modèle et données ────────────────────────────────────────────────
    "model_path"   : "resnet50_cifar10_spiked.keras",
    "n_hvp_samples": 128,
    "seed"         : 0,

    # ── Analyse spectrale (pour calculer les spikes à projeter) ──────────
    "lanczos_m"    : 30,
    "n_spikes"     : 9,        # nombre de directions spike à projeter

    # ── Fine-tuning ──────────────────────────────────────────────────────
    "lr"           : 1e-4,     # learning rate pour le FT bulk
    "epochs"       : 5,
    "batch_size"   : 64,

    # ── Recalcul des spikes ──────────────────────────────────────────────
    "recompute_every": 2,      # recalcule les spikes toutes les N epochs

    # ── Sortie ───────────────────────────────────────────────────────────
    "output_dir"   : "results/bulk_finetune",
}

CIFAR10_CLASSES = [
    "avion", "auto", "oiseau", "chat", "cerf",
    "chien", "grenouille", "cheval", "bateau", "camion"
]

# ════════════════════════════════════════════════════════════════════════════
# Fonctions utilitaires
# ════════════════════════════════════════════════════════════════════════════

def per_class_accuracy(model, x, y, n_classes=10):
    preds = model.predict(x, verbose=0, batch_size=256).argmax(axis=1)
    return np.array([(preds[y == c] == c).mean() for c in range(n_classes)])

def compute_spike_basis(model, loss_fn, x_hvp, y_hvp, lanczos_m, n_spikes):
    """Retourne les top-n_spikes vecteurs propres (matrice n_params × n_spikes)."""
    from spectral_tools import HessianVectorProduct, StochasticLanczosQuadrature
    hvp = HessianVectorProduct(
        model=model, loss_fn=loss_fn,
        data_x=x_hvp, data_y=y_hvp, batch_size=None,
    )
    slq = StochasticLanczosQuadrature(
        hvp=hvp, n_params=hvp.n_params, m=20, k=1, sigma2=1e-5,
    )
    ritz_vals, ritz_vecs = slq.estimate_top_eigenvalues(
        m_lanczos=lanczos_m, verbose=False)
    n = min(n_spikes, ritz_vecs.shape[1])
    return ritz_vals[:n], ritz_vecs[:, :n]

def project_to_bulk(grad_flat, spike_vecs):
    """
    Projette un gradient plat ⊥ aux spikes.
    g_bulk = g - Q @ Q^T @ g   où Q = spike_vecs (n_params × n_spikes)
    """
    # Coefficients sur chaque spike : c = Q^T g
    coeffs = spike_vecs.T @ grad_flat       # (n_spikes,)
    # Composante spike : Q @ c
    spike_component = spike_vecs @ coeffs    # (n_params,)
    return grad_flat - spike_component


class BulkFineTuner:
    """
    Fine-tuning avec gradients projetés dans le bulk (⊥ spikes).
    """

    def __init__(self, model, loss_fn, x_train, y_train,
                 x_test, y_test, x_hvp, y_hvp, cfg):
        self.model   = model
        self.loss_fn = loss_fn
        self.x_train = x_train
        self.y_train = y_train
        self.x_test  = x_test
        self.y_test  = y_test
        self.x_hvp   = x_hvp
        self.y_hvp   = y_hvp
        self.cfg     = cfg
        os.makedirs(cfg["output_dir"], exist_ok=True)

    def run(self):
        cfg = self.cfg
        lr  = cfg["lr"]
        bs  = cfg["batch_size"]
        n_train = len(self.x_train)
        steps_per_epoch = n_train // bs
        log = []

        # Baseline
        acc_baseline = per_class_accuracy(self.model, self.x_test, self.y_test)
        acc_global_0 = self.model.evaluate(
            self.x_test, self.y_test, verbose=0, batch_size=256)[1]
        print(f"\n  Baseline  global={acc_global_0:.4f}  "
              f"std={np.std(acc_baseline):.4f}")
        for c, name in enumerate(CIFAR10_CLASSES):
            print(f"    {name:12s} : {acc_baseline[c]*100:.1f}%")

        # Calcul initial des spikes
        spike_vecs = None

        for epoch in range(1, cfg["epochs"] + 1):
            t0 = time.time()

            # ── Recalcul des spikes si nécessaire ─────────────────────────
            if spike_vecs is None or epoch % cfg["recompute_every"] == 1:
                print(f"\n  [spike] Calcul des {cfg['n_spikes']} vecteurs propres ...")
                ritz_vals, spike_vecs = compute_spike_basis(
                    self.model, self.loss_fn, self.x_hvp, self.y_hvp,
                    cfg["lanczos_m"], cfg["n_spikes"],
                )
                print(f"  [spike] λ top-3 : {ritz_vals[:3].round(1).tolist()}")

            # ── Mélange des données ───────────────────────────────────────
            rng = np.random.default_rng(cfg["seed"] + epoch)
            perm = rng.permutation(n_train)

            epoch_loss = 0.0
            for step in range(steps_per_epoch):
                idx = perm[step * bs : (step + 1) * bs]
                x_batch = tf.constant(self.x_train[idx])
                y_batch = tf.constant(self.y_train[idx])

                # Forward + backward
                with tf.GradientTape() as tape:
                    logits = self.model(x_batch, training=True)
                    loss = self.loss_fn(y_batch, logits)
                grads = tape.gradient(loss, self.model.trainable_variables)
                epoch_loss += float(loss)

                # Aplatir le gradient
                grad_flat = np.concatenate(
                    [g.numpy().flatten() for g in grads])

                # ── Projection dans le bulk ───────────────────────────────
                grad_bulk = project_to_bulk(grad_flat, spike_vecs)

                # Appliquer la mise à jour
                offset = 0
                for var in self.model.trainable_variables:
                    size = int(np.prod(var.shape))
                    update = grad_bulk[offset:offset + size].reshape(var.shape)
                    var.assign_sub(tf.constant(lr * update, dtype=var.dtype))
                    offset += size

                if (step + 1) % 100 == 0:
                    print(f"    epoch {epoch}  step {step+1}/{steps_per_epoch}  "
                          f"loss={float(loss):.4f}")

            # ── Évaluation fin d'epoch ────────────────────────────────────
            acc_new    = per_class_accuracy(self.model, self.x_test, self.y_test)
            acc_global = self.model.evaluate(
                self.x_test, self.y_test, verbose=0, batch_size=256)[1]
            cur_std    = float(np.std(acc_new))
            elapsed    = time.time() - t0
            avg_loss   = epoch_loss / steps_per_epoch

            print(f"\n  Epoch {epoch}/{cfg['epochs']}  loss={avg_loss:.4f}  "
                  f"global={acc_global:.4f}  std={cur_std:.4f}  ({elapsed:.0f}s)")
            header = "  " + "  ".join(f"{n[:4]:>5s}" for n in CIFAR10_CLASSES)
            vals   = "  " + "  ".join(f"{acc_new[c]*100:>5.1f}" for c in range(10))
            delts  = "  " + "  ".join(
                f"{(acc_new[c]-acc_baseline[c])*100:>+5.1f}" for c in range(10))
            print(header)
            print(vals)
            print(f"  Δbase: {delts}")

            log.append({
                "epoch"     : epoch,
                "loss"      : avg_loss,
                "acc_global": float(acc_global),
                "std"       : cur_std,
                "elapsed_s" : elapsed,
                **{CIFAR10_CLASSES[c]: float(acc_new[c]) for c in range(10)},
            })

        # Sauvegarde
        import pandas as pd
        pd.DataFrame(log).to_csv(
            os.path.join(cfg["output_dir"], "bulk_ft_log.csv"), index=False)
        print(f"\n  CSV sauvegardé dans {cfg['output_dir']}/")
        return log


# ════════════════════════════════════════════════════════════════════════════
# Point d'entrée
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if "--dry-run" in sys.argv:
        print("Dry run — vérification de la config")
        print(f"  Modèle  : {CONFIG['model_path']}")
        print(f"  Epochs  : {CONFIG['epochs']}")
        print(f"  LR      : {CONFIG['lr']}")
        print(f"  Spikes  : {CONFIG['n_spikes']}")
        print(f"  Batch   : {CONFIG['batch_size']}")
        print("OK — prêt à lancer sans --dry-run")
        sys.exit(0)

    print("[1] Chargement de CIFAR-10 ...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
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

    tuner = BulkFineTuner(
        model=model, loss_fn=loss_fn,
        x_train=x_train, y_train=y_train,
        x_test=x_test, y_test=y_test,
        x_hvp=x_hvp, y_hvp=y_hvp,
        cfg=CONFIG,
    )
    tuner.run()
