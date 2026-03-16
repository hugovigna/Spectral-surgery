"""
run_analysis.py
---------------
Application de l'analyse spectrale de la Hessienne au ResNet50 entraîné sur CIFAR-10.

Pipeline :
  1. Chargement et prétraitement de CIFAR-10  (nativement 32×32 RGB, aucun resize)
  2. Chargement du modèle ResNet50
  3. Vérification du HVP (sanity check)
  4. Estimation des valeurs propres dominantes via Lanczos
  5. Estimation de la densité spectrale via Stochastic Lanczos Quadrature (SLQ)
  6. Calcul du critère de flatness  ζ = λ_max / λ_bulk
  7. Visualisation et sauvegarde des résultats

----------------------------------------------------------------------
Estimation du temps de run — ResNet50 (~23M params) sur CPU
----------------------------------------------------------------------
ResNet50 a ~2× plus de paramètres que ResNet18 → HVP ~2× plus lent.
Temps estimé par HVP (après compilation tf.function) :

  Mode             n_samples  m   k   HVPs totaux  Temps CPU estimé
  -------          ---------  --  --  -----------  ----------------
  fast (test)          128    20  20      400       ~  5.0 min
  full (précis)        512    90  10      900       ~ 60.0 min

  Note : slq_k=20 est nécessaire pour une estimation de ζ fiable.
  Avec k=3 la variance sur ζ est trop élevée (~44%) pour comparer
  deux minima. k=20 réduit l'écart-type d'un facteur √(20/3) ≈ 2.6×.

  Étape 4 (top eigenvalues, m=15, n=128)            ~   15 s

Notes :
  - Avec GPU CUDA (float32) : diviser par ~5–20×.
  - Avec Apple Metal (tensorflow-metal) : diviser par ~3–5×.
  - Augmenter n_samples améliore la précision (bruit de batch réduit)
    au prix d'un temps linéairement proportionnel.

Usage :
  python run_analysis.py               # mode complet (~60 min CPU)
  python run_analysis.py --mode fast   # test rapide  (~  1 min CPU)
"""

import os
import argparse
import time
from typing import Tuple
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")          # sans affichage interactif (adapté à un terminal)
import matplotlib.pyplot as plt

from spectral_tools import HessianVectorProduct, LanczosAlgorithm, StochasticLanczosQuadrature

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # supprime les logs TF verbeux


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG = {
    # Chemin du modèle sauvegardé (produit par train_cifar10.py)
    "model_path":  "resnet50_cifar10.keras",
    "output_dir":  "results",

    # Nombre d'échantillons utilisés pour les HVPs (sous-ensemble fixe).
    # 512 offre un bon compromis précision/vitesse.
    # Augmenter jusqu'à 2000+ pour plus de précision si GPU disponible.
    "n_samples": 512,

    # SLQ — mode complet
    "slq_m": 90,        # pas de Lanczos (ordre de la quadrature)
    "slq_k": 10,        # nombre de vecteurs sondes
    "slq_sigma2": 1e-5, # variance du noyau gaussien

    # Estimation des valeurs propres dominantes
    # RAM requise : n_params × top_m × 8 octets  (ex: 23M × 15 × 8 ≈ 2.7 Go)
    "top_m": 15,

    # Grille d'évaluation de la densité spectrale
    # t_max = 100 car CIFAR-10 est plus dur → spectre potentiellement plus étalé
    "t_min": -1.0,
    "t_max": 100.0,
    "t_npoints": 1000,
}

CONFIG_FAST = {
    **CONFIG,
    "n_samples": 128,   # ~1-2 s/HVP sur CPU
    "slq_m": 20,
    "slq_k": 20,   # k=20 requis pour variance acceptable sur ζ (std ~2.7× moindre vs k=3)
    "top_m": 10,
    "t_npoints": 300,
}


# ---------------------------------------------------------------------------
# 1. Chargement et prétraitement CIFAR-10
# ---------------------------------------------------------------------------

def load_cifar10(n_samples: int) -> dict:
    """
    Charge CIFAR-10 via tf.keras.datasets.

    CIFAR-10 est nativement (32, 32, 3) uint8 RGB :
      - Aucun resize nécessaire (contrairement à MNIST)
      - Aucune conversion de canal (déjà RGB)
      - Seule opération : normalisation en [0, 1]

    Retourne un dict avec les clés : x_train, y_train, x_test, y_test,
    x_sub (sous-ensemble de n_samples pour les HVPs).
    """
    print("[1] Chargement de CIFAR-10 ...")
    (x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = \
        tf.keras.datasets.cifar10.load_data()

    print(f"    x_train brut : {x_train_raw.shape}, dtype={x_train_raw.dtype}, "
          f"plage=[{x_train_raw.min()}, {x_train_raw.max()}]")

    # Normalisation [0, 255] → [0, 1] (pas de resize ni de conversion canal)
    x_train = x_train_raw.astype(np.float32) / 255.0
    x_test  = x_test_raw.astype(np.float32)  / 255.0

    # y a shape (N, 1) → (N,) pour sparse_categorical_crossentropy
    y_train = y_train_raw.squeeze().astype(np.int32)
    y_test  = y_test_raw.squeeze().astype(np.int32)

    print(f"    x_train final : {x_train.shape}, dtype={x_train.dtype}, "
          f"plage=[{x_train.min():.3f}, {x_train.max():.3f}]")

    # Sous-ensemble fixe (shuffle reproductible) pour les HVPs
    rng = np.random.default_rng(42)
    idx = rng.choice(len(x_train), size=min(n_samples, len(x_train)), replace=False)
    x_sub = x_train[idx]
    y_sub = y_train[idx]

    print(f"    Sous-ensemble HVP : {x_sub.shape[0]} images")

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_test":  x_test,
        "y_test":  y_test,
        "x_sub":   x_sub,
        "y_sub":   y_sub,
    }


# ---------------------------------------------------------------------------
# 2. Chargement du modèle
# ---------------------------------------------------------------------------

def load_model(path: str) -> tf.keras.Model:
    """Charge le modèle Keras et affiche un résumé succinct."""
    print("[2] Chargement du modèle ...")
    model = tf.keras.models.load_model(path)
    n_trainable = sum(int(tf.size(v)) for v in model.trainable_variables)
    print(f"    Architecture  : {model.name}")
    print(f"    Entrée        : {model.input_shape}")
    print(f"    Sortie        : {model.output_shape}")
    print(f"    Paramètres entraînables : {n_trainable:,}")
    return model


# ---------------------------------------------------------------------------
# 3. Évaluation du modèle sur le test set
# ---------------------------------------------------------------------------

def evaluate_model(model: tf.keras.Model, x_test: np.ndarray, y_test: np.ndarray):
    """Calcule et affiche la loss et l'accuracy sur le test set."""
    print("[3] Évaluation du modèle ...")
    results = model.evaluate(x_test, y_test, verbose=0, batch_size=256)
    metric_names = [m.name for m in model.metrics]
    for name, val in zip(metric_names, results):
        print(f"    {name}: {val:.4f}")
    return dict(zip(metric_names, results))


# ---------------------------------------------------------------------------
# 4. Sanity check du HVP
# ---------------------------------------------------------------------------

def hvp_sanity_check(hvp: HessianVectorProduct, tol: float = 1e-3):
    """
    Vérifie que le HVP est symétrique : (H @ u)^T v ≈ u^T (H @ v).
    Pour une Hessienne symétrique, l'égalité doit être vérifiée à la précision
    numérique près.
    """
    print("[4] Sanity check du HVP (symétrie) ...")
    rng = np.random.default_rng(0)
    u = rng.standard_normal(hvp.n_params)
    v = rng.standard_normal(hvp.n_params)
    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)

    hu = hvp.compute(u)
    hv = hvp.compute(v)

    lhs = float(np.dot(hu, v))    # (H @ u)^T v
    rhs = float(np.dot(u, hv))    # u^T (H @ v)

    err = abs(lhs - rhs) / (abs(lhs) + 1e-10)
    status = "OK" if err < tol else "ECHEC"
    print(f"    (H@u)^T v = {lhs:.6f}")
    print(f"    u^T (H@v) = {rhs:.6f}")
    print(f"    Erreur relative : {err:.2e}  [{status}]")
    return err < tol


# ---------------------------------------------------------------------------
# 5. Estimation des valeurs propres dominantes
# ---------------------------------------------------------------------------

def compute_top_eigenvalues(slq: StochasticLanczosQuadrature, top_m: int) -> np.ndarray:
    """
    Lance Lanczos avec ré-orthogonalisation pour estimer les top valeurs propres.
    Retourne les valeurs de Ritz triées décroissantes.
    """
    print(f"[5] Estimation des valeurs propres dominantes (m={top_m}) ...")
    t0 = time.time()
    ritz_vals, _ = slq.estimate_top_eigenvalues(m_lanczos=top_m, verbose=True)
    elapsed = time.time() - t0
    print(f"    Terminé en {elapsed:.1f} s")
    print(f"    Top-5 valeurs propres : {ritz_vals[:5]}")
    print(f"    λ_max ≈ {ritz_vals[0]:.4f}")
    return ritz_vals


# ---------------------------------------------------------------------------
# 6. Densité spectrale par SLQ
# ---------------------------------------------------------------------------

def compute_spectral_density(
    slq: StochasticLanczosQuadrature,
    t_min: float,
    t_max: float,
    t_npoints: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estime φ_σ(t) sur une grille [t_min, t_max].
    Retourne (t_values, density).
    """
    print(f"[6] Estimation de la densité spectrale "
          f"(m={slq.m}, k={slq.k}, σ²={slq.sigma2}) ...")
    t_values = np.linspace(t_min, t_max, t_npoints)

    t0 = time.time()
    density = slq.estimate_density(t_values, verbose=True)
    elapsed = time.time() - t0

    print(f"    Terminé en {elapsed:.1f} s  ({elapsed/60:.1f} min)")
    return t_values, density


# ---------------------------------------------------------------------------
# 7. Critère de flatness
# ---------------------------------------------------------------------------

def compute_flatness(
    slq: StochasticLanczosQuadrature,
    ritz_vals: np.ndarray,
    density: np.ndarray,
    t_values: np.ndarray,
):
    """Calcule et affiche le critère spectral de flatness ζ = λ_max / λ_bulk."""
    print("[7] Critère de flatness ...")
    lambda_max = float(ritz_vals[0])
    result = slq.compute_flatness_ratio(lambda_max, density, t_values)

    print(f"    λ_max   = {result['lambda_max']:.4f}")
    print(f"    λ_bulk  = {result['lambda_bulk']:.4f}  (médiane spectrale)")
    print(f"    ζ       = {result['ratio']:.2f}")
    print(f"    Minimum : {'POINTU (sharp)' if result['is_sharp'] else 'PLAT (flat)'}")
    return result


# ---------------------------------------------------------------------------
# 8. Visualisation
# ---------------------------------------------------------------------------

def plot_results(
    t_values: np.ndarray,
    density: np.ndarray,
    ritz_vals: np.ndarray,
    flatness: dict,
    output_dir: str,
):
    """Génère et sauvegarde les figures."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Figure gauche : densité spectrale ---
    ax = axes[0]
    ax.plot(t_values, density, color="steelblue", linewidth=1.5, label="φ̂_σ(t)")
    ax.axvline(flatness["lambda_max"], color="crimson",  linestyle="--",
               linewidth=1.2, label=f"λ_max = {flatness['lambda_max']:.2f}")
    ax.axvline(flatness["lambda_bulk"], color="darkorange", linestyle=":",
               linewidth=1.2, label=f"λ_bulk = {flatness['lambda_bulk']:.2f}")
    ax.set_xlabel("λ", fontsize=12)
    ax.set_ylabel("Densité spectrale estimée", fontsize=12)
    ax.set_title("Densité spectrale de la Hessienne\n(Stochastic Lanczos Quadrature)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # --- Figure droite : valeurs propres dominantes ---
    ax2 = axes[1]
    k_show = min(len(ritz_vals), 20)
    ax2.bar(range(1, k_show + 1), ritz_vals[:k_show], color="steelblue", edgecolor="navy")
    ax2.set_xlabel("Rang de la valeur propre", fontsize=12)
    ax2.set_ylabel("Valeur propre (Ritz)", fontsize=12)
    ax2.set_title(f"Top-{k_show} valeurs propres de Ritz\n(Lanczos avec ré-orthogonalisation)", fontsize=11)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "spectral_analysis.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"    Figure sauvegardée : {out_path}")


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyse spectrale Hessienne — ResNet50 / CIFAR-10")
    parser.add_argument("--mode", choices=["full", "fast"], default="full",
                        help="'full' : m=90, k=10 (précis, lent) | 'fast' : m=20, k=3 (test rapide)")
    args = parser.parse_args()

    cfg = CONFIG_FAST if args.mode == "fast" else CONFIG
    print(f"\n=== Analyse spectrale Hessienne — mode '{args.mode}' ===\n")

    # Localisation des fichiers par rapport à ce script
    base = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base, cfg["model_path"])
    output_dir = os.path.join(base, cfg["output_dir"])

    # -- Chargement des données et du modèle --
    cifar = load_cifar10(cfg["n_samples"])
    model = load_model(model_path)

    # -- Évaluation du modèle --
    metrics = evaluate_model(model, cifar["x_test"], cifar["y_test"])

    # -- Construction du HVP --
    # Utilise le sous-ensemble fixe de n_samples images.
    # loss_fn : SparseCategoricalCrossentropy (étiquettes entières, pas de one-hot).
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    hvp = HessianVectorProduct(
        model=model,
        loss_fn=loss_fn,
        data_x=cifar["x_sub"],
        data_y=cifar["y_sub"],
        batch_size=None,
    )

    print(f"\n    HVP configuré sur {cfg['n_samples']} images, "
          f"n_params = {hvp.n_params:,}\n")

    # -- Sanity check --
    hvp_sanity_check(hvp)

    # -- SLQ --
    slq = StochasticLanczosQuadrature(
        hvp=hvp,
        n_params=hvp.n_params,
        m=cfg["slq_m"],
        k=cfg["slq_k"],
        sigma2=cfg["slq_sigma2"],
    )

    # -- Top eigenvalues --
    ritz_vals = compute_top_eigenvalues(slq, top_m=cfg["top_m"])

    # -- Densité spectrale --
    t_values, density = compute_spectral_density(
        slq,
        t_min=cfg["t_min"],
        t_max=cfg["t_max"],
        t_npoints=cfg["t_npoints"],
    )

    # -- Flatness --
    flatness = compute_flatness(slq, ritz_vals, density, t_values)

    # -- Sauvegarde des résultats numériques (.npz) --
    os.makedirs(output_dir, exist_ok=True)
    np.savez(
        os.path.join(output_dir, "spectral_results.npz"),
        t_values=t_values,
        density=density,
        ritz_vals=ritz_vals,
        lambda_max=flatness["lambda_max"],
        lambda_bulk=flatness["lambda_bulk"],
        ratio=flatness["ratio"],
    )

    # -- Tableau synthèse CSV (top eigenvalues + flatness) --
    # Ligne de synthèse globale
    summary = {
        "mode":         args.mode,
        "n_samples":    cfg["n_samples"],
        "slq_m":        cfg["slq_m"],
        "slq_k":        cfg["slq_k"],
        "lambda_max":   flatness["lambda_max"],
        "lambda_bulk":  flatness["lambda_bulk"],
        "zeta":         flatness["ratio"],
        "is_sharp":     flatness["is_sharp"],
        "val_accuracy": metrics.get("compile_metrics", metrics.get("accuracy", None)),
        "val_loss":     metrics.get("loss", None),
    }
    df_summary = pd.DataFrame([summary])
    df_summary.to_csv(os.path.join(output_dir, "spectral_summary.csv"), index=False)

    # Tableau des top valeurs propres
    df_eigs = pd.DataFrame({
        "rank":       range(1, len(ritz_vals) + 1),
        "eigenvalue": ritz_vals,
    })
    df_eigs.to_csv(os.path.join(output_dir, "top_eigenvalues.csv"), index=False)

    print(f"\n    Résultats sauvegardés dans {output_dir}/")
    print(f"      spectral_results.npz  — densité + eigenvalues bruts")
    print(f"      spectral_summary.csv  — synthèse (λ_max, λ_bulk, ζ, acc)")
    print(f"      top_eigenvalues.csv   — top-{len(ritz_vals)} valeurs propres")

    # -- Visualisation --
    plot_results(t_values, density, ritz_vals, flatness, output_dir)

    print("\n=== Analyse terminée ===\n")


if __name__ == "__main__":
    main()
