# Hessian Surgery: Class-Targeted Post-Hoc Rebalancing

**Rééquilibrage post-hoc des performances par classe via la géométrie spectrale de la Hessienne.**

> Hugo Vigna, CentraleSupélec (2026). Article disponible sur arXiv.

## Résumé

Les réseaux entraînés par ERM présentent des disparités inter-classes importantes. **Hessian Surgery (SS)** exploite la structure des *spikes* de la Hessienne (un spike par classe, cf. Papyan 2020) pour redistribuer la performance entre classes **sans réentraînement** : on optimise des perturbations de poids dans le sous-espace spike, sous contrainte de maintien de l'accuracy globale.

## Résultats principaux

### CIFAR-10 (ResNet-50)
| Méthode | Acc | σ | Δσ | worst-2 |
|---|---|---|---|---|
| Baseline | 84.9% | 8.57% | – | 68.6 / 69.0 |
| Focal Loss FT | 84.8% | 7.81% | −0.76 | 70.2 / 72.3 |
| τ-norm (τ=2) | 84.7% | 7.90% | −0.67 | 74.8 / 66.3 |
| **Hessian Surgery** | **85.1%** | **5.78%** | **−2.79** | **76.1 / 73.9** |

### ISIC-2019 (ResNet-50, 8 classes, dermato déséquilibré)
| Méthode | bal. acc | σ |
|---|---|---|
| Baseline (CE) | 37.5% | 23.6% |
| SS seule | 50.1% | 17.2% |
| FL seule | 49.2% | 20.8% |
| FL + SS | 51.5% | 16.4% |
| **CB + SS** | **58.0%** | **11.3%** |

### CIFAR-100 (Deflated Surgery, 45 spikes / 3 phases)
Réduction de σ et amélioration des classes faibles — voir Table 13-14 du papier et `results/deflated_surgery_cifar100/`.

## Structure du projet

```
projet_recherche/
├── README.md
│
├── spectral_tools.py                        # Lib bas niveau : HVP (Pearlmutter), Lanczos, SLQ
├── hessian_surgery.py                      # Classe canonique `HessianSurgery` (CIFAR-10, ablations)
├── isic_ss.py                               # Variante ISIC partagée (CE / FL / CB)
├── spike_optimizer.py                       # SS legacy CIFAR-10 (utilisée par comparison_focal_balanced, linearization_diagnostic)
├── spike_optimizer_cifar100.py              # SS spécialisée CIFAR-100 (utilisée par deflated_surgery)
│
├── train_*.py                               # Entraînement des baselines
├── focal_loss_*.py                          # Fine-tuning Focal Loss
├── class_balanced_isic2019.py               # Fine-tuning Class-Balanced
│
├── ss_cifar10.py                            # SS sur CIFAR-10 (final, omega homogeneous)
├── deflated_surgery_cifar100.py             # Deflated SS sur CIFAR-100
├── spike_optimizer_isic2019_ce.py           # SS sur baseline CE ISIC
├── spike_optimizer_isic2019_fl.py           # FL + SS sur ISIC
├── spike_optimizer_isic2019_cb.py           # CB + SS sur ISIC
├── ablation_omega_{cifar10,isic2019}.py     # Ablation p-rule (sqrt/linear/square)
├── bulk_finetune.py                         # Bulk-projected FT (Table 12)
│
├── run_analysis.py                          # Analyse spectrale complète
├── run_spectral_density_resnet50.py         # Densité spectrale CIFAR-10
├── spectral_density_isic2019.py             # Densité spectrale ISIC
├── compute_sensitivity_matrix.py            # Matrice de sensibilité S
├── compute_S_baseline_cifar10.py            # S baseline (avant SS)
├── linearization_diagnostic.py              # Diagnostic linéarisation
├── test_hvp_batch_ablation.py               # Robustesse HVP (Tables 3-5)
├── comparison_focal_balanced.py             # FL/CB sur CIFAR-10
├── compare_posthoc_{cifar10,isic2019}.py    # τ-norm, logit adjustment
├── eval_baseline_spiked_test.py             # Évaluation sur test held-out
│
├── plot_spectral_resnet50.py                # Figure 1 (densité)
├── plot_linearization.py                    # Figures linéarisation
├── plot_deflated_results.py                 # Figure CIFAR-100
│
├── download_isic2019.py                     # Téléchargement dataset ISIC
├── resplit_isic2019.py                      # Re-split train/val/test ISIC
│
├── resnet50_*.keras                         # Modèles entraînés (15 fichiers, ~1.6 Go)
│
└── results/
    ├── cifar10/         {ss, fl_ss, focal_loss, classic_ft, bulk_finetune, ablation_omega, ...}
    ├── cifar100/        {deflated_surgery}
    ├── isic2019/        {ss, ce_ss, fl_ss, cb_ss, focal_loss, class_balanced, ablation_omega, spectral_density}
    └── spectral/        {density_cifar10, sensitivity_matrix, linearization, hvp_ablation, directed_walk}
```

## Modèles (.keras)

| Fichier | Description |
|---|---|
| `resnet50_cifar10.keras` | Baseline CIFAR-10 (84.9%) |
| `resnet50_cifar10_ss.keras` | Après SS (homogeneous, 15 iter) |
| `resnet50_cifar10_fl_ss.keras` | FL + SS |
| `resnet50_cifar10_ss_omega_{linear,sqrt,square}.keras` | Ablation p-rule |
| `resnet50_cifar100.keras` | Baseline CIFAR-100 (60.3%) |
| `resnet50_isic2019.keras` | Baseline ISIC-2019 CE |
| `resnet50_isic2019_ss.keras` | SS seul ISIC |
| `resnet50_isic2019_ce_ss.keras` | SS sur baseline CE (batch 256) |
| `resnet50_isic2019_fl_ss.keras` | FL + SS |
| `resnet50_isic2019_cb_ss.keras` | CB + SS |
| `resnet50_isic2019_ss_omega_{linear,sqrt,square}.keras` | Ablation p-rule ISIC |

## Reproduction

### Prérequis
```bash
pip install -r requirements.txt   # versions exactes du papier
```
Python 3.12. Testé sur macOS Metal (M4) ; CPU et CUDA Linux fonctionnent
sans modification (Keras 3 backend TF gère le placement automatiquement).

### Datasets
- **CIFAR-10/100** : téléchargé automatiquement via `keras.datasets`
- **ISIC-2019** : `python download_isic2019.py` puis `python resplit_isic2019.py`

### Entraînement des baselines
```bash
python train_cifar10.py            # ~85% acc
python train_resnet50_cifar100.py  # ~60% acc
python train_resnet50_isic2019.py  # baseline CE
```

### Hessian Surgery
```bash
# CIFAR-10
python ss_cifar10.py

# CIFAR-100 (deflated, 3 phases)
python deflated_surgery_cifar100.py

# ISIC-2019
python spike_optimizer_isic2019_ce.py   # SS sur baseline CE
python focal_loss_isic2019.py           # FT focal loss
python spike_optimizer_isic2019_fl.py   # SS sur FL checkpoint
python class_balanced_isic2019.py       # FT class-balanced
python spike_optimizer_isic2019_cb.py   # SS sur CB checkpoint
```

### Ablations & analyse
```bash
python ablation_omega_cifar10.py        # p-rule (sqrt/linear/square)
python ablation_omega_isic2019.py
python run_spectral_density_resnet50.py # Figure 1
python compute_sensitivity_matrix.py    # Matrice S
python linearization_diagnostic.py      # Diagnostic linéarisation
python test_hvp_batch_ablation.py       # Robustesse HVP
python bulk_finetune.py                 # Table 12
```

### Comparaisons post-hoc
```bash
python compare_posthoc_cifar10.py       # τ-norm, logit adj
python compare_posthoc_isic2019.py
python comparison_focal_balanced.py     # FL/CB (Table 11)
```

## Utilisation programmatique

La classe `HessianSurgery` (dans `hessian_surgery.py`) encapsule toute la procédure. Signature :

```python
import tensorflow as tf
from hessian_surgery import HessianSurgery

# 1. Modèle + loss
model   = tf.keras.models.load_model("resnet50_cifar10.keras", compile=False)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

# 2. Splits (sensitivity / val / test) + mini-batch HVP
#    - x_sens, y_sens : sert à estimer la matrice de sensibilité S
#    - x_val,  y_val  : suivi pendant l'optimisation (rollback)
#    - x_test, y_test : évaluation finale held-out
#    - x_hvp,  y_hvp  : batch fixe pour Lanczos / HVP

# 3. Config (tous les hyperparams ; voir ss_cifar10.py pour un exemple complet)
cfg = {
    "n_hvp_samples"    : 128,
    "lanczos_m"        : 10,
    "n_spikes"         : 9,           # = nb_classes - 1 typiquement
    "n_iter"           : 15,
    "omega_mode"       : "homogeneous",  # ou "sqrt" / "linear" / "square"
    "max_degrade_total": 0.06,
    "max_degrade_iter" : 0.03,
    "alpha_max_init"   : 0.02,
    "alpha_min"        : 0.002,
    "beta_ema"         : 0.7,
    "rollback_std_tol" : 0.005,
    "rollback_drop_tol": 0.07,
    "output_dir"       : "results/cifar10/ss",
    "save_model"       : True,
    "model_out"        : "resnet50_cifar10_ss.keras",
    "seed"             : 0,
    "class_names"      : [...],
}

runner = HessianSurgery(
    model, loss_fn,
    x_sens, y_sens,
    x_val,  y_val,
    x_test, y_test,
    x_hvp,  y_hvp,
    cfg,
)

# Les kwargs de run() surchargent cfg sans le muter (utile pour des sweeps)
runner.run(n_iter=15, omega_mode="homogeneous")
```

Scripts d'exemple complets : `ss_cifar10.py`, `spike_optimizer_isic2019_ce.py`, `deflated_surgery_cifar100.py`.

## Algorithmes

| Algorithme | Fichier | Référence |
|---|---|---|
| Hessian-Vector Product (Pearlmutter) | `spectral_tools.py` | Pearlmutter (1994) |
| Lanczos avec ré-orthogonalisation | `spectral_tools.py` | Ghorbani et al. (2019) |
| Stochastic Lanczos Quadrature (SLQ) | `spectral_tools.py` | Ghorbani et al. (2019) |
| Hessian Surgery (CIFAR-10) | `hessian_surgery.py` | Ce papier (Algo. 2) |
| Hessian Surgery (ISIC-2019) | `isic_ss.py` | Ce papier (Algo. 2, variante déséquilibrée) |
| Deflated Surgery | `deflated_surgery_cifar100.py` | Ce papier (Algo. 3) |
| Bulk-projected fine-tuning | `bulk_finetune.py` | Ce papier |

## Références

- Ghorbani, Krishnan, Xiao (2019). *An Investigation into Neural Net Optimization via Hessian Eigenvalue Density.* ICML.
- Papyan (2020). *Traces of Class/Cross-Class Structure Pervade Deep Learning Spectra.* JMLR.
- Pearlmutter (1994). *Fast Exact Multiplication by the Hessian.* Neural Computation.
- Cui et al. (2019). *Class-Balanced Loss Based on Effective Number of Samples.* CVPR.
- Lin et al. (2017). *Focal Loss for Dense Object Detection.* ICCV.
