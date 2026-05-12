# Spectral Surgery: Class-Targeted Post-Hoc Rebalancing

**Rééquilibrage post-hoc des performances par classe via la géométrie spectrale de la Hessienne.**

> Hugo Vigna, CentraleSupélec (2026). Article disponible sur arXiv.

## Résumé

Les réseaux entraînés par ERM présentent des disparités inter-classes importantes. **Spectral Surgery (SS)** exploite la structure des *spikes* de la Hessienne (un spike par classe, cf. Papyan 2020) pour redistribuer la performance entre classes **sans réentraînement** : on optimise des perturbations de poids dans le sous-espace spike, sous contrainte de maintien de l'accuracy globale.

## Résultats principaux

### CIFAR-10 (ResNet-50)
| Méthode | Acc | σ | Δσ | worst-2 |
|---|---|---|---|---|
| Baseline | 84.9% | 8.57% | – | 68.6 / 69.0 |
| Focal Loss FT | 84.8% | 7.81% | −0.76 | 70.2 / 72.3 |
| τ-norm (τ=2) | 84.7% | 7.90% | −0.67 | 74.8 / 66.3 |
| **Spectral Surgery** | **85.1%** | **5.78%** | **−2.79** | **76.1 / 73.9** |

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
├── article.tex                              # Source LaTeX du papier
├── README.md
│
├── spectral_tools.py                        # Lib core : HVP, Lanczos, SLQ
├── spike_optimizer.py                       # Lib SS (utilisée par les variantes)
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
├── spectral_blocks.py                       # Décomposition par bloc
├── compute_sensitivity_matrix.py            # Matrice de sensibilité S
├── compute_S_baseline_cifar10.py            # S baseline (avant SS)
├── linearization_diagnostic.py              # Diagnostic linéarisation
├── test_hvp_batch_ablation.py               # Robustesse HVP (Tables 3-5)
├── comparison_focal_balanced.py             # FL/CB sur CIFAR-10
├── compare_posthoc_{cifar10,isic2019}.py    # τ-norm, logit adjustment
├── tta_bootstrap_eval.py                    # TTA + bootstrap IC 95% (ISIC)
├── eval_baseline_spiked_test.py             # Évaluation sur test held-out
│
├── plot_spectral_resnet50.py                # Figure 1 (densité)
├── plot_linearization.py                    # Figures linéarisation
├── plot_deflated_results.py                 # Figure CIFAR-100
├── plot_ss_comparison.py                    # Bar plot acc par classe avant/après SS
│
├── download_isic2019.py                     # Téléchargement dataset ISIC
├── resplit_isic2019.py                      # Re-split train/val/test ISIC
│
├── resnet50_*.keras                         # Modèles entraînés (15 fichiers, ~1.6 Go)
│
└── results/
    ├── cifar10/         {ss, fl_ss, focal_loss, classic_ft, bulk_finetune, ablation_omega, ...}
    ├── cifar100/        {deflated_surgery}
    ├── isic2019/        {ss, ce_ss, fl_ss, cb_ss, focal_loss, class_balanced, ablation_omega, tta_bootstrap, spectral_density}
    └── spectral/        {density_cifar10, sensitivity_matrix, linearization, hvp_ablation, directed_walk, blocks}
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
pip install tensorflow keras numpy pandas scipy matplotlib scikit-learn
```

### Datasets
- **CIFAR-10/100** : téléchargé automatiquement via `keras.datasets`
- **ISIC-2019** : `python download_isic2019.py` puis `python resplit_isic2019.py`

### Entraînement des baselines
```bash
python train_cifar10.py            # ~85% acc
python train_resnet50_cifar100.py  # ~60% acc
python train_resnet50_isic2019.py  # baseline CE
```

### Spectral Surgery
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

## Algorithmes

| Algorithme | Fichier | Référence |
|---|---|---|
| Hessian-Vector Product (Pearlmutter) | `spectral_tools.py` | Pearlmutter (1994) |
| Lanczos avec ré-orthogonalisation | `spectral_tools.py` | Ghorbani et al. (2019) |
| Stochastic Lanczos Quadrature (SLQ) | `spectral_tools.py` | Ghorbani et al. (2019) |
| Spectral Surgery | `spike_optimizer.py` | Ce papier (Algo. 2) |
| Deflated Surgery | `deflated_surgery_cifar100.py` | Ce papier (Algo. 3) |
| Bulk-projected fine-tuning | `bulk_finetune.py` | Ce papier |

## Références

- Ghorbani, Krishnan, Xiao (2019). *An Investigation into Neural Net Optimization via Hessian Eigenvalue Density.* ICML.
- Papyan (2020). *Traces of Class/Cross-Class Structure Pervade Deep Learning Spectra.* JMLR.
- Pearlmutter (1994). *Fast Exact Multiplication by the Hessian.* Neural Computation.
- Cui et al. (2019). *Class-Balanced Loss Based on Effective Number of Samples.* CVPR.
- Lin et al. (2017). *Focal Loss for Dense Object Detection.* ICCV.
