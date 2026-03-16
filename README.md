# Spectral Geometry of Deep Network Loss Landscapes

**Spectral Surgery : rééquilibrage post-hoc des performances par classe via la géométrie spectrale de la Hessienne.**

> Projet de recherche — Hugo Vigna, CentraleSupélec (2026)

## Résumé

Les réseaux de neurones profonds entraînés par ERM présentent des disparités significatives de performance entre classes. Ce projet exploite la structure spectrale de la Hessienne de la loss pour corriger ces déséquilibres **sans réentraînement**.

L'approche repose sur trois observations clés issues de la théorie des matrices aléatoires (RMT) :
1. Le spectre de la Hessienne se décompose en un **bulk** continu et des **spikes** isolés
2. Les spikes encodent la structure inter-classes (un spike par classe, cf. Papyan 2020)
3. Perturber les poids le long des eigenvectors spike permet de redistribuer la performance entre classes

**Spectral Surgery** (Algorithme 2 du papier) optimise itérativement des perturbations dans le sous-espace spike pour minimiser l'écart-type des accuracies par classe, sous contrainte de maintien de l'accuracy globale.

## Résultats principaux

### CIFAR-10 (ResNet-50)
| Métrique | Baseline | Après Surgery | Delta |
|----------|----------|---------------|-------|
| Accuracy globale | 84.8% | 84.7% | -0.1% |
| Std inter-classes | 8.57% | 5.52% | **-3.05 pp** |
| Pire classe (chat) | 68.6% | 76.1% | **+7.5 pp** |

### CIFAR-100 (ResNet-50, Deflated Surgery)
Résultats : voir Table 13-14 dans le papier et `results/deflated_surgery_cifar100/`.

## Structure du projet

```
├── article_hugo_vigna.pdf          # Article complet
│
├── spectral_tools.py               # Librairie : HVP, Lanczos, SLQ
├── spike_optimizer.py              # Spectral Surgery CIFAR-10
├── spike_optimizer_cifar100.py     # Utilitaires CIFAR-100
├── deflated_surgery_cifar100.py    # Deflated Surgery CIFAR-100 (4 phases)
│
├── run_analysis.py                 # Analyse spectrale complète (Tables 1-8)
├── run_spectral_density_resnet50.py # Densité spectrale (Figure 1)
├── comparison_focal_balanced.py    # Comparaison baselines (Table 11)
├── bulk_finetune.py                # Fine-tuning projeté dans le bulk (Table 12)
├── test_hvp_batch_ablation.py      # Robustesse HVP (Tables 3-5)
│
├── train_cifar10.py                # Entraînement ResNet-50 CIFAR-10
├── train_resnet50_cifar100.py      # Entraînement ResNet-50 CIFAR-100
│
├── plot_spectral_resnet50.py       # Génération Figure 1
├── plot_deflated_results.py        # Génération Figure 2
│
└── results/
    ├── spike_optimizer/            # Résultats CIFAR-10 Surgery
    ├── deflated_surgery_cifar100/  # Résultats CIFAR-100 Surgery
    ├── spectral_resnet50/          # Densité spectrale
    ├── comparison/                 # Comparaison Focal/CB
    ├── ablation_hvp/               # Ablation taille batch HVP
    └── directed_walk/              # Marche dirigée dans le bulk
```

## Algorithmes implémentés

| Algorithme | Fichier | Référence |
|------------|---------|-----------|
| Hessian-Vector Product (Pearlmutter) | `spectral_tools.py` | Pearlmutter (1994) |
| Lanczos avec ré-orthogonalisation | `spectral_tools.py` | Ghorbani et al. (2019) |
| Stochastic Lanczos Quadrature | `spectral_tools.py` | Ghorbani et al. (2019) |
| Spectral Surgery | `spike_optimizer.py` | Ce travail (Algo. 2) |
| Deflated Surgery | `deflated_surgery_cifar100.py` | Ce travail (Algo. 3) |
| Bulk-projected fine-tuning | `bulk_finetune.py` | Ce travail |

## Utilisation

### Prérequis
```bash
pip install tensorflow numpy pandas scipy matplotlib
```

### Modèles pré-entraînés
Les fichiers `.keras` (90-280 MB) ne sont pas inclus dans le repo. Pour reproduire :
```bash
python train_cifar10.py              # ~85% accuracy
python train_resnet50_cifar100.py    # ~60% accuracy
```

### Lancer Spectral Surgery
```bash
# CIFAR-10 (10 itérations, ~15 min CPU)
python spike_optimizer.py

# CIFAR-100 (4 phases deflated, ~2h CPU)
python deflated_surgery_cifar100.py
```

### Analyse spectrale
```bash
python run_analysis.py               # Tables 1-8
python run_spectral_density_resnet50.py  # Figure 1
```

## Références

- Ghorbani, Krishnan, Xiao (2019). *An Investigation into Neural Net Optimization via Hessian Eigenvalue Density.* ICML.
- Papyan (2020). *Traces of Class/Cross-Class Structure Pervade Deep Learning Spectra.* JMLR.
- Cui et al. (2019). *Class-Balanced Loss Based on Effective Number of Samples.* CVPR.
