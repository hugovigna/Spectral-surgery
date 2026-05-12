import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# CIFAR-10 baseline (val) and SS final (test)
cifar10_baseline = {
    'plane': 0.892,
    'car': 0.907,
    'bird': 0.817,
    'cat': 0.686,
    'deer': 0.852,
    'dog': 0.690,
    'frog': 0.906,
    'horse': 0.905,
    'ship': 0.931,
    'truck': 0.895,
}

cifar10_ss = {
    'plane': 0.881,
    'car': 0.924,
    'bird': 0.819,
    'cat': 0.744,
    'deer': 0.831,
    'dog': 0.735,
    'frog': 0.906,
    'horse': 0.891,
    'ship': 0.912,
    'truck': 0.897,
}

# ISIC-2019 baseline and SS
isic2019_baseline = {
    'MEL': 0.5206,
    'NV': 0.8586,
    'BCC': 0.6954,
    'AK': 0.3615,
    'BKL': 0.4975,
    'DF': 0.4444,
    'VASC': 0.2105,
    'SCC': 0.3511,
}

isic2019_ss = {
    'MEL': 0.51,
    'NV': 0.844,
    'BCC': 0.675,
    'AK': 0.369,
    'BKL': 0.528,
    'DF': 0.444,
    'VASC': 0.289,
    'SCC': 0.351,
}

# Sort by baseline accuracy (ascending)
cifar10_sorted = sorted(cifar10_baseline.items(), key=lambda x: x[1])
cifar10_classes = [x[0] for x in cifar10_sorted]
cifar10_baseline_vals = [x[1] for x in cifar10_sorted]
cifar10_ss_vals = [cifar10_ss[cls] for cls in cifar10_classes]

isic2019_sorted = sorted(isic2019_baseline.items(), key=lambda x: x[1])
isic2019_classes = [x[0] for x in isic2019_sorted]
isic2019_baseline_vals = [x[1] for x in isic2019_sorted]
isic2019_ss_vals = [isic2019_ss[cls] for cls in isic2019_classes]

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# CIFAR-10 plot
x = np.arange(len(cifar10_classes))
width = 0.35

bars1 = ax1.bar(x - width/2, cifar10_baseline_vals, width, label='Baseline', color='steelblue', alpha=0.8)
bars2 = ax1.bar(x + width/2, cifar10_ss_vals, width, label='Spectral Surgery', color='crimson', alpha=0.8)

ax1.set_xlabel('Class (sorted by baseline accuracy)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
ax1.set_title('CIFAR-10: Per-Class Accuracy (Baseline vs SS)', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(cifar10_classes, rotation=45, ha='right')
ax1.set_ylim([0.6, 1.0])
ax1.legend(fontsize=10, loc='lower right')
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars (smaller font)
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)

# ISIC-2019 plot
x = np.arange(len(isic2019_classes))

bars3 = ax2.bar(x - width/2, isic2019_baseline_vals, width, label='Baseline', color='steelblue', alpha=0.8)
bars4 = ax2.bar(x + width/2, isic2019_ss_vals, width, label='Spectral Surgery', color='crimson', alpha=0.8)

ax2.set_xlabel('Class (sorted by baseline accuracy)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
ax2.set_title('ISIC-2019: Per-Class Accuracy (Baseline vs SS)', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(isic2019_classes, rotation=45, ha='right')
ax2.set_ylim([0.0, 1.0])
ax2.legend(fontsize=10, loc='lower right')
ax2.grid(axis='y', alpha=0.3)

# Add value labels on bars (smaller font)
for bars in [bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('results/ss_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('results/ss_comparison.pdf', bbox_inches='tight')
print("Plots saved to results/ss_comparison.png and results/ss_comparison.pdf")
