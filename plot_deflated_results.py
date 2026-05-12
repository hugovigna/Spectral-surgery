"""
plot_deflated_results.py
Generate summary PNGs for Deflated Hessian Surgery on CIFAR-100.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = "results/cifar100/deflated_surgery"

# ── Load all phase logs ────────────────────────────────────────────────
phases_data = []
for p in range(1, 8):
    csv_path = os.path.join(OUTPUT_DIR, f"phase{p}", "iteration_log.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df["phase"] = p
        phases_data.append(df)

all_df = pd.concat(phases_data, ignore_index=True)

# Original accuracy (baseline)
acc_original = np.load(os.path.join(OUTPUT_DIR, "acc_original.npy"))

# Final accuracy = last iteration of last available phase
last_phase = phases_data[-1]
last_row = last_phase.iloc[-1]

# 100 CIFAR-100 class columns
class_cols = [c for c in all_df.columns if c not in [
    "iteration", "acc_global", "std", "std_ema", "alpha_max",
    "lambda_max", "alpha_norm", "rolled_back", "elapsed_s", "phase"
]]

acc_final = np.array([last_row[c] for c in class_cols])

# ════════════════════════════════════════════════════════════════════════
# PNG 1: alpha_max evolution, perturbation budget, std, global accuracy
# ════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Deflated Hessian Surgery — CIFAR-100 (4 phases, $\\alpha_{max} \\propto 1/\\sqrt{\\lambda_{max}}$)",
             fontsize=14, fontweight="bold")

all_df["global_iter"] = range(1, len(all_df) + 1)
phase_colors = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 4: "#d62728",
                5: "#9467bd", 6: "#8c564b", 7: "#e377c2"}

# Panel 1: alpha_max per iteration
ax = axes[0, 0]
for phase in all_df["phase"].unique():
    mask = all_df["phase"] == phase
    ax.plot(all_df.loc[mask, "global_iter"], all_df.loc[mask, "alpha_max"],
            "o-", color=phase_colors.get(phase, "gray"), ms=5, label=f"Phase {phase}")
ax.set_xlabel("Global iteration")
ax.set_ylabel("$\\alpha_{max}$")
ax.set_title("$\\alpha_{max}$ evolution")
ax.legend(fontsize=7, ncol=2)
ax.grid(alpha=0.3)

# Panel 2: Perturbation budget (should be ~constant with perfect adaptation)
ax = axes[0, 1]
all_df["budget"] = all_df["alpha_max"] * np.sqrt(all_df["lambda_max"])
for phase in all_df["phase"].unique():
    mask = all_df["phase"] == phase
    ax.plot(all_df.loc[mask, "global_iter"], all_df.loc[mask, "budget"],
            "o-", color=phase_colors.get(phase, "gray"), ms=5, label=f"Phase {phase}")
ax.set_xlabel("Global iteration")
ax.set_ylabel("$\\alpha_{max} \\times \\sqrt{\\lambda_{max}}$")
ax.set_title("Perturbation budget ($\\alpha_{max} \\times \\sqrt{\\lambda_{max}}$)")
ax.axhline(0.025 * np.sqrt(600), color="crimson", ls=":", lw=1, alpha=0.6, label="Ref (0.61)")
ax.legend(fontsize=7, ncol=2)
ax.grid(alpha=0.3)

# Panel 3: Inter-class std
ax = axes[1, 0]
for phase in all_df["phase"].unique():
    mask = all_df["phase"] == phase
    ax.plot(all_df.loc[mask, "global_iter"], all_df.loc[mask, "std"] * 100,
            "o-", color=phase_colors.get(phase, "gray"), ms=5, label=f"Phase {phase}")
ax.axhline(np.std(acc_original) * 100, color="crimson", ls=":", lw=1.5, alpha=0.7, label="Baseline")
ax.set_xlabel("Global iteration")
ax.set_ylabel("Inter-class std (%)")
ax.set_title("Inter-class standard deviation ($\\sigma$)")
ax.legend(fontsize=7, ncol=2)
ax.grid(alpha=0.3)

# Panel 4: Global accuracy
ax = axes[1, 1]
for phase in all_df["phase"].unique():
    mask = all_df["phase"] == phase
    ax.plot(all_df.loc[mask, "global_iter"], all_df.loc[mask, "acc_global"] * 100,
            "o-", color=phase_colors.get(phase, "gray"), ms=5, label=f"Phase {phase}")
ax.axhline(np.mean(acc_original) * 100, color="crimson", ls=":", lw=1.5, alpha=0.7, label="Baseline")
ax.set_xlabel("Global iteration")
ax.set_ylabel("Global accuracy (%)")
ax.set_title("Global accuracy")
ax.legend(fontsize=7, ncol=2)
ax.grid(alpha=0.3)

plt.tight_layout()
path1 = os.path.join(OUTPUT_DIR, "deflated_surgery_alpha_evolution.png")
plt.savefig(path1, dpi=150)
plt.close()
print(f"[1] Saved: {path1}")

# ════════════════════════════════════════════════════════════════════════
# PNG 2: Per-class accuracy (baseline vs final), sorted worst to best
# ════════════════════════════════════════════════════════════════════════

sort_idx = np.argsort(acc_original)
sorted_classes = [class_cols[i] for i in sort_idx]
sorted_initial = acc_original[sort_idx]
sorted_final = acc_final[sort_idx]

fig, ax = plt.subplots(figsize=(20, 8))

x = np.arange(len(sorted_classes))
width = 0.35

bars_init = ax.bar(x - width/2, sorted_initial * 100, width, label="Baseline",
                   color="#bdbdbd", edgecolor="gray", linewidth=0.5)
bars_final = ax.bar(x + width/2, sorted_final * 100, width, label="After surgery (phases 1-4)",
                    color="#4292c6", edgecolor="#2171b5", linewidth=0.5)

for i, (init, final) in enumerate(zip(sorted_initial, sorted_final)):
    if final > init + 0.02:
        bars_final[i].set_facecolor("#2ca02c")
        bars_final[i].set_edgecolor("#1a7a1a")
    elif final < init - 0.02:
        bars_final[i].set_facecolor("#d62728")
        bars_final[i].set_edgecolor("#a01010")

ax.set_xlabel("Classes (sorted by baseline accuracy, worst to best)", fontsize=11)
ax.set_ylabel("Accuracy (%)", fontsize=11)
ax.set_title(f"Per-class accuracy — Baseline vs After Deflated Surgery (phases 1-4, spikes 1-60)\n"
             f"Global: {np.mean(acc_original)*100:.1f}% \u2192 {np.mean(acc_final)*100:.1f}%  |  "
             f"Std: {np.std(acc_original)*100:.2f}% \u2192 {np.std(acc_final)*100:.2f}%  |  "
             f"Green = improved > 2%, Red = degraded > 2%",
             fontsize=12, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(sorted_classes, rotation=90, fontsize=6)
ax.legend(fontsize=10, loc="upper left")
ax.set_ylim(0, 100)
ax.grid(axis="y", alpha=0.3)

ax.axhline(np.mean(acc_original) * 100, color="gray", ls="--", lw=0.8, alpha=0.5)
ax.axhline(np.mean(acc_final) * 100, color="#4292c6", ls="--", lw=0.8, alpha=0.5)

plt.tight_layout()
path2 = os.path.join(OUTPUT_DIR, "deflated_surgery_class_accuracy.png")
plt.savefig(path2, dpi=150)
plt.close()
print(f"[2] Saved: {path2}")

# ════════════════════════════════════════════════════════════════════════
# PNG 3: Same bar chart but with end-of-phase-3 results (best std)
# ════════════════════════════════════════════════════════════════════════

phase3_df = phases_data[2]  # index 2 = phase 3
phase3_last = phase3_df.iloc[-1]
acc_phase3 = np.array([phase3_last[c] for c in class_cols])
sorted_phase3 = acc_phase3[sort_idx]

fig, ax = plt.subplots(figsize=(20, 8))

x = np.arange(len(sorted_classes))
width = 0.35

bars_init = ax.bar(x - width/2, sorted_initial * 100, width, label="Baseline",
                   color="#bdbdbd", edgecolor="gray", linewidth=0.5)
bars_p3 = ax.bar(x + width/2, sorted_phase3 * 100, width, label="After phase 3 (spikes 1-45)",
                 color="#4292c6", edgecolor="#2171b5", linewidth=0.5)

for i, (init, final) in enumerate(zip(sorted_initial, sorted_phase3)):
    if final > init + 0.02:
        bars_p3[i].set_facecolor("#2ca02c")
        bars_p3[i].set_edgecolor("#1a7a1a")
    elif final < init - 0.02:
        bars_p3[i].set_facecolor("#d62728")
        bars_p3[i].set_edgecolor("#a01010")

ax.set_xlabel("Classes (sorted by baseline accuracy, worst to best)", fontsize=11)
ax.set_ylabel("Accuracy (%)", fontsize=11)
ax.set_title(f"Per-class accuracy — Baseline vs After Phase 3 (best $\\sigma$, spikes 1-45)\n"
             f"Global: {np.mean(acc_original)*100:.1f}% \u2192 {np.mean(acc_phase3)*100:.1f}%  |  "
             f"Std: {np.std(acc_original)*100:.2f}% \u2192 {np.std(acc_phase3)*100:.2f}%  |  "
             f"Green = improved > 2%, Red = degraded > 2%",
             fontsize=12, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(sorted_classes, rotation=90, fontsize=6)
ax.legend(fontsize=10, loc="upper left")
ax.set_ylim(0, 100)
ax.grid(axis="y", alpha=0.3)

ax.axhline(np.mean(acc_original) * 100, color="gray", ls="--", lw=0.8, alpha=0.5)
ax.axhline(np.mean(acc_phase3) * 100, color="#4292c6", ls="--", lw=0.8, alpha=0.5)

plt.tight_layout()
path3 = os.path.join(OUTPUT_DIR, "deflated_surgery_class_accuracy_phase3.png")
plt.savefig(path3, dpi=150)
plt.close()
print(f"[3] Saved: {path3}")

# ── Summary stats ─────────────────────────────────────────────────────
delta_p3 = sorted_phase3 - sorted_initial
n_imp3 = (delta_p3 > 0.02).sum()
n_deg3 = (delta_p3 < -0.02).sum()
n_stab3 = len(delta_p3) - n_imp3 - n_deg3
print(f"\n  Phase 3 summary:")
print(f"    Global: {np.mean(acc_original)*100:.1f}% -> {np.mean(acc_phase3)*100:.1f}%")
print(f"    Std:    {np.std(acc_original)*100:.2f}% -> {np.std(acc_phase3)*100:.2f}%")
print(f"    Improved (>2%): {n_imp3}  Degraded (<-2%): {n_deg3}  Stable: {n_stab3}")
print(f"    Best:  {sorted_classes[np.argmax(delta_p3)]} ({delta_p3.max()*100:+.1f}%)")
print(f"    Worst: {sorted_classes[np.argmin(delta_p3)]} ({delta_p3.min()*100:+.1f}%)")

delta = sorted_final - sorted_initial
n_improved = (delta > 0.02).sum()
n_degraded = (delta < -0.02).sum()
n_stable = len(delta) - n_improved - n_degraded

print(f"\n  Final summary (phase 4):")
print(f"    Baseline: global={np.mean(acc_original)*100:.1f}%  std={np.std(acc_original)*100:.2f}%")
print(f"    Final:    global={np.mean(acc_final)*100:.1f}%  std={np.std(acc_final)*100:.2f}%")
print(f"    Improved (>2%): {n_improved}  Degraded (<-2%): {n_degraded}  Stable: {n_stable}")
print(f"    Best:  {sorted_classes[np.argmax(delta)]} ({delta.max()*100:+.1f}%)")
print(f"    Worst: {sorted_classes[np.argmin(delta)]} ({delta.min()*100:+.1f}%)")
