"""Génère 2 plots compacts pour le diagnostic de linéarisation (CIFAR-10)."""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit

CSV = "results/spectral/linearization/linearization_sweep.csv"
OUT = "results/spectral/linearization"

df = pd.read_csv(CSV).sort_values("alpha_norm").reset_index(drop=True)
a = df["alpha_norm"].values
err_abs = df["lin_err_abs"].values
pred = df["pred_norm"].values        # ‖Δacc_predicted‖ = ‖S^T α‖
obs = df["obs_norm"].values          # ‖Δacc_measured‖

# Régression log-log : log(err_abs) ~ p · log(‖α‖) + const
mask = (a > 0) & (err_abs > 0)
log_a = np.log(a[mask])
log_e = np.log(err_abs[mask])
p_exp, log_c, r, p_val, se = stats.linregress(log_a, log_e)
c_pref = np.exp(log_c)
print(f"Pure power-law:  err ≈ {c_pref:.3f} · ‖α‖^{p_exp:.2f}   (R²={r**2:.3f})")

# Modèle additif: err = c + b · ‖α‖^d
def model(x, c, b, d):
    return c + b * np.power(x, d)
popt, pcov = curve_fit(model, a[mask], err_abs[mask], p0=[1e-2, 1.0, 1.0],
                       bounds=([0, 0, 0.1], [1.0, 1e3, 4.0]), maxfev=20000)
c_add, b_add, d_add = popt
err_pred = model(a[mask], *popt)
ss_res = np.sum((err_abs[mask] - err_pred)**2)
ss_tot = np.sum((err_abs[mask] - err_abs[mask].mean())**2)
r2_add = 1.0 - ss_res/ss_tot
print(f"Additive model:  err ≈ {c_add:.4f} + {b_add:.3f}·‖α‖^{d_add:.2f}   (R²={r2_add:.3f})")

plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.8})

# ── Plot A : Δacc predicted vs measured, vs ‖α‖ ────────────────────────
# Affine fits: y = a0 + b0 · ‖α‖
b_pred, a_pred = np.polyfit(a, pred, 1)
b_obs,  a_obs  = np.polyfit(a, obs,  1)
ss_res_obs = np.sum((obs - (a_obs + b_obs*a))**2)
ss_tot_obs = np.sum((obs - obs.mean())**2)
r2_obs = 1.0 - ss_res_obs/ss_tot_obs
print(f"Affine pred:  pred = {a_pred:.3e} + {b_pred:.3f}·‖α‖")
print(f"Affine meas:  obs  = {a_obs:.3e} + {b_obs:.3f}·‖α‖   R²={r2_obs:.3f}")

fig, ax = plt.subplots(figsize=(5.0, 3.6))
ax.plot(a, pred, "o", color="#1f77b4", ms=4,
        label=r"$\|\Delta\mathrm{acc}_{\mathrm{predicted}}\|_2$")
ax.plot(a, obs, "s", color="#d62728", ms=4,
        label=r"$\|\Delta\mathrm{acc}_{\mathrm{measured}}\|_2$")
xx = np.linspace(0, a.max(), 200)
ax.plot(xx, a_pred + b_pred*xx, "-", color="#1f77b4", lw=1.0, alpha=0.7,
        label=fr"affine: ${b_pred:.2f}\,\|\alpha\|$")
ax.plot(xx, a_obs + b_obs*xx, "-", color="#d62728", lw=1.0, alpha=0.7,
        label=fr"affine: ${a_obs:.2f}+{b_obs:.2f}\,\|\alpha\|$ ($R^2={r2_obs:.2f}$)")
ax.set_xlabel(r"$\|\alpha\|_2$")
ax.set_ylabel(r"Magnitude of per-class accuracy change")
ax.set_title("Predicted vs.\\ measured perturbation magnitude")
ax.grid(alpha=0.3)
ax.legend(fontsize=8, frameon=False)
plt.tight_layout()
plt.savefig(f"{OUT}/linearization_norms.pdf", bbox_inches="tight")
plt.savefig(f"{OUT}/linearization_norms.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Plot B : log-log fit pour identifier l'exposant ────────────────────
fig, ax = plt.subplots(figsize=(5.0, 3.6))
ax.loglog(a, err_abs, "o", color="#2ca02c", ms=5, mec="white", mew=0.5,
          label="measured", zorder=3)
xx = np.linspace(a.min(), a.max(), 200)
ax.loglog(xx, c_pref * xx**p_exp, "--", color="gray", lw=1.0,
          label=fr"pure: ${c_pref:.2f}\,\|\alpha\|^{{{p_exp:.2f}}}$  ($R^2={r**2:.2f}$)", zorder=2)
ax.loglog(xx, model(xx, *popt), "-", color="black", lw=1.0,
          label=fr"additive: ${c_add:.3f}+{b_add:.2f}\,\|\alpha\|^{{{d_add:.2f}}}$  ($R^2={r2_add:.2f}$)", zorder=2)
ax.set_xlabel(r"$\|\alpha\|_2$")
ax.set_ylabel(r"$\|\Delta\mathrm{acc}_{\mathrm{predicted}} - \Delta\mathrm{acc}_{\mathrm{measured}}\|_2$")
ax.set_title("Linearization error (log--log)")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=9, frameon=False, loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUT}/linearization_quadratic.pdf", bbox_inches="tight")
plt.savefig(f"{OUT}/linearization_quadratic.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"Saved → {OUT}/linearization_norms.{{pdf,png}}")
print(f"Saved → {OUT}/linearization_quadratic.{{pdf,png}}")
