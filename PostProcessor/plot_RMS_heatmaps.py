import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# Heatmaps of the simulated Force RMS over the (Eta_Max, a1) grid, one panel per strain
# step, plus the error against the experimental RMS (run compute_RMS.py first).
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(ROOT_DIR, "Samples")
EXP_FILE = os.path.join(ROOT_DIR, "BaselineFiles", "ForceRMS_experiment.csv")
out_dir = os.path.dirname(os.path.abspath(__file__))

strains = np.array([5, 10, 15, 20, 25])

# rms_summary.csv: sample, Eta_Max, a1, rms_5pct_uN, ..., rms_25pct_uN
summary = np.genfromtxt(os.path.join(SAMPLES_DIR, "rms_summary.csv"), delimiter=",",
                        names=True, dtype=None, encoding=None)

# ForceRMS_experiment.csv: strain (%), Force RMS (µN), SD (µN)
exp = np.loadtxt(EXP_FILE, delimiter=",", skiprows=1)
exp_rms = np.array([exp[exp[:, 0] == s, 1][0] for s in strains])

# Place every sample on the full grid; samples that have not finished stay NaN
eta_vals = np.unique(summary["Eta_Max"])
a1_vals = np.unique(summary["a1"])
i_eta = np.searchsorted(eta_vals, summary["Eta_Max"])
i_a1 = np.searchsorted(a1_vals, summary["a1"])

rms_grid = np.full((len(strains), len(eta_vals), len(a1_vals)), np.nan)
for k, s in enumerate(strains):
    rms_grid[k, i_eta, i_a1] = summary[f"rms_{s}pct_uN"]

err_grid = rms_grid - exp_rms[:, None, None]    # signed error, sim - exp (µN)
mse_grid = np.mean(err_grid**2, axis=0)         # NaN wherever any strain is missing


def draw_heatmap(ax, grid, cmap, norm, fmt):
    """Draw one grid (rows Eta_Max, cols a1); missing cells are hatched, values annotated."""
    cmap = plt.get_cmap(cmap).copy()
    cmap.set_bad("white")
    im = ax.imshow(np.ma.masked_invalid(grid), origin="lower", aspect="auto", cmap=cmap, norm=norm)

    # Hatch the samples with no result so they don't read as a low value
    for r, c in zip(*np.where(np.isnan(grid))):
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, hatch="///",
                                   edgecolor="0.75", linewidth=0))

    # Value labels, dark text on light cells and white text on dark cells
    for r, c in zip(*np.where(~np.isnan(grid))):
        lum = np.dot(im.cmap(im.norm(grid[r, c]))[:3], [0.299, 0.587, 0.114])
        ax.text(c, r, format(grid[r, c], fmt), ha="center", va="center", fontsize=6,
                color="white" if lum < 0.5 else "black")

    ax.set_xticks(range(len(a1_vals)), [f"{v:g}" for v in a1_vals], fontsize=8)
    ax.set_yticks(range(len(eta_vals)), [f"{v:.0f}" for v in eta_vals], fontsize=8)
    ax.set_xlabel("$a_1$", fontsize=11)
    ax.set_ylabel(r"$\eta_{max}$ (Pa)", fontsize=11)
    return im


# Figure 1: simulated RMS at each strain, one shared color scale so panels are comparable
fig, axes = plt.subplots(1, len(strains), figsize=(5 * len(strains), 4.8), constrained_layout=True)
norm = plt.Normalize(np.nanmin(rms_grid), np.nanmax(rms_grid))
for k, (ax, s) in enumerate(zip(axes, strains)):
    im = draw_heatmap(ax, rms_grid[k], "Blues", norm, ".3f")
    ax.set_title(f"{s}% strain  (exp: {exp_rms[k]:.3f} µN)", fontsize=12)
fig.colorbar(im, ax=axes, shrink=0.9, label="Simulated Force RMS (µN)")
fig.savefig(os.path.join(out_dir, "rms_heatmaps.png"), dpi=450)
plt.close(fig)

# Figure 2: signed error per strain (diverging, centered at 0) + MSE over all strains
fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
lim = np.nanmax(np.abs(err_grid))
err_norm = TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim)
for k, (ax, s) in enumerate(zip(axes.flat, strains)):
    im = draw_heatmap(ax, err_grid[k], "RdBu_r", err_norm, ".2f")
    ax.set_title(f"{s}% strain: sim − exp", fontsize=12)
fig.colorbar(im, ax=axes[:, :2], shrink=0.6, label="Force RMS error, sim − exp (µN)")

ax = axes.flat[-1]
im = draw_heatmap(ax, mse_grid, "Purples", plt.Normalize(np.nanmin(mse_grid), np.nanmax(mse_grid)), ".3f")
r, c = np.unravel_index(np.nanargmin(mse_grid), mse_grid.shape)
ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=2))
ax.set_title(f"MSE over 5–25%  (best: $\\eta_{{max}}$={eta_vals[r]:.0f}, $a_1$={a1_vals[c]:g})", fontsize=12)
fig.colorbar(im, ax=ax, shrink=0.9, label="MSE (µN²)")
fig.savefig(os.path.join(out_dir, "rms_error_heatmaps.png"), dpi=450)
plt.close(fig)

print(f"best fit: Eta_Max = {eta_vals[r]:.1f} Pa, a1 = {a1_vals[c]:g}, MSE = {mse_grid[r, c]:.4g} µN^2")
