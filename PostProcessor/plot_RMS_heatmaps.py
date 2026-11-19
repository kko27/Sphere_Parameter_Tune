import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# Figure font sizes, used by every figure below
plt.rcParams.update({"xtick.labelsize": 14, "ytick.labelsize": 14, "axes.labelsize": 16,
                     "legend.fontsize": 14, "legend.title_fontsize": 14, "axes.titlesize": 18})

# Heatmaps of the simulated Force RMS over the (Eta_Max, a1) grid, one panel per strain
# step, plus the error against the experimental RMS (run compute_RMS.py first).
# Usage: python PostProcessor/plot_RMS_heatmaps.py      (HOLDOUT=0 python ... for the full 5-25% fit)
#
# HOLDOUT: the MSE (and best fit) uses 5-20% only and 25% is the validation strain, scored by its
# squared error. Holdout figures go to PostProcessor/holdout_25, full-fit figures to PostProcessor/.
HOLDOUT = os.environ.get("HOLDOUT", "1") != "0"
VAL_STRAIN = 25              # % strain held out when HOLDOUT is on

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(ROOT_DIR, "Samples")
EXP_FILE = os.path.join(ROOT_DIR, "BaselineFiles", "ForceRMS_experiment.csv")
out_dir = os.path.dirname(os.path.abspath(__file__))
if HOLDOUT:
    out_dir = os.path.join(out_dir, f"holdout_{VAL_STRAIN}")
os.makedirs(out_dir, exist_ok=True)

strains = np.array([5, 10, 15, 20, 25])
val_mask = (strains == VAL_STRAIN) if HOLDOUT else np.zeros(len(strains), bool)
train_mask = ~val_mask
train_label = f"{strains[train_mask].min()}–{strains[train_mask].max()}%"

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
mse_grid = np.mean(err_grid[train_mask]**2, axis=0)    # training MSE, NaN wherever a strain is missing
val_grid = err_grid[val_mask][0]**2 if HOLDOUT else None   # squared error at the validation strain


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
        ax.text(c, r, format(grid[r, c], fmt), ha="center", va="center", fontsize=8,
                color="white" if lum < 0.5 else "black")

    ax.set_xticks(range(len(a1_vals)), [f"{v:g}" for v in a1_vals])
    ax.set_yticks(range(len(eta_vals)), [f"{v:.0f}" for v in eta_vals])
    ax.set_xlabel("$a_1$")
    ax.set_ylabel(r"$\eta_{max}$ (Pa)")
    return im


def two_three_axes(fig):
    """Five equal panels: 2 centred on the top row, 3 on the bottom row."""
    gs = fig.add_gridspec(2, 6)
    return [fig.add_subplot(gs[0, 1:3]), fig.add_subplot(gs[0, 3:5]),
            fig.add_subplot(gs[1, 0:2]), fig.add_subplot(gs[1, 2:4]), fig.add_subplot(gs[1, 4:6])]


# Figure 1: simulated RMS at each strain, one shared color scale so panels are comparable
fig = plt.figure(figsize=(19, 11), constrained_layout=True)
axes = two_three_axes(fig)
norm = plt.Normalize(np.nanmin(rms_grid), np.nanmax(rms_grid))
for k, (ax, s) in enumerate(zip(axes, strains)):
    im = draw_heatmap(ax, rms_grid[k], "Blues", norm, ".3f")
    ax.set_title(f"{s}% strain  (exp: {exp_rms[k]:.3f} µN)")
    if k not in (0, 2):   # η_max label only on the first panel of each row
        ax.set_ylabel("")
fig.colorbar(im, ax=axes, shrink=0.6, label="Simulated Force RMS (µN)")
fig.savefig(os.path.join(out_dir, "rms_heatmaps.png"), dpi=450)
plt.close(fig)

# Figure 2: signed error per strain (diverging, centered at 0) + MSE over all strains
fig, axes = plt.subplots(2, 3, figsize=(20, 11.5), constrained_layout=True)
lim = np.nanmax(np.abs(err_grid))
err_norm = TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim)
for k, (ax, s) in enumerate(zip(axes.flat, strains)):
    im = draw_heatmap(ax, err_grid[k], "RdBu_r", err_norm, ".2f")
    ax.set_title(f"{s}% strain: sim − exp")
fig.colorbar(im, ax=axes[:, :2], shrink=0.6, label="Force RMS error, sim − exp (µN)")

ax = axes.flat[-1]
im = draw_heatmap(ax, mse_grid, "Purples", plt.Normalize(np.nanmin(mse_grid), np.nanmax(mse_grid)), ".3f")
r, c = np.unravel_index(np.nanargmin(mse_grid), mse_grid.shape)
ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=2))
ax.set_title(f"MSE over {train_label}\nbest: $\\eta_{{max}}$={eta_vals[r]:.0f} Pa, $a_1$={a1_vals[c]:g}")
fig.colorbar(im, ax=ax, shrink=0.9, label="MSE (µN²)")
fig.savefig(os.path.join(out_dir, "rms_error_heatmaps.png"), dpi=450)
plt.close(fig)

msg = f"best fit: Eta_Max = {eta_vals[r]:.1f} Pa, a1 = {a1_vals[c]:g}, train MSE ({train_label}) = {mse_grid[r, c]:.4g} µN^2"

# Figure 3 (holdout only): training MSE next to the validation squared error,
# with the training best fit boxed in both so its validation error can be read off
if HOLDOUT:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), constrained_layout=True)
    for ax, grid, cmap, title, label in [
            (axes[0], mse_grid, "Purples", f"Training MSE over {train_label}", "MSE (µN²)"),
            (axes[1], val_grid, "Oranges", f"Validation squared error at {VAL_STRAIN}%", "Squared error (µN²)")]:
        im = draw_heatmap(ax, grid, cmap, plt.Normalize(np.nanmin(grid), np.nanmax(grid)), ".3f")
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=2))
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.9, label=label)
    fig.savefig(os.path.join(out_dir, "rms_validation_heatmaps.png"), dpi=450)
    plt.close(fig)
    msg += f", val SE ({VAL_STRAIN}%) = {val_grid[r, c]:.4g} µN^2"

print(msg)
