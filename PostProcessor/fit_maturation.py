import numpy as np
import os
import re
from glob import glob
from matplotlib import pyplot as plt
from scipy.signal import detrend

# Figure font sizes, used by every figure below
plt.rcParams.update({"xtick.labelsize": 14, "ytick.labelsize": 14, "axes.labelsize": 16,
                     "legend.fontsize": 14, "legend.title_fontsize": 14, "axes.titlesize": 18})

# Fits the sweep to the Force RMS of each day of maturity (Results/experimental_data/dNN_force_rms.csv)
# and plots the Peak Active Stress vs End-Diastolic Fiber Stretch curve of the best grid sample.
# Each day is fitted three times, against the mean and mean +/- SD RMS, so the curve comes with a band.
# Inputs come from Results/force_rms (sNNN_section_force.csv, sNNN_frank_starling.csv).
# Usage: python PostProcessor/fit_maturation.py      (HOLDOUT=0 python ... for the full 5-25% fit)
#
# HOLDOUT: fit (Eta_Max, a1) to 5-20% only and keep 25% as the validation point. The validation
# error is the squared error at 25% against the same target (-SD / mean / +SD) the fit used.
# Holdout outputs go to Results/maturation_fit/holdout_25, full-fit outputs to Results/maturation_fit.
HOLDOUT = os.environ.get("HOLDOUT", "1") != "0"
VAL_STRAIN = 25              # % strain held out when HOLDOUT is on

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(ROOT_DIR, "Samples")
EXP_DIR = os.path.join(ROOT_DIR, "Results", "experimental_data")
FORCE_DIR = os.path.join(ROOT_DIR, "Results", "force_rms")
out_dir = os.path.join(ROOT_DIR, "Results", "maturation_fit")
if HOLDOUT:
    out_dir = os.path.join(out_dir, f"holdout_{VAL_STRAIN}")
os.makedirs(out_dir, exist_ok=True)

# Same plateau windows as compute_RMS.py: 5% strain per step, sigmoid ramps centred
# every 10 s, skip 2 s after the ramp centre, then 6 beats (activation period 1.1 s)
strains = np.array([5, 10, 15, 20, 25])
step_centers = np.array([5.0, 15.0, 25.0, 35.0, 45.0])   # s
settle = 2.0                 # s after the step centre
beat_period = 1.1            # s, activation_cardioid.dat
n_beats = 6
windows = [(c + settle, c + settle + n_beats * beat_period) for c in step_centers]
min_samples = 50

targets = ["-SD", "mean", "+SD"]   # experimental RMS each day is fitted against

# Strains used for the fit (training) and the held-out validation strain, if any
val_mask = (strains == VAL_STRAIN) if HOLDOUT else np.zeros(len(strains), bool)
train_mask = ~val_mask
train_label = f"{strains[train_mask].min()}–{strains[train_mask].max()}%"


def plateau_rms(t, F, t0, t1):
    """SD of the linearly detrended force in [t0, t1), as in compute_RMS.py."""
    mask = (t >= t0) & (t < t1)
    if np.count_nonzero(mask) < min_samples:
        return np.nan
    return float(np.std(detrend(F[mask], type="linear")))


# samples.txt: sample i_eta i_a1 Eta_Max a1
params = np.genfromtxt(os.path.join(SAMPLES_DIR, "samples.txt"), names=True, dtype=None, encoding=None)

# Simulated RMS per sample and strain; samples without results stay NaN
rms = np.full((len(params), len(strains)), np.nan)
fs_curves = {}
for i, p in enumerate(params):
    force_csv = os.path.join(FORCE_DIR, f"{p['sample']}_section_force.csv")
    fs_csv = os.path.join(FORCE_DIR, f"{p['sample']}_frank_starling.csv")
    if not (os.path.exists(force_csv) and os.path.exists(fs_csv)):
        continue
    data = np.loadtxt(force_csv, delimiter=",", skiprows=1)
    platen_F = data[:, 1] / 1e6    # z = 0 section force, convert to µN
    rms[i] = [plateau_rms(data[:, 0], platen_F, t0, t1) for t0, t1 in windows]
    fs_curves[i] = np.genfromtxt(fs_csv, delimiter=",", names=True)

valid = ~np.any(np.isnan(rms), axis=1)
print(f"{valid.sum()} / {len(params)} samples with results, missing: "
      f"{', '.join(params['sample'][~valid]) or 'none'}")

# Experimental data: one file per day, columns strain (%), Force RMS (µN), SD (µN)
exp_files = sorted(glob(os.path.join(EXP_DIR, "d*_force_rms.csv")),
                   key=lambda f: int(re.search(r"d(\d+)_", os.path.basename(f)).group(1)))
days = np.array([int(re.search(r"d(\d+)_", os.path.basename(f)).group(1)) for f in exp_files])
exp_mean, exp_sd = [], []
for f in exp_files:
    exp = np.genfromtxt(f, delimiter=",", skip_header=1)
    exp_mean.append([exp[exp[:, 0] == s, 1][0] for s in strains])
    exp_sd.append([exp[exp[:, 0] == s, 2][0] for s in strains])
exp_mean, exp_sd = np.array(exp_mean), np.array(exp_sd)


def stress_stretch(i):
    fs = fs_curves[i]
    return fs["ed_fiber_stretch"], fs["peak_active_stress_Pa"]


# Best grid sample (lowest MSE over the training strains) for every day and target
best = {}
val_se = {}     # squared error at the held-out strain of the best fit, per (day, target)
rows = []
for d, day in enumerate(days):
    for target, sign in zip(targets, [-1, 0, 1]):
        exp_rms = exp_mean[d] + sign * exp_sd[d]
        sq_err = (rms - exp_rms) ** 2
        mse = np.mean(sq_err[:, train_mask], axis=1)
        mse[~valid] = np.inf
        i = int(np.argmin(mse))
        lam, sigma = stress_stretch(i)
        slope = np.polyfit(lam, sigma, 1)[0]    # Pa per unit stretch
        best[day, target] = (i, slope)
        row = [f"d{day}", target, params["sample"][i], params["Eta_Max"][i], params["a1"][i],
               f"{mse[i]:.4g}", f"{slope:.2f}"]
        msg = (f"d{day} {target:>4}: {params['sample'][i]}  Eta_Max = {params['Eta_Max'][i]:.0f} Pa, "
               f"a1 = {params['a1'][i]:g}, train MSE ({train_label}) = {mse[i]:.4g} µN^2")
        if HOLDOUT:
            val_se[day, target] = float(sq_err[i, val_mask][0])
            row.append(f"{val_se[day, target]:.4g}")
            msg += f", val SE ({VAL_STRAIN}%) = {val_se[day, target]:.4g} µN^2"
        rows.append(row)
        print(msg + f", dσ/dλ = {slope:.1f} Pa")

header = "day,target,sample,Eta_Max,a1,mse_uN2,slope_Pa"     # mse over the training strains
if HOLDOUT:
    header += f",val_se_{VAL_STRAIN}pct_uN2"
np.savetxt(os.path.join(out_dir, "best_fits.csv"), np.array(rows, dtype=object), delimiter=",", fmt="%s",
           header=header, comments="")

# One shade of blue per day, light (youngest) to dark (oldest)
day_colors = plt.cm.Blues(np.linspace(0.45, 0.95, len(days)))


def draw_band(ax, day, color, alpha=0.2, edges=False):
    """Mean-fit curve with the region between the -SD and +SD fit curves shaded."""
    lam_lo, sig_lo = stress_stretch(best[day, "-SD"][0])
    lam_hi, sig_hi = stress_stretch(best[day, "+SD"][0])
    ax.fill(np.r_[lam_lo, lam_hi[::-1]], np.r_[sig_lo, sig_hi[::-1]], color=color, alpha=alpha, linewidth=0)
    if edges:   # outline the band so overlapping days stay distinguishable
        ax.plot(lam_lo, sig_lo, ":", color=color, linewidth=1)
        ax.plot(lam_hi, sig_hi, "--", color=color, linewidth=1)
    lam, sigma = stress_stretch(best[day, "mean"][0])
    ax.plot(lam, sigma, marker="o", markersize=5, color=color, linewidth=2, label=f"Day {day}")


# Figure 1: stress-stretch fit for each day
fig, axes = plt.subplots(1, len(days), figsize=(5.5 * len(days), 5.5), sharey=True, constrained_layout=True)
for ax, day, color in zip(np.atleast_1d(axes), days, day_colors):
    draw_band(ax, day, color)
    i = best[day, "mean"][0]
    ax.set_title(f"Day {day}\n$\\eta_{{max}}$={params['Eta_Max'][i]:.0f} Pa, $a_1$={params['a1'][i]:g}")
    ax.set_xlabel("End-Diastolic Fiber Stretch (-)")
    ax.grid(True, color="0.9")
np.atleast_1d(axes)[0].set_ylabel("Peak Active Stress (Pa)")
fig.savefig(os.path.join(out_dir, "stress_stretch_per_day.png"), dpi=450)
plt.close(fig)

# Figure 2: fitted RMS against the experiment, to check the quality of each fit (2 columns of days)
n_rows = int(np.ceil(len(days) / 2))
fig, axes = plt.subplots(n_rows, 2, figsize=(14, 5.5 * n_rows), sharex=True, sharey=True,
                         squeeze=False, constrained_layout=True)
for ax in axes.flat[len(days):]:
    ax.set_visible(False)
for ax in axes[:, 0]:
    ax.set_ylabel("Force RMS (µN)")
for d, (ax, day, color) in enumerate(zip(axes.flat, days, day_colors)):
    if HOLDOUT:   # shade the held-out strain so it reads as a prediction, not a fitted point
        ax.axvspan(VAL_STRAIN - 2.5, VAL_STRAIN + 2.5, color="0.93", zorder=0, label="validation")
    ax.errorbar(strains, exp_mean[d], yerr=exp_sd[d], fmt="s", color="0.3", capsize=3, label="Experiment")
    for target, ls in zip(targets, [":", "-", "--"]):
        i = best[day, target][0]
        label = f"fit to {target} ({params['sample'][i]})"
        if HOLDOUT:
            label += f", val SE {val_se[day, target]:.3g}"
        ax.plot(strains, rms[i], ls, marker="o", markersize=4, color=color, linewidth=1.5, label=label)
    ax.set_title(f"Day {day}  (fit to {train_label})")
    if d + 2 >= len(days):   # bottom panel of its column
        ax.set_xlabel("Compression Strain (%)")
        ax.tick_params(labelbottom=True)
    ax.grid(True, color="0.9")
    ax.legend(frameon=False, loc="upper left")
fig.savefig(os.path.join(out_dir, "rms_fit_per_day.png"), dpi=450)
plt.close(fig)

# Figure 3: all days' stress-stretch curves overlaid
fig, ax = plt.subplots(figsize=(10, 6.5), constrained_layout=True)
for day, color in zip(days, day_colors):
    draw_band(ax, day, color, alpha=0.08, edges=True)
ax.set_xlabel("End-Diastolic Fiber Stretch (-)")
ax.set_ylabel("Peak Active Stress (Pa)")
ax.grid(True, color="0.9")
# Day entries (mean fit), then line-style keys for the -SD / +SD fits; outside the axes to keep the curves clear
handles, _ = ax.get_legend_handles_labels()
handles += [plt.Line2D([], [], ls=":", color="0.4", label="−SD fit"),
            plt.Line2D([], [], ls="--", color="0.4", label="+SD fit")]
ax.legend(handles=handles, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1))
fig.savefig(os.path.join(out_dir, "stress_stretch_maturation.png"), dpi=450)
plt.close(fig)

# Figure 4: slope, Eta_Max and a1 against the day (mean fit, gray bar spans the -SD and +SD fits)
fig, trend_axes = plt.subplots(3, 1, figsize=(6.5, 9), sharex=True, constrained_layout=True)
per_target = {
    "slope": lambda i, s: s,
    "Eta_Max": lambda i, s: params["Eta_Max"][i],
    "a1": lambda i, s: params["a1"][i],
}
labels = [r"$d\sigma/d\lambda$ (Pa)", r"$\eta_{max}$ (Pa)", r"$a_1$"]
for tax, (key, value), label in zip(trend_axes, per_target.items(), labels):
    vals = {t: np.array([value(*best[day, t]) for day in days]) for t in targets}
    lo, hi = np.minimum(vals["-SD"], vals["+SD"]), np.maximum(vals["-SD"], vals["+SD"])
    tax.vlines(days, lo, hi, color="0.75", linewidth=2)
    tax.plot(days, vals["mean"], color="0.6", linewidth=1, zorder=1)
    tax.scatter(days, vals["mean"], c=day_colors, s=70, zorder=2, edgecolor="white", linewidth=1)
    tax.set_ylabel(label)
    tax.set_xticks(days)
    tax.grid(True, color="0.9")
trend_axes[-1].set_xlabel("Day of maturity")
fig.savefig(os.path.join(out_dir, "properties_maturation.png"), dpi=450)
plt.close(fig)


def two_three_axes(fig):
    """Five equal panels: 2 centred on the top row, 3 on the bottom row."""
    gs = fig.add_gridspec(2, 6)
    return [fig.add_subplot(gs[0, 1:3]), fig.add_subplot(gs[0, 3:5]),
            fig.add_subplot(gs[1, 0:2]), fig.add_subplot(gs[1, 2:4]), fig.add_subplot(gs[1, 4:6])]


# Figure 5: simulated RMS heatmaps over the (Eta_Max, a1) grid, one panel per strain, no best fit marked
eta_vals = np.unique(params["Eta_Max"])
a1_vals = np.unique(params["a1"])
i_eta = np.searchsorted(eta_vals, params["Eta_Max"])
i_a1 = np.searchsorted(a1_vals, params["a1"])
rms_grid = np.full((len(strains), len(eta_vals), len(a1_vals)), np.nan)
for k in range(len(strains)):
    rms_grid[k, i_eta, i_a1] = rms[:, k]

cmap = plt.get_cmap("Blues").copy()
cmap.set_bad("white")
norm = plt.Normalize(np.nanmin(rms_grid), np.nanmax(rms_grid))
fig = plt.figure(figsize=(19, 11), constrained_layout=True)
axes = two_three_axes(fig)
for k, (ax, s) in enumerate(zip(axes, strains)):
    grid = rms_grid[k]
    im = ax.imshow(np.ma.masked_invalid(grid), origin="lower", aspect="auto", cmap=cmap, norm=norm)
    # Hatch the samples with no result so they don't read as a low value
    for r, c in zip(*np.where(np.isnan(grid))):
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, hatch="///",
                                   edgecolor="0.75", linewidth=0))
    for r, c in zip(*np.where(~np.isnan(grid))):
        lum = np.dot(im.cmap(im.norm(grid[r, c]))[:3], [0.299, 0.587, 0.114])
        ax.text(c, r, f"{grid[r, c]:.3f}", ha="center", va="center", fontsize=8,
                color="white" if lum < 0.5 else "black")
    ax.set_xticks(range(len(a1_vals)), [f"{v:g}" for v in a1_vals])
    ax.set_yticks(range(len(eta_vals)), [f"{v:.0f}" for v in eta_vals])
    ax.set_xlabel("$a_1$")
    if k in (0, 2):   # first panel of each row
        ax.set_ylabel(r"$\eta_{max}$ (Pa)")
    ax.set_title(f"{s}% strain")
fig.colorbar(im, ax=axes, shrink=0.6, label="Simulated Force RMS (µN)")
fig.savefig(os.path.join(out_dir, "rms_heatmaps.png"), dpi=450)
plt.close(fig)

if HOLDOUT:
    print(f"validation ({VAL_STRAIN}%) SE, mean over days: " + ", ".join(
        f"{t} {np.mean([val_se[day, t] for day in days]):.4g} µN^2" for t in targets))
print(f"figures and best_fits.csv saved in {out_dir}")
