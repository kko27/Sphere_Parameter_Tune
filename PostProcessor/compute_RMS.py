import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.signal import detrend

# Force RMS at each strain step for every sweep sample that has a
# section_force.csv (run compute_reaction_forces.py first).
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(ROOT_DIR, "Samples")
RESULTS_SUBDIR = "24-procs"
LOCAL_DIR = os.path.join(SAMPLES_DIR, "force_rms")   # copies pulled from Sherlock: sNNN_section_force.csv
file_name = "section_force.csv"

# Plate steps (top_plate_step.dat): 5% strain per step, sigmoid ramps centred every 10 s
strains = np.array([5, 10, 15, 20, 25])
step_centers = np.array([5.0, 15.0, 25.0, 35.0, 45.0])   # s

# RMS window on each plateau: skip the 1-2 distorted beats after the ramp, then take
# exactly 6 beats (activation period 1.1 s), ending before the next ramp moves the plate
settle = 2.0                 # s after the step centre
beat_period = 1.1            # s, activation_cardioid.dat
n_beats = 6
windows = [(c + settle, c + settle + n_beats * beat_period) for c in step_centers]
min_samples = 50


def find_csv(sample):
    """Force CSV and where to save its plot: Sherlock results folder, else Samples/force_rms."""
    sherlock = os.path.join(SAMPLES_DIR, sample, RESULTS_SUBDIR, file_name)
    if os.path.exists(sherlock):
        return sherlock, os.path.join(os.path.dirname(sherlock), "reaction_force_over_time.png")
    local = os.path.join(LOCAL_DIR, f"{sample}_{file_name}")
    if os.path.exists(local):
        return local, os.path.join(LOCAL_DIR, f"{sample}_reaction_force_over_time.png")
    return None, None


def plateau_rms(t, F, t0, t1):
    """SD of the linearly detrended force in [t0, t1), as in the experiments.
    No noise-floor subtraction: the simulation is noise-free and the experimental
    values already have the floor removed."""
    mask = (t >= t0) & (t < t1)
    if np.count_nonzero(mask) < min_samples:
        return np.nan
    return float(np.std(detrend(F[mask], type="linear")))


# samples.txt: sample i_eta i_a1 Eta_Max a1
params = np.genfromtxt(os.path.join(SAMPLES_DIR, "samples.txt"), names=True, dtype=None, encoding=None)
found = [find_csv(p["sample"]) for p in params]
has_csv = [csv is not None for csv, _ in found]
params = params[has_csv]
found = [f for f, ok in zip(found, has_csv) if ok]
if not found:
    raise SystemExit(f"no {file_name} found under {SAMPLES_DIR}")

# Combined RMS figure and the summary table are saved in Samples/
out_dir = SAMPLES_DIR

colors = plt.cm.viridis(np.linspace(0, 0.9, len(found)))

fig_rms, ax_rms = plt.subplots(figsize=(5, 5))
summary_rows = []

for p, (csv_path, png_path), color in zip(params, found, colors):
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)

    time_s = data[:,0]
    platen_F = data[:,1]/1e6       # z = 0 section force, convert to µN

    # Individual reaction force plot, with the RMS windows shaded
    fig, ax = plt.subplots(figsize=(8, 5))
    for t0, t1 in windows:
        ax.axvspan(t0, t1, color='0.85', zorder=0)
    ax.plot(time_s, np.abs(platen_F), label="Reaction Force", color='b', linewidth=1)
    ax.set_xlabel("Time (s)", fontsize = 28)
    ax.set_ylabel("Force (µN)", fontsize = 28)

    ax.tick_params(labelsize=20)
    ax.grid(True)
    plt.tight_layout()
    fig.savefig(png_path, dpi=450)
    plt.close(fig)

    rms_values = np.array([plateau_rms(time_s, platen_F, t0, t1) for t0, t1 in windows])
    ax_rms.plot(strains, rms_values, marker='o', color=color, linewidth=1)
    summary_rows.append([p["Eta_Max"], p["a1"], *rms_values])

# One row per sample: Eta_Max, a1, Force RMS (µN) at each strain step
np.savetxt(os.path.join(out_dir, "rms_summary.csv"),
           np.column_stack([params["sample"], np.array(summary_rows, dtype=object)]),
           delimiter=",", fmt="%s",
           header=",".join(["sample", "Eta_Max", "a1"] + [f"rms_{s}pct_uN" for s in strains]),
           comments="")

# Combined RMS plot
ax_rms.set_xlabel("Strain (%)", fontsize = 28)
ax_rms.set_ylabel("Force RMS (µN)", fontsize = 28)
ax_rms.tick_params(labelsize=24)
ax_rms.grid()
fig_rms.tight_layout()
fig_rms.savefig(os.path.join(out_dir, "rms_reaction_force.png"), dpi=450)
plt.close(fig_rms)
