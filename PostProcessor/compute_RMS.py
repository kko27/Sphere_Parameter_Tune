import numpy as np
import os
from matplotlib import pyplot as plt

# Force RMS at each strain step for every sweep sample that has a
# net_reaction_force.csv (run compute_reaction_forces.py first).
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(ROOT_DIR, "Samples")
RESULTS_SUBDIR = "24-procs"
file_name = "net_reaction_force.csv"

# samples.txt: sample i_eta i_a1 Eta_Max a1
params = np.genfromtxt(os.path.join(SAMPLES_DIR, "samples.txt"), names=True, dtype=None, encoding=None)
fold_list = [os.path.join(SAMPLES_DIR, p["sample"], RESULTS_SUBDIR) for p in params]
has_csv = [os.path.exists(os.path.join(f, file_name)) for f in fold_list]
params = params[has_csv]
fold_list = [f for f, ok in zip(fold_list, has_csv) if ok]
if not fold_list:
    raise SystemExit(f"no {file_name} found under {SAMPLES_DIR}")

# Combined RMS / peak-to-peak figures and the summary table are saved in Samples/
out_dir = SAMPLES_DIR

# For each 50 time points, compute the force RMS (SD about the window mean, no detrend
# or noise-floor correction)
window_size = 50
min_samples = 40    # last window (25-29.5 s) has only 45 samples
displacements = np.array([0, 5, 10, 15, 20, 25])

colors = plt.cm.viridis(np.linspace(0, 0.9, len(fold_list)))

fig_rms, ax_rms = plt.subplots(figsize=(5, 5))
summary_rows = []

for p, fold, color in zip(params, fold_list, colors):
    data = np.loadtxt(os.path.join(fold, file_name), delimiter=',', skiprows=1)

    time_s = data[:,0]
    sphere_top_Fz = data[:,3]/1e6       # Convert to µN

    # Individual reaction force plot, one per folder
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(time_s, np.abs(sphere_top_Fz), label="Reaction Force", color='b', linewidth=1)
    ax.set_xlabel("Time (s)", fontsize = 28)
    ax.set_ylabel("Force (µN)", fontsize = 28)

    ax.tick_params(labelsize=20)
    # ax.set_title(label, fontsize = 14)
    # ax.legend(fontsize = 20)
    ax.grid(True)
    plt.tight_layout()
    fig.savefig(os.path.join(fold, "reaction_force_over_time.png"), dpi=450)
    plt.close(fig)

    rms_values = []
    peak_to_peak_values = []
    for i in range(0, len(sphere_top_Fz), window_size):
        window = sphere_top_Fz[i:i+window_size]
        if len(window) >= min_samples:
            rms_values.append(np.std(window))
            peak_to_peak_values.append(np.ptp(window))  # Peak to peak value

    n = min(len(displacements), len(rms_values))
    ax_rms.plot(displacements[:n], rms_values[:n], marker='o', color=color, linewidth=1)

    rms_padded = np.full(len(displacements), np.nan)
    rms_padded[:n] = rms_values[:n]
    summary_rows.append([p["Eta_Max"], p["a1"], *rms_padded])

# One row per sample: Eta_Max, a1, Force RMS (µN) at each strain step
np.savetxt(os.path.join(out_dir, "rms_summary.csv"),
           np.column_stack([params["sample"], np.array(summary_rows, dtype=object)]),
           delimiter=",", fmt="%s",
           header=",".join(["sample", "Eta_Max", "a1"] + [f"rms_{d}pct_uN" for d in displacements]),
           comments="")

# Combined RMS plot
ax_rms.set_xlabel("Strain (%)", fontsize = 28)
ax_rms.set_ylabel("Force RMS (µN)", fontsize = 28)
ax_rms.tick_params(labelsize=24)
ax_rms.grid()
fig_rms.tight_layout()
fig_rms.savefig(os.path.join(out_dir, "rms_reaction_force.png"), dpi=450)
plt.close(fig_rms)
