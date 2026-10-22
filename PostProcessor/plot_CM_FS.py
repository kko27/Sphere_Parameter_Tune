# This python script opens the directory specified by the user and plots the
# Peak Active Stress Against the End-Diastolic Fiber Stretch for each
# Step of the compressive strain loading. The window for analysis is
# conceptually very similar to the RMS analysis window. The fiber stretch should
# be the ones taken in the equatorial plane of the sphere. In addition, it should
# be at the most circumferential (as that represents the hoop strain). The other
# caveat is that the fiber stretch should be taken at the end of diastole, which
# is when the cardiac organoid is not actively contracting
# The fiber stretch can be averaged from each window.
# Script returns a plot of Active Stress Peak vs End-Diastolic Fiber Stretch

import pyvista as pv
import numpy as np
import os
import re
import sys
from matplotlib import pyplot as plt

# Usage: python PostProcessor/plot_CM_FS.py [sample_dir]   (relative to the repo root)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_NAME = sys.argv[1] if len(sys.argv) > 1 else './FullSimulations/s085'
SIM_DIR = os.path.join(ROOT_DIR, DIR_NAME)
RESULTS_DIR = os.path.join(SIM_DIR, os.environ.get("RESULTS_SUBDIR", "24-procs"))
MESH_DIR = os.path.join(ROOT_DIR, "BaselineFiles", "mesh_scaled")

# Same plateau windows as compute_RMS.py: 5% strain per step, sigmoid ramps centred
# every 10 s, skip 2 s after the ramp centre, then 6 beats (activation period 1.1 s)
strains = np.array([5, 10, 15, 20, 25])
step_centers = np.array([5.0, 15.0, 25.0, 35.0, 45.0])   # s
settle = 2.0                 # s after the step centre
beat_period = 1.1            # s, activation_cardioid.dat
n_beats = 6
windows = [(c + settle, c + settle + n_beats * beat_period) for c in step_centers]

# End-diastole: last frame before activation_cardioid.dat rises (onset at +0.10 s in each beat)
ed_offset = 0.05             # s after the beat start

# Equatorial ROI: active (DOMAIN_ID 1) nodes near z = 0 whose fibers are circumferential
active_domain = 1
eq_half_width = 20.0         # µm, |z| in the reference configuration
min_circ_cos = 0.95          # |f0 . e_phi|


def time_step_size(solver_xml):
    with open(solver_xml) as f:
        return float(re.search(r"<Time_step_size>\s*([0-9.eE+-]+)\s*<", f.read()).group(1))


def equatorial_roi():
    """Node mask of the equatorial, circumferentially oriented cardiomyocyte shell."""
    dom_mesh = pv.read(os.path.join(MESH_DIR, "mesh-complete.domains.vtu"))
    fib_mesh = pv.read(os.path.join(MESH_DIR, "fibersLong.vtu"))

    # Reference fibers are stored per cell: average them onto the nodes of the active cells only
    active_cells = np.flatnonzero(dom_mesh.cell_data["DOMAIN_ID"] == active_domain)
    shell = fib_mesh.extract_cells(active_cells).cell_data_to_point_data()
    node_ids = shell.point_data["vtkOriginalPointIds"]
    f0 = shell.point_data["FIB_DIR"]
    f0 = f0 / np.linalg.norm(f0, axis=1, keepdims=True)

    X = dom_mesh.points[node_ids]
    e_phi = np.column_stack([-X[:, 1], X[:, 0], np.zeros(len(X))])
    e_phi /= np.linalg.norm(e_phi, axis=1, keepdims=True)
    circ = np.abs(np.sum(f0 * e_phi, axis=1)) > min_circ_cos
    equator = np.abs(X[:, 2]) < eq_half_width

    mask = np.zeros(dom_mesh.n_points, dtype=bool)
    mask[node_ids[circ & equator]] = True
    return mask, dom_mesh.points


def read_frame(n):
    return pv.read(os.path.join(RESULTS_DIR, f"result_{n:03d}.vtu"))


dt = time_step_size(os.path.join(SIM_DIR, "solver.xml"))
roi, ref_points = equatorial_roi()
print(f"{DIR_NAME}: {np.count_nonzero(roi)} equatorial circumferential nodes in the ROI")
if not np.allclose(read_frame(1).points, ref_points, atol=1e-3):
    raise SystemExit("result VTU node ordering does not match the baseline mesh")

# Per strain step: mean (and SD across beats) of the ED stretch and peak active stress
rows = []
for strain, (t0, t1) in zip(strains, windows):
    beat_starts = beat_period * np.arange(np.ceil(t0 / beat_period - 1e-9), t1 / beat_period)
    ed_stretch, peak_stress = [], []
    for tb in beat_starts:
        n_ed = int(round((tb + ed_offset) / dt))
        n_end = int(round((tb + beat_period) / dt))
        frames = range(n_ed, n_end)
        if not os.path.exists(os.path.join(RESULTS_DIR, f"result_{n_end - 1:03d}.vtu")):
            continue   # simulation did not reach the end of this beat

        Ta = []
        for n in frames:
            m = read_frame(n)
            if n == n_ed:
                ed_stretch.append(m.point_data["Fiber_stretch"][roi].mean())
            Ta.append(m.point_data["Active_tension_fibers"][roi].mean())
        peak_stress.append(max(Ta))

    if not ed_stretch:
        print(f"  {strain}%: no complete beats in [{t0:.1f}, {t1:.1f}) s, skipped")
        continue
    rows.append([strain, len(ed_stretch),
                 np.mean(ed_stretch), np.std(ed_stretch),
                 np.mean(peak_stress), np.std(peak_stress)])
    print(f"  {strain}%: {len(ed_stretch)} beats, ED stretch {rows[-1][2]:.4f}, "
          f"peak active stress {rows[-1][4]:.2f} Pa")

if not rows:
    raise SystemExit(f"no complete beats found in {RESULTS_DIR}")
rows = np.array(rows)

# One row per strain step
np.savetxt(os.path.join(RESULTS_DIR, "frank_starling.csv"), rows, delimiter=",",
           fmt=["%d", "%d", "%.6f", "%.6f", "%.4f", "%.4f"],
           header="strain_pct,n_beats,ed_fiber_stretch,ed_fiber_stretch_sd,"
                  "peak_active_stress_Pa,peak_active_stress_sd_Pa",
           comments="")

fig, ax = plt.subplots(figsize=(6, 5))
ax.errorbar(rows[:, 2], rows[:, 4], xerr=rows[:, 3], yerr=rows[:, 5],
            marker='o', color='b', linewidth=1, capsize=3)
for strain, x, y in zip(rows[:, 0], rows[:, 2], rows[:, 4]):
    ax.annotate(f"{strain:.0f}%", (x, y), textcoords="offset points", xytext=(6, -12), fontsize=14)
ax.set_xlabel("End-Diastolic Fiber Stretch (-)", fontsize = 22)
ax.set_ylabel("Peak Active Stress (Pa)", fontsize = 22)
ax.tick_params(labelsize=16)
ax.grid(True)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "frank_starling.png"), dpi=450)
plt.close(fig)
