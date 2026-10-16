"""Axial platen force for the squish (ustruct) simulation, from cross-sections.

The upper half of the sphere (z > 0) is loaded in z only by the top platen
contact and by the stress transmitted across a horizontal cut: the rest of its
surface is traction-free, pin_axis_top constrains x/y only, there are no body
forces, and inertia is negligible (~1e-5 uN).  Equilibrium therefore gives

    F_platen = - integral over the cut of  sigma_zz  dA        (deformed configuration)

for any horizontal cut through the sphere.  Integrating sigma . n over the
sphere_top face instead is not usable: that face is the whole upper hemisphere,
not the contact patch, and the nodal-averaged stress on the free surface is far
from traction-free once the fibres are active (spurious dips and sign flips at
twitch onset).

Cuts are taken at z = -150, 0, 150 um in the deformed mesh (all well inside the
sphere at 25% strain).  The z = 0 cut is the reported force; the other two are a
check, since in equilibrium all three must agree.

Units: model is um / mg / s  =>  stress in Pa, force in pN  (1 Pa.um^2 = 1e-12 N).
Force is compressive-positive.

Writes:  section_force.csv   (time, force at each cut)
"""

import glob
import os
import re

import numpy as np
import pyvista as pv
from matplotlib import pyplot as plt


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(ROOT_DIR, "Samples")
RESULTS_SUBDIR = "24-procs"     # svMultiPhysics writes to "<nprocs>-procs"
OUT_NAME = "section_force.csv"
CUTS_Z = [0.0, -150.0, 150.0]   # um, deformed configuration; first one is the reported force
SPREAD_WARN = 0.10              # warn when the cuts disagree by more than 10% of the peak force (~5% is typical)


def section_force(warped, z):
    """Compressive axial force (pN) transmitted across the horizontal plane at z."""
    s = warped.slice(normal="z", origin=(0.0, 0.0, z)).triangulate()
    s = s.compute_cell_sizes(area=True).point_data_to_cell_data()
    szz = np.asarray(s.cell_data["Cauchy_stress"])[:, 2]      # Voigt [xx, yy, zz, xy, yz, xz]
    return -float(np.sum(szz * s.cell_data["Area"]))


def main():
    """Section forces for every DONE sample of the sweep that doesn't have them yet."""
    samples = sorted(d for d in glob.glob(os.path.join(SAMPLES_DIR, "s[0-9]*")) if os.path.isdir(d))
    for sample in samples:
        status_file = os.path.join(sample, "status")
        if not os.path.exists(status_file) or open(status_file).readline().strip() != "DONE":
            continue
        dir_path = os.path.join(sample, RESULTS_SUBDIR)
        if os.path.exists(os.path.join(dir_path, OUT_NAME)):
            continue
        print(f"=== {os.path.basename(sample)}")
        process_run(dir_path)


def process_run(DIR_PATH):
    PLOTFLAG = False
    OUT_CSV = os.path.join(DIR_PATH, OUT_NAME)
    OUT_PNG = os.path.join(DIR_PATH, "section_force.png")

    files = sorted(glob.glob(os.path.join(DIR_PATH, "result_*.vtu")),
                key=lambda p: int(re.search(r"result_(\d+)", p).group(1)))
    if not files:
        raise SystemExit(f"no result_*.vtu in {DIR_PATH}")

    times = []
    forces = []                                     # one row per step, one column per cut
    for fn in files:
        vol = pv.read(fn)
        t = float(vol.field_data["TimeValue"][0])
        warped = vol.warp_by_vector("Displacement")
        f = [section_force(warped, z) for z in CUTS_Z]
        times.append(t)
        forces.append(f)
        print(f"t={t:6.2f}s  " + "  ".join(f"z={z:+.0f}: {fz: .3e} pN" for z, fz in zip(CUTS_Z, f)))

    times = np.asarray(times)
    order = np.argsort(times)
    times = times[order]
    forces = np.asarray(forces)[order]

    # Equilibrium check: all cuts carry the same force
    spread = np.ptp(forces, axis=1).max() / np.abs(forces[:, 0]).max()
    print(f"max spread between cuts: {100 * spread:.1f}% of peak force")
    if spread > SPREAD_WARN:
        print(f"WARNING: cuts disagree by more than {100 * SPREAD_WARN:.0f}%, check {DIR_PATH}")

    # --- CSV -------------------------------------------------------------------
    header = ["time_s"] + [f"F_z{z:+.0f}_pN" for z in CUTS_Z]
    np.savetxt(OUT_CSV, np.column_stack([times, forces]), delimiter=",",
            header=",".join(header), comments="")
    print(f"\nwrote {OUT_CSV}")

    # ----------------------------- plot -------------------------------------
    if PLOTFLAG:
        plt.figure()
        for k, z in enumerate(CUTS_Z):
            plt.plot(times, forces[:, k] / 1e6, "-o", ms=3, label=f"cut z = {z:+.0f} µm")
        plt.xlabel("Time (s)")
        plt.ylabel("Axial platen force (µN)")
        plt.title("Squish platen force")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT_PNG, dpi=150)
        plt.close()
        print(f"wrote {OUT_PNG}")

if __name__ == "__main__":
    main()
