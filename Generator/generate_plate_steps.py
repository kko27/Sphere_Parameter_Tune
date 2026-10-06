# Create a temporal history of the displacement.
# The displacement follows a stepwise loading/unloading schedule, but instead of
# hard (discontinuous) steps we blend each level change with a tanh so the signal
# is C-infinity continuous.  A smooth signal is far better behaved under the
# Fourier interpolation svFSI applies to the temporal history (no Gibbs ringing).
#
# Nominal (hard-step) schedule, as a fraction of the total height (700 um):
#   t <  5s : 0.00        25s <= t < 35s : 0.15        
#   5s <= t < 15s : 0.05  35s <= t < 45s : 0.20        
#   15s <= t < 25s : 0.10  45s <= t < 55s : 0.25

# Each transition is replaced by  0.5 * delta * (1 + tanh(slope * (t - t0))),
# where `slope` (1/s) controls how sharp the transition is: large slope -> close
# to a hard step, small slope -> gentle ramp.  The transition is ~90% complete
# over an interval of width ~ 2 / slope centred on t0.
import numpy as np

total_height = 700.0   # um
total_time = 55.0      # s
fourier_modes = 64

# Sampling: use a fine dt so each tanh transition is well resolved for the
# Fourier fit.  (The hard-step version only sampled once per second.)
dt = 0.05              # s
slope = 2.0            # 1/s -- steepness of each tanh transition

time = np.arange(0.0, total_time + dt, dt)

# Loading/unloading schedule as (transition_time, level_after) in fractions of
# total_height.  Levels are the plateau values the hard-step function holds.
schedule = [(0.0, 0.00), (5.0, 0.05), (15.0, 0.10), 
            (25.0, 0.15), (35.0, 0.20), (45.0, 0.25)]

def tanh_history(time, schedule, slope, amplitude):
    """Smooth step history: start at the first level, then add a tanh blend for
    every subsequent level change."""
    levels = [lvl for _, lvl in schedule]
    disp = np.full_like(time, levels[0] * amplitude)
    for (t0, _), level_before, level_after in zip(
        schedule[1:], levels[:-1], levels[1:]
    ):
        delta = (level_after - level_before) * amplitude
        disp += 0.5 * delta * (1.0 + np.tanh(slope * (time - t0)))
    return disp


displacement = tanh_history(time, schedule, slope, total_height)

# Save the temporal history to "top_plate_step.dat".
# First row: number of time points, and number of Fourier modes.
# First column is time in seconds, second column is displacement in microns
# (negative -> compression toward the sphere).
with open("../BaselineFiles/top_plate_step.dat", "w") as f:
    f.write(f"{len(time)} {fourier_modes}\n")
    for t, d in zip(time, displacement):
        f.write(f"{t:.4f} {-d:.6f}\n")

# Corresponding bottom_plate_step.dat: same time points, zero displacement.
with open("../BaselineFiles/bottom_plate_step.dat", "w") as f:
    f.write(f"{len(time)} {fourier_modes}\n")
    for t in time:
        f.write(f"{t:.4f} {0.0:.6f}\n")
