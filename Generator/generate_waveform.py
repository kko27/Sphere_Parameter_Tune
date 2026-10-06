import itertools
import numpy as np
from matplotlib import pyplot as plt

# Code generates a half sinuidal waveform for stress over time
T = 1001
Eta_max = 1     # Normalized Unitless 
time_factor = 1 # seconds to milliseconds
N_cycles = 1    # number of cardiac cycles to prescribe
Delay = 0.1     # seconds of zero stress to prescribe before the waveform starts

def smoothstep_function(x, a, b):
    s = np.zeros_like(x)
    for i in range(len(x)):
        if x[i] <= a:
            s[i] = 1
        elif x[i] >= b:
            s[i] = 0
        else:
            s[i] = np.exp(1 - 1/(1 - (x[i] - a)/(b - a))**2)
    return s

def generate_waveform(T, Eta_max):
    t = np.arange(0, 1, 1/(T-1))
    Etas = np.sin(2 * np.pi * t) * Eta_max
    # If Etas < 0, set it to 0
    Etas[Etas < 0] = 0
    return t, Etas

def hill_function(x, k, n):
    # Rising Hill/sigmoid: 0 -> 1 as x increases past k, steepness set by n
    x = np.clip(x, 0, None)
    return x**n / (x**n + k**n)

def double_hill_waveform(t, Eta_max, k1, n1, k2, n2):
    # Product of a rising Hill (turns pulse on) and a falling Hill (turns it off)
    rise = hill_function(t, k1, n1)
    fall = 1 - hill_function(t, k2, n2)
    pulse = rise * fall
    return Eta_max * pulse / pulse.max()

t, Etas = generate_waveform(T, Eta_max)
s = smoothstep_function(t, 0.45, 0.55)

final_waveform = Etas * s

def fit_double_hill(t, target, Eta_max):
    # Coarse-to-fine grid search over (k1, n1, k2, n2) to minimize squared error
    # against the target waveform (numpy-only, no scipy dependency).
    best_params = None
    best_error = np.inf
    k1_range = np.linspace(0.05, 0.35, 16)
    n1_range = np.array([2, 3, 4, 6, 8, 12])
    k2_range = np.linspace(0.3, 0.55, 16)
    n2_range = np.array([2, 3, 4, 6, 8, 12])
    for k1, n1, k2, n2 in itertools.product(k1_range, n1_range, k2_range, n2_range):
        candidate = double_hill_waveform(t, Eta_max, k1, n1, k2, n2)
        error = np.sum((candidate - target) ** 2)
        if error < best_error:
            best_error = error
            best_params = (k1, n1, k2, n2)
    return best_params

# Fit a single cardiac cycle, then repeat it for N_cycles
k1, n1, k2, n2 = fit_double_hill(t, final_waveform, Eta_max)
hill_waveform = double_hill_waveform(t, Eta_max, k1, n1, k2, n2)

# Tile the single-cycle waveforms across N_cycles, shifting each repeat by one
# cycle length so time keeps increasing monotonically
t_multi = np.concatenate([t + cycle for cycle in range(N_cycles)]) + Delay
final_waveform_multi = np.tile(final_waveform, N_cycles)
hill_waveform_multi = np.tile(hill_waveform, N_cycles)

# Prepend a zero-stress delay segment, sampled at the same spacing as the
# cardiac cycle, so the waveform only starts ramping up after Delay seconds
dt = t[1] - t[0]
n_delay = int(round(Delay / dt))
delay_t = np.arange(0, n_delay) * dt
delay_values = np.zeros(n_delay)

t_multi = np.concatenate([delay_t, t_multi])
final_waveform_multi = np.concatenate([delay_values, final_waveform_multi])
hill_waveform_multi = np.concatenate([delay_values, hill_waveform_multi])

plt.figure(figsize=(10, 8))
# plt.plot(t_multi, final_waveform_multi, color='red', linestyle='--', label='Original (truncated sine)')
plt.plot(t_multi, hill_waveform_multi, color='blue', label='Double Hill Waveform', linewidth = 4)
plt.title('Double Hill Activation Function $\\eta(t)$', fontsize = 36)
plt.xlabel('Time (s)', fontsize = 30)
plt.ylabel('Stress (Pa)', fontsize = 30)
# plt.legend(fontsize = 14)
plt.xticks(fontsize = 26)
plt.yticks(fontsize = 26)
plt.grid(True)
plt.tight_layout()
plt.savefig(f'stress_waveform_cardiac.png', dpi=450)
plt.close()

# Save the time and hill waveform to a DAT file
# First line contains number of time points, second is number of fourier modes
time = t_multi * time_factor  # Convert to milliseconds

# Save with a unique filename: 'activation_cardioid_{}.dat' where {} is the Eta_max 
with open('../BaselineFiles/activation_cardioid.dat', 'w') as f:
    f.write(f'{len(time)} 64\n')
    for i in range(len(time)):
        f.write(f'{time[i]:.6f} {hill_waveform_multi[i]:.6f}\n')