# Python code that generates the .xml files for
# svmultiphysics simulations

# Perform LHS (Latin Hypercube Sampling) for the 2-dimensional parameter space
# Parameters are: Eta_Max and a1
# Sample from 20 different combinations of Eta_Max and a1.

# Loop through each 20 combinations of Eta_Max and a1
# Create a save directory inside Samples
# Label Samples as s1, s2, s3, ..., s20
# Save the sample parameters as a txt file fields: sample_number, Eta_Max, a1
# This txt file can be saved as samples.txt inside Samples

import os
import re

import numpy as np
from scipy.stats import qmc

# Paths (resolved relative to this script so it can be run from anywhere)
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR     = os.path.dirname(SCRIPT_DIR)
BASELINE_DIR = os.path.join(ROOT_DIR, 'BaselineFiles')
SAMPLES_DIR  = os.path.join(ROOT_DIR, 'Samples')

# Sample Space:
N_SAMPLES   = 20
SEED        = 42
ETA_MAX_RNG = (50.0, 300.0)   # Pascals
A1_RNG      = (1.5, 6.0)      # Frank-Starling Slope

# User-Specified Inputs
Youngs_Modulus = 200 # Pascals

# Latin Hypercube Sampling of (Eta_Max, a1)
sampler = qmc.LatinHypercube(d=2, seed=SEED)
unit_samples = sampler.random(n=N_SAMPLES)
samples = qmc.scale(unit_samples,
                    [ETA_MAX_RNG[0], A1_RNG[0]],
                    [ETA_MAX_RNG[1], A1_RNG[1]])

# Read in the baseline files once
with open(os.path.join(BASELINE_DIR, 'baseline_solver.xml')) as f:
    baseline_xml = f.read()

# activation_cardioid.dat: first line is a header (n_points, n_modes),
# remaining lines are (time, normalized activation)
with open(os.path.join(BASELINE_DIR, 'activation_cardioid.dat')) as f:
    activation_header = f.readline().strip()
activation = np.loadtxt(os.path.join(BASELINE_DIR, 'activation_cardioid.dat'), skiprows=1)

with open(os.path.join(BASELINE_DIR, 'run_simulation.sh')) as f:
    baseline_run_script = f.read()

def make_solver_xml(E, a1):
    xml = baseline_xml
    # Replace Elasticity_modulus (all domains) and a1
    xml = re.sub(r'<Elasticity_modulus>\s*[^<]*</Elasticity_modulus>',
                 f'<Elasticity_modulus> {float(E)} </Elasticity_modulus>', xml)
    xml = re.sub(r'<a1>\s*[^<]*</a1>', f'<a1> {a1:.6f} </a1>', xml)
    # Point the mesh and plate step files back to BaselineFiles
    xml = xml.replace('../../mesh_scaled/', '../../BaselineFiles/mesh_scaled/')
    xml = re.sub(r'<Temporal_values_file_path>\s*(\w+_plate_step\.dat)\s*</Temporal_values_file_path>',
                 r'<Temporal_values_file_path> ../../BaselineFiles/\1 </Temporal_values_file_path>', xml)
    return xml

os.makedirs(SAMPLES_DIR, exist_ok=True)

with open(os.path.join(SAMPLES_DIR, 'samples.txt'), 'w') as f_samples:
    f_samples.write('sample_number Eta_Max a1\n')

    for i, (Eta_Max, a1) in enumerate(samples, start=1):
        # Create a save directory for each sample
        sample_dir = os.path.join(SAMPLES_DIR, f's{i}')
        os.makedirs(sample_dir, exist_ok=True)

        # Solver xml with Elasticity_modulus and a1 replaced
        with open(os.path.join(sample_dir, 'solver.xml'), 'w') as f:
            f.write(make_solver_xml(Youngs_Modulus, a1))

        # Activation waveform with the 2nd column scaled by Eta_Max
        scaled = activation.copy()
        scaled[:, 1] *= Eta_Max
        np.savetxt(os.path.join(sample_dir, 'activation_cardioid.dat'), scaled,
                   fmt='%.6f', header=activation_header, comments='')

        # Run script pointing at solver.xml
        run_path = os.path.join(sample_dir, 'run_simulation.sh')
        with open(run_path, 'w') as f:
            f.write(baseline_run_script.replace('solver_tanH.xml', 'solver.xml'))
        os.chmod(run_path, 0o755)

        f_samples.write(f'{i} {Eta_Max:.6f} {a1:.6f}\n')

print(f'Generated {N_SAMPLES} samples in {SAMPLES_DIR}')
