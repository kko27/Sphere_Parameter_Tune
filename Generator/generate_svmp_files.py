# Python code that generates the .xml files for
# svmultiphysics simulations

# Builds a full-factorial grid over the 2-dimensional parameter space
# Parameters are: Eta_Max and a1 (N_ETA_MAX x N_A1 = 10 x 10 = 100 combinations)

# For each combination, create a run directory inside Samples
# Label Samples as s001, s002, ..., s100 (Eta_Max is the outer index, a1 the inner)
# Save the sample parameters as a txt file with fields: sample, i_eta, i_a1, Eta_Max, a1
# This txt file is saved as samples.txt inside Samples

# The runs themselves are launched by ../driver.py (see ../submit_sweep.sbatch)

import os
import re
import sys

import numpy as np

# Paths (resolved relative to this script so it can be run from anywhere)
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR     = os.path.dirname(SCRIPT_DIR)
BASELINE_DIR = os.path.join(ROOT_DIR, 'BaselineFiles')
SAMPLES_DIR  = os.path.join(ROOT_DIR, 'Samples')

# Sample Space (grid includes both end points of each range):
N_ETA_MAX   = 10
N_A1        = 10
ETA_MAX_RNG = (50.0, 300.0)   # Pascals
A1_RNG      = (1.5, 6.0)      # Frank-Starling Slope

# User-Specified Inputs
Youngs_Modulus = 200 # Pascals

eta_max_values = np.linspace(*ETA_MAX_RNG, N_ETA_MAX)
a1_values      = np.linspace(*A1_RNG, N_A1)

# Refuse to overwrite inputs of samples that have already been run: the results
# would no longer match the parameters in samples.txt
if os.path.isdir(SAMPLES_DIR):
    started = [d for d in os.listdir(SAMPLES_DIR)
               if os.path.exists(os.path.join(SAMPLES_DIR, d, '.claim'))]
    if started:
        sys.exit(f'{len(started)} samples in {SAMPLES_DIR} have already been run '
                 f'(e.g. {sorted(started)[0]}). Move or delete Samples/ before regenerating.')

# Read in the baseline files once
with open(os.path.join(BASELINE_DIR, 'baseline_solver.xml')) as f:
    baseline_xml = f.read()

# activation_cardioid.dat: first line is a header (n_points, n_modes),
# remaining lines are (time, normalized activation)
with open(os.path.join(BASELINE_DIR, 'activation_cardioid.dat')) as f:
    activation_header = f.readline().strip()
activation = np.loadtxt(os.path.join(BASELINE_DIR, 'activation_cardioid.dat'), skiprows=1)

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
    f_samples.write('sample i_eta i_a1 Eta_Max a1\n')

    sample_number = 0
    for i_eta, Eta_Max in enumerate(eta_max_values):
        for i_a1, a1 in enumerate(a1_values):
            sample_number += 1
            name = f's{sample_number:03d}'

            # Create a save directory for each sample
            sample_dir = os.path.join(SAMPLES_DIR, name)
            os.makedirs(sample_dir, exist_ok=True)

            # Solver xml with Elasticity_modulus and a1 replaced
            with open(os.path.join(sample_dir, 'solver.xml'), 'w') as f:
                f.write(make_solver_xml(Youngs_Modulus, a1))

            # Activation waveform with the 2nd column scaled by Eta_Max
            scaled = activation.copy()
            scaled[:, 1] *= Eta_Max
            np.savetxt(os.path.join(sample_dir, 'activation_cardioid.dat'), scaled,
                       fmt='%.6f', header=activation_header, comments='')

            f_samples.write(f'{name} {i_eta} {i_a1} {Eta_Max:.6f} {a1:.6f}\n')

print(f'Generated {sample_number} samples ({N_ETA_MAX} Eta_Max x {N_A1} a1) in {SAMPLES_DIR}')
