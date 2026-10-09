# Description 

This codebase contains the driver script required to find the parameters $\eta_{max}$ and $a_{1}$, which determine the maximum cardiomyocyte active stress (Pa) and the Frank-Starling Gradient respectively. As inputs, the user provides the Force RMS measurements obtained from the experiments. The expected input file is a .csv text file where the first column contains the compressive strain (%) and the second column contains the Force RMS readouts (𝜇𝑁). 

Geometry files using image segmentation and meshing techniques from the optical images obtained during the experiment and the Youngs Modulus measured from passive characterization are also required as inputs to this parameter tuning procedure. 

The parameters are found with a full-factorial parameter sweep: forward simulations (structural simulations from Simvascular svmultiphysics, up to ~30 minutes each) are run on a 10 x 10 grid of $(\eta_{max}, a_{1})$ points, and the simulated RMS-Strain curves are compared with the experimental ones.

<figure>
  <img src=./Figures.png" alt="Schematic of the Calibration Workflow">
  <figcaption>Schematic of the Calibration Workflow</figcaption>
</figure>

The forward simulation is a digital twin of the compressive experiment, where step load increases in compressive strain (sweeping from 5 to 25%) is applied. The baseline simulation files are provided under `./BaselineFiles/`

The error (objective) function is the MSE error between the simulated RMS-Strain and experimental RMS-Strain points. The goal would be to minimize these differences. The slope values are constrained in the range of $a_{1} = [1.5, 6.0]$ while $\eta_{max} = [50, 300]$ Pa. The grid includes both end points ($\Delta a_1 = 0.5$, $\Delta \eta_{max} \approx 27.8$ Pa).

## File Architecture 

- `Generator/generate_svmp_files.py`: builds the grid in `Samples/s001 ... s100` (solver.xml + scaled activation waveform per point) and `Samples/samples.txt` (sample, i_eta, i_a1, Eta_Max, a1)
- `driver.py` / `run_functions.py`: sweep worker. Runs pending samples one after another and records the outcome of each in `Samples/sNNN/status` and `Samples/sweep_log.csv`
- `submit_sweep.sbatch`: Sherlock job script that runs one worker
- `submit_postprocess.sbatch`: Sherlock job script that runs both post-processing scripts on a compute node
- `PostProcessor/compute_reaction_forces.py`: platen reaction forces for every `DONE` sample
- `PostProcessor/compute_RMS.py`: Force RMS per strain step for every sample, written to `Samples/rms_summary.csv`

## Running the Sweep on Sherlock

A full run writes ~2 GB of VTU files (~200 GB for the whole grid), so put the repository on `$SCRATCH`, not `$HOME`.

```bash
python3 Generator/generate_svmp_files.py     # once (needs numpy)
sbatch --array=1-4 submit_sweep.sbatch       # 4 parallel 24 h workers
python3 driver.py status                     # progress (-v lists every sample)
```

Each worker claims the next unclaimed sample (an atomic `.claim` directory in the sample folder), so any number of workers can share `Samples/`, and resubmitting later only picks up what is left. A run is stopped and the worker moves on to the next sample when:

1. `histor.dat` contains `WARNING: The linear system solution has not converged` → `DIVERGED`
2. it has run for more than 2 hours (`--time-limit`) → `TIMEOUT`

A worker won't start a new run with less than 40 minutes of wall time left. A run that is cut off by the job's wall time (or by `scancel`) goes back to `PENDING` rather than being marked as failed. A solver exiting with a non-zero code is marked `FAILED`.

```bash
python3 driver.py reset --stale                 # free samples whose job died (e.g. node failure)
python3 driver.py reset --status TIMEOUT FAILED # queue those samples again
```

After the sweep (needs pyvista, scipy, matplotlib), run the post-processing on a compute node, since the login nodes don't have enough memory:

```bash
sbatch submit_postprocess.sbatch             # reaction forces, then rms_summary.csv
```

This runs `compute_reaction_forces.py` followed by `compute_RMS.py`. Samples that already have `net_reaction_force.csv` are skipped, so it is safe to resubmit as more samples finish.

### Internal Notes to Consider 
- RMS: Double check that the experimental RMS is the same as the computational RMS 
- Time Window Considerations (only consider the steady plateau of each strain step and not the ramps)
- Increase the time resolution of simulations (step size should be 0.02 instead of 0.1 s) as the twitch happens quickly 
- Only load to the 25% (stop run at 28s),. No need to simulate the unloading phase 
- Compare against the experimental error bars at each of the 5 RMS outputs rather than only a scalar MSE 
