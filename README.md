# Description 

This codebase contains the driver script required to find the parameters $\eta_{max}$ and $a_{1}$, which determine the maximum cardiomyocyte active stress (Pa) and the Frank-Starling Gradient respectively. As inputs, the user provides the Force RMS measurements obtained from the experiments. The expected input file is a .csv text file where the first column contains the compressive strain (%) and the second column contains the Force RMS readouts (𝜇𝑁). 

Geometry files using image segmentation and meshing techniques from the optical images obtained during the experiment and the Youngs Modulus measured from passive characterization are also required as inputs to this parameter tuning procedure. 

The forward simulation runs are relatively expensive (as it requires running the structural simulation from Simvascular svmultiphysics) and can take up to 30 minutes for each solve. For this reason, we make use of Gaussian Process Emulators as a surrogate model to accelerate the parameter search space. An overview of the estimation process is depicted in the figure below. 

<figure>
  <img src=./Figures.pdf" alt="Schematic of the Calibration Workflow">
  <figcaption>Schematic of the Calibration Workflow</figcaption>
</figure>

The forward simulation is a digital twin of the compressive experiment, where step load increases in compressive strain (sweeping from 5 to 25%) is applied. These simulation files are provided under `./ExampleFiles/`

The error (objective) function is the MSE error between the simulated RMS-Strain and experimental RMS-Strain points. The goal would be to minimize these differences. The slope values are constrained in the range of $a_{1} = [1.5, 6.0]$ while $\eta_{max} = [50, 300]$ Pa. 

## File Architecture 





### Internal Notes to Consider 
- RMS: Double check that the experimental RMS is the same as the computational RMS 
- Time Window Considerations (only consider the steady plateau of each strain step and not the ramps)
- Increase the time resolution of simulations (step size should be 0.02 instead of 0.1 s) as the twitch happens quickly 
- Only load to the 25% (stop run at 28s),. No need to simulate the unloading phase 
- Using experimental error bars: Fit the GP to the 5 RMS outputs rather than a scalar MSE 

### Engineering pieces still to build: 
- Driver loop: generate a space-filling design of parameter points (use Latin Hypercube Sampling)
- For each point, write the waveform and XML into its own run directory 
- Launch the simulation, check if it converged, and extract RMS. 
- Fit the GP and run the optimization or Bayesian step 
- Add points near the best fit and repeat 
- Run one confirmation simulation at the final optimum 

Failure Handlig: if simulation stops early or there are NaNs in the output or the linear solver gets stuck. Exclude these points or mark them, rather than letting garbage reach the GP 

Reworking of generate_waveform.py: Multiply the activation function by the scalar \eta_max instead of running it every time 




