#!/bin/bash
# Stampede3 version of submit_postprocess.sbatch.
# Final post-processing, queued by launch_stampede.sh to start once every sweep
# worker has ended (--dependency=afterany).  Runs in the $SCRATCH copy:
#   1. section_force.csv and frank_starling.csv/png for any DONE sample the
#      workers did not get to (e.g. a worker hit its wall time)
#   2. compute_RMS.py         -> Samples/rms_summary.csv, rms_reaction_force.png,
#                                Samples/sNNN/48-procs/reaction_force_over_time.png
#   3. plot_RMS_heatmaps.py   -> PostProcessor/rms_heatmaps.png, rms_error_heatmaps.png
#   4. copies all of it (no VTUs) and the job logs to $WORK_REPO
#
# Resubmitting is safe (from the $SCRATCH copy, with WORK_REPO set); finished
# outputs are skipped:
#   WORK_REPO=$WORK/Sphere_Parameter_Tune sbatch submit_postprocess_stampede.sh
#
# Note: skx bills the whole node (for the time actually used, not -t), even though
# this job is serial.

#SBATCH -J EtaA1Post                # Job name
#SBATCH -o postprocess_%j.out       # Name of stdout output file
#SBATCH -e postprocess_%j.err       # Name of stderr error file
#SBATCH -p skx                      # Queue (partition) name
#SBATCH -N 1                        # Total # of nodes
#SBATCH -n 1                        # Total # of tasks
#SBATCH -t 08:00:00                 # Run time (hh:mm:ss)
#SBATCH --mail-user=kbko@stanford.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source PostProcessor/stampede_env.sh

# Fail fast if the venv is missing a package or was built with another python
python3 -c "import sys, numpy, scipy, matplotlib, pyvista; \
print('python', sys.version.split()[0], '| numpy', numpy.__version__, '| scipy', scipy.__version__, \
'| matplotlib', matplotlib.__version__, '| pyvista', pyvista.__version__)"

python3 driver.py status

# 1. Per-sample outputs the workers did not produce
python3 PostProcessor/compute_reaction_forces.py || echo "some samples failed; see above"
for s in Samples/s[0-9]*; do
    if [ "$(head -n 1 "$s/status" 2>/dev/null)" != DONE ] || [ -f "$s/$RESULTS_SUBDIR/frank_starling.csv" ]; then
        continue
    fi
    echo "=== plot_CM_FS $s"
    python3 PostProcessor/plot_CM_FS.py "$s" || echo "plot_CM_FS failed for $s"
done

# 2-3. Grid-wide summaries
python3 PostProcessor/compute_RMS.py
python3 PostProcessor/plot_RMS_heatmaps.py

# 4. Copy to $WORK: every sample (DIVERGED/FAILED ones too, for their logs), the summaries and the job logs
for s in Samples/s[0-9]*; do
    copy_to_work "$s"
done
for f in Samples/samples.txt Samples/sweep_log.csv Samples/rms_summary.csv Samples/rms_reaction_force.png \
         PostProcessor/rms_heatmaps.png PostProcessor/rms_error_heatmaps.png; do
    if [ -e "$f" ]; then
        copy_to_work "$f"
    fi
done
if [ -n "${WORK_REPO:-}" ]; then
    mkdir -p "$WORK_REPO/logs"
    cp sweep_*.out sweep_*.err postprocess_*.out postprocess_*.err "$WORK_REPO/logs/" 2>/dev/null || true
fi
echo "post-processing finished; results copied to ${WORK_REPO:-<WORK_REPO not set>}"
