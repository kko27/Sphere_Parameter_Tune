#!/bin/bash
# Launch the (Eta_Max, a1) sweep on Stampede3.  Run on a login node from the
# repository on $WORK (after Generator/generate_svmp_files.py has made Samples/):
#
#   ./launch_stampede.sh          # 4 sweep workers
#   ./launch_stampede.sh 8        # 8 sweep workers
#
# 1. Copies the code, BaselineFiles and the sample inputs to $SCRATCH/<repo name>.
#    Run state already on $SCRATCH (status, claims, results) is never overwritten,
#    so running this again just adds workers for whatever is still pending.
# 2. Submits the sweep workers (submit_sweep_stampede.sh) from there.
# 3. Queues submit_postprocess_stampede.sh to start once all workers have ended.
# Results (no VTUs) are copied back into this repository: Samples/sNNN/, Samples/rms_summary.csv,
# PostProcessor/rms_*heatmaps.png, logs/.  The VTUs stay on $SCRATCH.
#
# Progress:   cd $SCRATCH/<repo name> && python3 driver.py status

set -euo pipefail

N_WORKERS=${1:-4}
WORK_REPO=$(cd "$(dirname "$0")" && pwd)
RUN_DIR=$SCRATCH/$(basename "$WORK_REPO")

if [ ! -f "$WORK_REPO/Samples/samples.txt" ]; then
    echo "No Samples/samples.txt; run Generator/generate_svmp_files.py first." >&2
    exit 1
fi

echo "Staging $WORK_REPO -> $RUN_DIR"
mkdir -p "$RUN_DIR/Samples"
rsync -a --exclude='.git' --exclude='Samples/' --exclude='Samples_*/' --exclude='FullSimulations/' \
    --exclude='sherlock' --exclude='logs/' --exclude='*.out' --exclude='*.err' \
    "$WORK_REPO/" "$RUN_DIR/"
# Sample inputs only
rsync -a --include='samples.txt' --include='s*/' --include='solver.xml' --include='activation_cardioid.dat' \
    --exclude='*' "$WORK_REPO/Samples/" "$RUN_DIR/Samples/"

cd "$RUN_DIR"
export WORK_REPO

# TACC's sbatch prints a banner, so keep only the job id
submit() { sbatch --parsable --export=ALL "$@" | grep -Eo '^[0-9]+' | tail -n 1; }

workers=()
for i in $(seq 1 "$N_WORKERS"); do
    workers+=("$(submit submit_sweep_stampede.sh)")
done
dep=$(IFS=:; echo "${workers[*]}")
post=$(submit --dependency=afterany:"$dep" submit_postprocess_stampede.sh)

echo "Sweep workers:   ${workers[*]}"
echo "Post-processing: $post (starts after all workers end)"
echo "Run directory:   $RUN_DIR"
echo "Results to:      $WORK_REPO"
