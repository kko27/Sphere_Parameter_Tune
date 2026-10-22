#!/bin/bash
# Post-processing of one sample on Stampede3, called by driver.py (--postprocess)
# right after the sample finishes with DONE, while its VTUs are still on $SCRATCH:
#   section_force.csv                      compute_reaction_forces.py
#   frank_starling.csv, frank_starling.png plot_CM_FS.py
# then copies the sample (without VTUs) to $WORK_REPO/Samples/<sample>/.
#
#   PostProcessor/postprocess_sample_stampede.sh Samples/s001
#
# Anything that fails here is redone by submit_postprocess_stampede.sh.

set -euo pipefail

SAMPLE_DIR=$(cd "$1" && pwd)
source "$(dirname "$0")/stampede_env.sh"

python3 "$RUN_ROOT/PostProcessor/compute_reaction_forces.py" "$SAMPLE_DIR"
python3 "$RUN_ROOT/PostProcessor/plot_CM_FS.py" "$SAMPLE_DIR"

copy_to_work "Samples/$(basename "$SAMPLE_DIR")"
