# Sourced (not run) by the Stampede3 post-processing scripts, from the run copy of the
# repository on $SCRATCH.  Sets up the python venv and defines copy_to_work, which
# copies results back to the repository on $WORK ($WORK_REPO, set by launch_stampede.sh).

RUN_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export RESULTS_SUBDIR=${RESULTS_SUBDIR:-48-procs}     # svMultiPhysics output dir of a 48-rank run

# `module` is a shell function; a script started by driver.py may not have it
type module >/dev/null 2>&1 || source "$LMOD_PKG/init/bash"

# The sweep job loads TACC's vtk module, which would shadow the venv's vtk wheel
# (pyvista then fails to import vtkCommonMath).
module unload vtk 2>/dev/null || true
unset PYTHONPATH

# numpy, scipy, matplotlib and pyvista all live in the venv
module load python/3.12.11
source /work2/11985/kbko/stampede3/python-envs/test/bin/activate

export MPLBACKEND=Agg          # no display on compute nodes
export OMP_NUM_THREADS=1       # serial post-processing

# copy_to_work <path relative to the repository root>
# Copies a file or directory to the same place under $WORK_REPO, leaving out the
# VTU/VTP results and restart files (these stay on $SCRATCH).
copy_to_work() {
    if [ -z "${WORK_REPO:-}" ] || [ "$WORK_REPO" -ef "$RUN_ROOT" ]; then
        echo "WORK_REPO not set (or same as this directory); not copying $1"
        return 0
    fi
    local dest="$WORK_REPO/$(dirname "$1")"
    mkdir -p "$dest"
    rsync -a --exclude='*.vtu' --exclude='*.vtp' --exclude='*.bin' --exclude='.claim' \
        "$RUN_ROOT/$1" "$dest/"
}
