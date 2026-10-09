"""Driver for the (Eta_Max, a1) grid sweep.

Each call to `run` is one worker: it repeatedly claims the next pending sample in
Samples/, runs svMultiPhysics on it, and records the outcome, until no pending
samples are left or the Slurm job is close to its wall time.  Any number of
workers (Slurm jobs) can share the same Samples/ directory.

A run is stopped early and the worker moves on when
  - histor.dat reports "The linear system solution has not converged" -> DIVERGED
  - the run exceeds --time-limit (default 2 h)                         -> TIMEOUT

Usage (from the repository root):
    python3 driver.py run --exe /path/to/svmultiphysics      # one worker
    python3 driver.py status                                  # progress table
    python3 driver.py reset --stale                           # free claims of dead jobs
    python3 driver.py reset --status TIMEOUT FAILED           # queue those again
"""

import argparse
import os
import signal
import sys
from collections import Counter

import run_functions as rf


def cmd_run(args):
    job_id = os.environ.get('SLURM_JOB_ID')
    nprocs = args.nprocs or int(os.environ.get('SLURM_NTASKS', '1'))
    # svMultiPhysics writes its results and histor.dat to "<nprocs>-procs/"
    results_dir = '{}-procs'.format(nprocs)
    if args.launcher == 'srun':
        cmd = ['srun', args.exe, 'solver.xml']
    else:
        cmd = ['mpirun', '-n', str(nprocs), args.exe, 'solver.xml']

    signal.signal(signal.SIGTERM, rf.raise_terminated)
    print('Worker job={} nprocs={} time_limit={}s'.format(job_id, nprocs, args.time_limit), flush=True)

    n_run = 0
    for sample_dir in rf.list_samples():
        name = os.path.basename(sample_dir)
        if rf.read_status(sample_dir) != rf.PENDING:
            continue

        # Never start a run that the job's wall time can't accommodate.  The
        # per-run limit shrinks near the end of the job; a run cut short by that
        # (rather than by the 2 h limit) is put back in the pending pool.
        time_left = rf.slurm_time_left(job_id)
        limit = args.time_limit
        if time_left is not None:
            budget = time_left - args.margin
            if budget < args.min_start:
                print('Only {:.0f} min of wall time left; stopping.'.format(time_left / 60), flush=True)
                break
            limit = min(limit, budget)

        if not rf.try_claim(sample_dir, job_id):
            continue

        print('[{}] start (limit {:.0f} min)'.format(name, limit / 60), flush=True)
        try:
            status, elapsed = rf.run_simulation(sample_dir, cmd, results_dir, limit, args.poll)
        except (rf.Terminated, KeyboardInterrupt):
            print('[{}] interrupted; returning it to the queue'.format(name), flush=True)
            rf.release_claim(sample_dir)
            sys.exit(1)

        if status == rf.TIMEOUT and limit < args.time_limit:
            print('[{}] cut off by job wall time after {:.0f} min; returning it to the queue'
                  .format(name, elapsed / 60), flush=True)
            rf.release_claim(sample_dir)
            rf.append_log(job_id, sample_dir, 'REQUEUED', elapsed)
            break

        rf.write_status(sample_dir, status, job=job_id, elapsed_s='{:.0f}'.format(elapsed))
        rf.append_log(job_id, sample_dir, status, elapsed)
        print('[{}] {} after {:.1f} min'.format(name, status, elapsed / 60), flush=True)
        n_run += 1

    print('Worker finished: {} simulations run.'.format(n_run), flush=True)


def cmd_status(args):
    samples = rf.list_samples()
    statuses = {s: rf.read_status(s) for s in samples}
    counts = Counter(statuses.values())
    print('  '.join('{}={}'.format(k, counts.get(k, 0))
                    for k in (rf.PENDING, rf.RUNNING) + rf.FINISHED))
    if args.verbose:
        for s, st in statuses.items():
            owner = rf.read_owner(s) if st != rf.PENDING else ''
            print('{:6s} {:9s} {}'.format(os.path.basename(s), st, owner or ''))


def cmd_reset(args):
    targets = set(args.status or [])
    n_reset = 0
    for s in rf.list_samples():
        st = rf.read_status(s)
        reset = st in targets
        if args.stale and st == rf.RUNNING:
            owner = rf.read_owner(s)
            reset = owner is None or not rf.slurm_job_alive(owner)
        if reset:
            rf.release_claim(s)
            n_reset += 1
            print('reset {} ({})'.format(os.path.basename(s), st))
    print('{} samples returned to PENDING.'.format(n_reset))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command')
    sub.required = True

    p_run = sub.add_parser('run', help='work through pending samples')
    p_run.add_argument('--exe', required=True, help='path to the svmultiphysics binary')
    p_run.add_argument('--launcher', choices=['srun', 'mpirun'], default='srun')
    p_run.add_argument('--nprocs', type=int, help='MPI ranks (default: $SLURM_NTASKS)')
    p_run.add_argument('--time-limit', type=float, default=2 * 3600,
                       help='max seconds per simulation before it is killed (default 7200)')
    p_run.add_argument('--min-start', type=float, default=40 * 60,
                       help='do not start a run with less than this many seconds available (default 2400)')
    p_run.add_argument('--margin', type=float, default=10 * 60,
                       help='seconds kept free before the job wall time for clean-up (default 600)')
    p_run.add_argument('--poll', type=float, default=30, help='seconds between histor.dat checks')
    p_run.set_defaults(func=cmd_run)

    p_status = sub.add_parser('status', help='summarise sample states')
    p_status.add_argument('-v', '--verbose', action='store_true', help='list every sample')
    p_status.set_defaults(func=cmd_status)

    p_reset = sub.add_parser('reset', help='return samples to PENDING')
    p_reset.add_argument('--stale', action='store_true',
                         help='RUNNING samples whose Slurm job is no longer alive')
    p_reset.add_argument('--status', nargs='+', choices=rf.FINISHED,
                         help='samples with these final states (e.g. TIMEOUT FAILED)')
    p_reset.set_defaults(func=cmd_reset)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
