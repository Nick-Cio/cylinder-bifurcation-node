"""Run the whole sweep as N independent single-core processes.

One training run uses about 1.2 cores, so on a 14-core M4 Pro a serial sweep
leaves most of the machine idle.  This keeps `--workers` of them in flight.

Each job writes its own file and skips if that file already exists, so the sweep
is resumable: kill it, rerun it, and it picks up where it stopped.
"""
import argparse, subprocess, sys, time, itertools
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parent.parent
ALL_RE = [30, 36, 41, 45, 48, 51, 54, 58, 62, 67, 73, 80, 90, 102, 118, 138, 160]


def run(job, py, out, quiet):
    re_, modes, seed, steps = job
    tag = f'Re{re_:g}_r{modes}_seed{seed}_{steps}'
    if (out / f'{tag}.pkl').exists():
        return tag, 'cached', 0.0
    t0 = time.time()
    cmd = [py, str(ROOT / 'sweep' / 'train_one.py'), '--re', str(re_),
           '--modes', str(modes), '--seed', str(seed), '--steps', str(steps),
           '--out', str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return tag, f'FAILED: {r.stderr.strip().splitlines()[-1] if r.stderr else "?"}', time.time()-t0
    if not quiet and r.stdout:
        print('  ' + r.stdout.strip(), flush=True)
    return tag, 'done', time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--modes', type=int, default=14)
    ap.add_argument('--steps', type=int, default=15000)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--re', type=float, nargs='+', default=None,
                    help='default: all 17 Reynolds numbers')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out', type=Path, default=ROOT / 'sweep' / 'runs')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args()

    res = a.re if a.re is not None else ALL_RE
    jobs = [(r, a.modes, s, a.steps) for r in res for s in a.seeds]
    a.out.mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs
            if not (a.out / f'Re{j[0]:g}_r{j[1]}_seed{j[2]}_{j[3]}.pkl').exists()]

    # measured on an M4 Pro: 2000 steps at 14 modes takes 18.6 s alone, and eight
    # concurrent single-thread jobs finish in 24.2 s, i.e. 6.15x throughput.
    per = (18.6 / 60) * (a.steps / 2000) * (1.0 + 0.02 * (a.modes - 14))
    speedup = min(a.workers, 6.15 * a.workers / 8)
    print(f'{len(jobs)} jobs ({len(res)} Re x {len(a.seeds)} seeds), '
          f'{len(jobs)-len(todo)} already done, {len(todo)} to run')
    print(f'{a.modes} modes, {a.steps:,} steps, {a.workers} workers')
    print(f'estimate: ~{per:.1f} min per job alone, ~{per*len(todo)/speedup:.0f} min ({per*len(todo)/speedup/60:.1f} h) wall clock at {a.workers} workers\n')
    if a.dry_run or not todo:
        return

    py = str(ROOT / '.venv' / 'bin' / 'python')
    t0 = time.time(); done = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for tag, status, secs in ex.map(lambda j: run(j, py, a.out, a.quiet), todo):
            done += 1
            el = (time.time()-t0)/60
            eta = el/done*(len(todo)-done)
            flag = '' if status == 'done' else f'  <-- {status}'
            print(f'[{done}/{len(todo)}] {tag}  {secs/60:.1f} min   '
                  f'elapsed {el:.0f} min, eta {eta:.0f} min{flag}', flush=True)
    print(f'\nfinished in {(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
