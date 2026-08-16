"""Fold the per-run files into notebooks/node_fits.pkl so the notebooks see them."""
import pickle, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / 'notebooks' / 'node_fits.pkl'
RUNS = ROOT / 'sweep' / 'runs'

cache = pickle.load(open(CACHE, 'rb')) if CACHE.exists() else {}
before = len(cache)
added = skipped = 0
for f in sorted(RUNS.glob('*.pkl')):
    key = f.stem
    if key in cache:
        skipped += 1
        continue
    cache[key] = pickle.load(open(f, 'rb'))
    added += 1
if added:
    tmp = CACHE.with_suffix('.tmp')
    with open(tmp, 'wb') as fh:
        pickle.dump(cache, fh)
    tmp.replace(CACHE)
print(f'{CACHE.name}: {before} -> {len(cache)} keys  ({added} added, {skipped} already present)')
