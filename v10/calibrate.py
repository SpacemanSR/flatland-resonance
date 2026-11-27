"""Pilot amendment: calibrate the turn rule on H_L (baseline) curves only.
Rule v2: a turn is a peak in the rise rate s(k) = D(k+1) - D(k) with prominence >= f * (2 m) bits,
where m = qubits per layer (2m bits = one layer's full information). f is the smallest grid value
for which baseline runs show exactly one turn in >= 95% of seeds (so p_L <= 0.05)."""
import json
import numpy as np
from scipy.signal import find_peaks
import v10

CONFIGS = {15: (9, [1, 1, 1]), 20: (8, [2, 2, 2])}
SEEDS = range(1000, 1030)
F_GRID = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]


def count_turns(D, prom):
    s = np.concatenate([[0.0], np.diff(D), [0.0]])
    pk, _ = find_peaks(s, prominence=prom)
    return len(pk), pk - 1


curves = {}
for N, (nc, layers) in CONFIGS.items():
    curves[N] = np.array([v10.run_schedule(nc, layers, sd, [0] * 3, settle=8)[0] for sd in SEEDS])

res = {}
for N, (nc, layers) in CONFIGS.items():
    m = layers[0]
    rows = []
    for f in F_GRID:
        T = [count_turns(D, f * 2 * m)[0] for D in curves[N]]
        rows.append(dict(f=f, one=float(np.mean([t == 1 for t in T])), p_L=float(np.mean([t >= 2 for t in T])),
                         zero=float(np.mean([t == 0 for t in T]))))
    res[N] = rows
    for r in rows:
        print(N, r)

ok = [f for f in F_GRID if all(next(r for r in res[N] if r['f'] == f)['one'] >= 0.95 for N in CONFIGS)]
print('f passing at both N:', ok)
json.dump(dict(grid=res, passing=ok, curves={N: np.round(c, 4).tolist() for N, c in curves.items()}),
          open('calibration.json', 'w'))
