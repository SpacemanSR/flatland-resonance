"""Pilot 2 (after the literature check, 27 Nov 2026): baselines only, for the redesigned trial.

Design changes it calibrates:
  - Every release is followed by t* = 4 scrambling layers before the next emission
    (Hayden-Preskill instantaneous-scrambler idealization; removes a finite-size leak).
  - Two baselines: H_L young (all matter in at step 0) and H_L old (all matter released together
    just past the core's halfway point). Both must show exactly one turn.
  - Layers of m = 1 qubit (N = 15, 20) and m = 2 (N = 20), so the threshold's scaling with m is checked.
Seeds 2001-2030 (seed 2000 was used for one debugging run and is excluded everywhere).
Writes pilot2_results.json."""
import json, time
import numpy as np
from scipy.signal import find_peaks
import v10

TSTAR = 4
CONFIGS = {'N15_m1': (9, [1, 1, 1]), 'N20_m1': (14, [1, 1, 1]), 'N20_m2': (8, [2, 2, 2])}
SEEDS = range(2001, 2031)
F_GRID = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]


def count_turns(D, prom):
    s = np.concatenate([[0.0], np.diff(D), [0.0]])
    pk, _ = find_peaks(s, prominence=prom)
    return len(pk)


out = {}
for name, (nc, layers) in CONFIGS.items():
    t0 = time.time()
    m = layers[0]
    kp = v10.page_step(nc)
    arms = {
        'HL_young': dict(release_steps=[0] * 3, settle=TSTAR, mix_on_join=TSTAR),
        'HL_old': dict(release_steps=[kp] * 3, settle=0, mix_on_join=TSTAR),
    }
    res = {'n_core': nc, 'layers': layers, 'page_step': kp}
    for arm, kw in arms.items():
        Ds = np.array([v10.run_schedule(nc, layers, sd, **kw)[0] for sd in SEEDS])
        rows = []
        for f in F_GRID:
            T = [count_turns(D, f * 2 * m) for D in Ds]
            rows.append(dict(f=f, one=float(np.mean([t == 1 for t in T])),
                             two_plus=float(np.mean([t >= 2 for t in T])),
                             zero=float(np.mean([t == 0 for t in T]))))
        res[arm] = dict(grid=rows, D_mean=np.round(Ds.mean(0), 3).tolist(),
                        final_D=float(Ds[:, -1].mean()), expected_final=2 * sum(layers))
        print(name, arm, 'final D', round(res[arm]['final_D'], 3), '/', 2 * sum(layers), flush=True)
        for r in rows:
            print('   ', r, flush=True)
    res['seconds'] = round(time.time() - t0, 1)
    out[name] = res

passing = [f for f in F_GRID
           if all(next(r for r in out[c][a]['grid'] if r['f'] == f)['one'] >= 0.95
                  for c in CONFIGS for a in ('HL_young', 'HL_old'))]
out['passing_f'] = passing
print('f passing for every config and both baselines:', passing)

# scrambling clock at the new hole sizes (probe only)
tst = {}
for n_hole in (17, 22):
    A = np.array([v10.tstar_probe(n_hole, sd, max_layers=12)[0] for sd in range(2001, 2011)])
    mcurve = A.mean(0)
    plat = mcurve[-5:].mean()
    tst[n_hole] = int(np.argmax(mcurve >= plat - 0.05))
    print('t* at hole', n_hole, '=', tst[n_hole], flush=True)
out['tstar'] = tst
json.dump(out, open('pilot2_results.json', 'w'), indent=1)
