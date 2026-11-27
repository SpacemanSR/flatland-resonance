"""Confirmatory runner for the preregistered trial (Protocol sections 4-7).

  Timing run (allowed before filing; baseline arm, validation seed 4001):
      python3 run_confirmatory.py --timing --register N30_m1 --backend cupy
  Confirmatory runs (refused without the OSF registration DOI):
      python3 run_confirmatory.py --register N25_m1 --osf 10.17605/OSF.IO/XXXXX --out results/

Each condition writes one JSON file: every seed's D(k) and per-layer traces, the run manifest,
and the OSF registration it ran under. Existing files are skipped, so interrupted runs resume."""
import argparse, json, os, re, time
import numpy as np
import v10, conditions, manifest

p = argparse.ArgumentParser()
p.add_argument('--register', required=True, choices=sorted(conditions.REGISTERS))
p.add_argument('--backend', default='numpy', choices=['numpy', 'cupy'])
p.add_argument('--osf', help='OSF registration DOI; required for confirmatory seeds')
p.add_argument('--out', default='results')
p.add_argument('--timing', action='store_true', help='one H_L old run with validation seed 4001')
a = p.parse_args()
v10.set_backend(a.backend)

if a.timing:
    label, kw = [c for c in conditions.conditions(a.register) if c[0] == 'HL_old'][0]
    t0 = time.time()
    D, per = v10.run_schedule(seed=4001, **kw)
    rec = dict(register=a.register, condition=label, seed=4001, seconds=round(time.time() - t0, 1),
               final_D=float(D[-1]), expected_final=2 * sum(kw['layer_sizes']), manifest=manifest.manifest(a.backend))
    if a.backend == 'cupy':
        import cupy
        rec['gpu_peak_bytes'] = int(cupy.get_default_memory_pool().total_bytes())
    os.makedirs(a.out, exist_ok=True)
    json.dump(rec, open(os.path.join(a.out, f'timing_{a.register}_{a.backend}.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in rec.items() if k != 'manifest'}))
    raise SystemExit

if not a.osf or not re.fullmatch(r'10\.17605/OSF\.IO/[A-Z0-9]{4,8}', a.osf, re.I):
    raise SystemExit('Refusing confirmatory runs: pass --osf with the OSF registration DOI '
                     '(10.17605/OSF.IO/XXXXX). File the preregistration first.')

os.makedirs(a.out, exist_ok=True)
for label, kw in conditions.conditions(a.register):
    path = os.path.join(a.out, f'{a.register}__{label}.json')
    if os.path.exists(path):
        print('skip', path)
        continue
    t0 = time.time()
    runs = []
    for sd in range(1, conditions.n_seeds(a.register) + 1):
        D, per = v10.run_schedule(seed=sd, **kw)
        runs.append(dict(seed=sd, D=np.round(D, 10).tolist(), per_layer=np.round(per, 10).tolist()))
    json.dump(dict(register=a.register, condition=label, schedule={k: v for k, v in kw.items()},
                   osf=a.osf, seconds=round(time.time() - t0, 1), manifest=manifest.manifest(a.backend),
                   runs=runs), open(path, 'w'))
    print('done', path, round(time.time() - t0, 1), 's', flush=True)
