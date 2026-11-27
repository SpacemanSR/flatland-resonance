"""Tests for analyze.py on SYNTHETIC D(k) curves (no simulation; no confirmatory data).
Each scenario must produce the outcome the decision rules (Protocol section 7) assign to it."""
import json, os, tempfile
import numpy as np
import analyze, conditions

rng = np.random.default_rng(0)


def ramp(n, starts, m):
    """Synthetic D(k): one smooth 2m-bit rise per entry in `starts`, plus small noise."""
    k = np.arange(n)
    D = sum(2 * m / (1 + np.exp(-(k - s) * 2.0)) for s in starts)
    return (D + rng.normal(0, 0.01, n)).tolist()


def write(folder, reg, label, curves):
    json.dump(dict(register=reg, condition=label, runs=[dict(seed=i + 1, D=c) for i, c in enumerate(curves)]),
              open(os.path.join(folder, f'{reg}__{label}.json'), 'w'))


def scenario(sep_needed):
    """sep_needed[m] = smallest delta at which old-arm layers separate (inf = never)."""
    folder = tempfile.mkdtemp()
    for reg in ('N25_m1', 'N30_m1', 'N25_m2', 'N30_m2'):
        N, m, core = conditions.REGISTERS[reg]
        n = core + 3 * m + 1
        for label, kw in conditions.conditions(reg):
            if label.startswith('HF_old'):
                d = int(label.split('_d')[1])
                starts = [10, 10 + 3 * d, 10 + 6 * d] if d >= sep_needed[m] else [12]
            else:
                starts = [12]
            write(folder, reg, label, [ramp(n + 20, starts, m) for _ in range(20)])
    return analyze.load(folder)


cases = {'Residual refuted': {1: 2, 2: 4}, 'Residual survives': {1: 2, 2: 2},
         'Mechanism absent': {1: float('inf'), 2: float('inf')}}
for expected, sep in cases.items():
    got = analyze.outcome(scenario(sep), analyze.F_PRIMARY)
    print(f'{expected:18s} -> {got[0]:18s} ({got[1]})')
    assert got[0] == expected, got
# Halt: a young-hole control that staggers
data = scenario({1: 2, 2: 4})
for c in data['N30_m1']:
    if c.startswith('HF_young'):
        for r in data['N30_m1'][c]['runs']:
            r['D'] = ramp(len(r['D']), [8, 20, 32], 1)
got = analyze.outcome(data, analyze.F_PRIMARY)
print(f"{'Halt':18s} -> {got[0]:18s} ({got[1]})")
assert got[0] == 'Halt'
print('all analysis tests passed')
