"""Preregistered analysis (Protocol sections 6-7). Reads the runner's JSON files and reports
p(delta) per condition, Delta_c(m) per register, and the outcome by the decision rules.
Fixed before any confirmatory data exist; f = 0.3 primary, 0.25 and 0.4 as sensitivity checks."""
import glob, json, sys
from collections import defaultdict
import numpy as np
from scipy.signal import find_peaks
import conditions

F_PRIMARY, F_SENS = 0.3, (0.25, 0.4)


def turns(D, m, f):
    s = np.concatenate([[0.0], np.diff(np.asarray(D)), [0.0]])
    return len(find_peaks(s, prominence=f * 2 * m)[0])


def load(folder):
    data = defaultdict(dict)
    for path in glob.glob(f'{folder}/*.json'):
        d = json.load(open(path))
        if 'runs' in d:
            data[d['register']][d['condition']] = d
    return data


def staggered_share(cond, m, f):
    return float(np.mean([turns(r['D'], m, f) >= 2 for r in cond['runs']]))


def delta_c(reg, m, f):
    """Smallest swept delta with p >= 0.80 in H_F old, provided H_L old's staggered share <= 0.05."""
    if 'HL_old' not in reg or staggered_share(reg['HL_old'], m, f) > 0.05:
        return None, 'baseline false-positive rate above 0.05'
    dmax = conditions.OLD_DMAX
    for d in range(1, 99):
        c = reg.get(f'HF_old_d{d}')
        if c is None:
            break
        if staggered_share(c, m, f) >= 0.80:
            return d, None
    return float('inf'), None


def outcome(data, f):
    # Halt checks
    for name, reg in data.items():
        m = conditions.REGISTERS[name][1]
        for lab, c in reg.items():
            if lab.startswith('HF_young') and staggered_share(c, m, f) > 0.20:
                return 'Halt', f'{name} {lab}: young-hole control staggered in more than 20% of seeds'
        if 'HS' in reg:
            ok = np.mean([turns(r['D'], m, f) == 3 for r in reg['HS']['runs']])
            if ok < 1.0:
                return 'Halt', f'{name} HS: not 3 turns in every seed'
    dc = {}
    for name in ('N25_m1', 'N30_m1', 'N25_m2', 'N30_m2'):
        if name in data:
            dc[name], why = delta_c(data[name], conditions.REGISTERS[name][1], f)
            if why:
                return 'Halt', f'{name}: {why}'
    need = {'N25_m1', 'N30_m1', 'N30_m2'}
    if not need <= set(dc):
        return 'Incomplete', f'missing registers: {sorted(need - set(dc))}'
    a, b = dc['N25_m1'], dc['N30_m1']
    if not (np.isinf(a) and np.isinf(b)) and abs(a - b) > 1:
        return 'Inconclusive', f'Delta_c(1) = {a} at N=25 vs {b} at N=30'
    if np.isinf(b):
        return 'Mechanism absent', 'Delta_c(1) > 5 at N=30'
    if dc['N30_m2'] > b:
        return 'Residual refuted', f"Delta_c(2) = {dc['N30_m2']} > Delta_c(1) = {b} at N=30"
    return 'Residual survives', f"Delta_c(2) = {dc['N30_m2']} <= Delta_c(1) = {b} at N=30"


if __name__ == '__main__':
    data = load(sys.argv[1] if len(sys.argv) > 1 else 'results')
    for f in (F_PRIMARY,) + F_SENS:
        print(f'f = {f}:', *outcome(data, f))
    for name, reg in sorted(data.items()):
        m = conditions.REGISTERS[name][1]
        for lab in sorted(reg):
            print(f'  {name:7s} {lab:13s} staggered share = {staggered_share(reg[lab], m, F_PRIMARY):.2f}')
