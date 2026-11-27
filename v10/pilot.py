"""Pilot (Protocol section 8): H_L only, N = 15 and 20, 30 seeds each, seeds 1000-1029.
Sets t* (section 4) and sigma (section 6); estimates the false-positive rate p_L of the turn rule."""
import json, time, sys
import numpy as np
import v10

CONFIGS = {15: (9, [1, 1, 1]), 20: (8, [2, 2, 2])}
SEEDS = range(1000, 1030)
SETTLE = 8          # layers; > t* so the baseline starts fully scrambled


def plateau_step(curves, tol=0.05):
    m = curves.mean(axis=0)
    plat = m[-10:].mean()
    return int(np.argmax(m >= plat - tol)), float(plat)


def turns(D, sigma, flat=2, rise=0.5, ahead=3):
    s = np.diff(D)
    T = []
    for k in range(flat, len(s)):
        if s[k] >= 3 * sigma and all(s[k - j] < sigma for j in range(1, flat + 1)):
            if D[min(k + ahead, len(D) - 1)] - D[k] >= rise:
                T.append(k)
    return T


out = {}
for N, (nc, layers) in CONFIGS.items():
    t0 = time.time()
    n_hole = nc + sum(layers)
    A, B, Ds = [], [], []
    for sd in SEEDS:
        a, b = v10.tstar_probe(n_hole, sd)
        A.append(a); B.append(b)
        D, _ = v10.run_schedule(nc, layers, sd, [0] * len(layers), settle=SETTLE)
        Ds.append(D)
    A, B, Ds = np.array(A), np.array(B), np.array(Ds)
    ts_a, pa = plateau_step(A)
    ts_b, pb = plateau_step(B)
    # sigma: slope spread before the Page time, where D is flat
    kflat = n_hole // 2 - 2
    s = np.diff(Ds, axis=1)[:, :kflat]
    sigma = float(s.std())
    T = [turns(D, sigma) for D in Ds]
    nT = [len(t) for t in T]
    pL = float(np.mean([n >= 2 for n in nT]))
    out[N] = dict(n_hole=n_hole, tstar_protocol=ts_a, plateau_protocol=pa,
                  tstar_HP=ts_b, plateau_HP=pb, kflat=kflat, sigma=sigma,
                  turn_counts=nT, first_turn=[t[0] if t else None for t in T], p_L=pL,
                  D_mean=np.round(Ds.mean(0), 3).tolist(), D_sd=np.round(Ds.std(0), 3).tolist(),
                  final_D_mean=float(Ds[:, -1].mean()), expected_final=2 * sum(layers),
                  seconds=round(time.time() - t0, 1))
    print(N, {k: v for k, v in out[N].items() if k not in ('D_mean', 'D_sd', 'turn_counts', 'first_turn')}, flush=True)
    print('  turns per seed:', nT, flush=True)
    print('  D mean:', out[N]['D_mean'], flush=True)

json.dump(out, open('pilot_results.json', 'w'), indent=1)
