"""The confirmatory design, exactly as preregistered (Protocol sections 4-5)."""
import v10

TSTAR = 4
REGISTERS = {          # name: (N, m, core)
    'N20_m1': (20, 1, 14), 'N20_m2': (20, 2, 8),
    'N25_m1': (25, 1, 19), 'N25_m2': (25, 2, 13),
    'N30_m1': (30, 1, 24), 'N30_m2': (30, 2, 18)}
OLD_DMAX = {'N20_m1': 2, 'N20_m2': 1, 'N25_m1': 4, 'N25_m2': 3, 'N30_m1': 5, 'N30_m2': 4}
YOUNG_DELTAS = {'N20_m1': [1, 2, 3], 'N20_m2': [1, 2, 3], 'N25_m1': [1, 2, 3, 4, 5],
                'N25_m2': [1, 2, 3, 4], 'N30_m1': [5], 'N30_m2': [4]}
SEEDS = {'N20': 50, 'N25': 50, 'N30': 20}


def conditions(register):
    """Yield (label, kwargs for v10.run_schedule) for every confirmatory condition of a register."""
    N, m, core = REGISTERS[register]
    layers = [m] * 3
    kp = v10.page_step(core)
    yield 'HL_young', dict(n_core=core, layer_sizes=layers, release_steps=[0, 0, 0], settle=TSTAR, mix_on_join=TSTAR)
    yield 'HL_old', dict(n_core=core, layer_sizes=layers, release_steps=[kp] * 3, settle=0, mix_on_join=TSTAR)
    for d in range(1, OLD_DMAX[register] + 1):
        yield f'HF_old_d{d}', dict(n_core=core, layer_sizes=layers, release_steps=[kp, kp + d, kp + 2 * d],
                                   settle=0, mix_on_join=TSTAR)
    for d in YOUNG_DELTAS[register]:
        yield f'HF_young_d{d}', dict(n_core=core, layer_sizes=layers, release_steps=[0, d, 2 * d],
                                     settle=TSTAR, mix_on_join=TSTAR)
    if register == 'N20_m1':
        yield 'HS', dict(n_core=core, layer_sizes=layers, release_steps=[0, 0, 0], strict=True)


def n_seeds(register):
    return SEEDS[register[:3]]


def total_runs():
    return sum(n_seeds(r) * len(list(conditions(r))) for r in REGISTERS)
