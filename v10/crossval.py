"""Cross-validation of v10 against Qiskit Aer (an independent, widely used state-vector simulator).

For each run, v10 exports its complete gate sequence. The same circuit is rebuilt in Qiskit
(UnitaryGate per v10 gate) and simulated with qiskit-aer's AerSimulator. D(k) = I2(R_M : R) is then
recomputed at every step with Qiskit's own partial_trace and purity, and the final state vectors are
compared by fidelity. Nothing from v10's entropy code is reused on the Qiskit side.

Only baseline arms (H_L young, H_L old) are used, with validation seeds 3001-3003, disjoint from the
pilots and from confirmatory seeds. Writes crossval_results.json."""
import json, time, platform
import numpy as np
import qiskit, qiskit_aer
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import partial_trace
from qiskit_aer import AerSimulator
import v10

TSTAR = 4
CONFIGS = {'N15_m1': (9, [1, 1, 1]), 'N20_m1': (14, [1, 1, 1]), 'N20_m2': (8, [2, 2, 2])}
SEEDS = (3001, 3002, 3003)


def S2_qiskit(sv, keep, N):
    keep = sorted(set(keep))
    if not keep or len(keep) == N:
        return 0.0
    if len(keep) > N / 2:
        keep = [q for q in range(N) if q not in keep]
    traced = [q for q in range(N) if q not in keep]
    return float(-np.log2(partial_trace(sv, traced).purity().real))


def replay(trace):
    N = trace['N']
    qc = QuantumCircuit(N)
    done = 0
    for i, (n_gates, _) in enumerate(trace['checkpoints']):
        for U, qs in trace['log'][done:n_gates]:
            # v10 puts the first listed qubit as the most significant index; Qiskit the least
            qc.append(UnitaryGate(U, check_input=False), list(reversed(qs)))
        done = n_gates
        qc.save_statevector(label=f'k{i}')
    sim = AerSimulator(method='statevector', precision='double')
    res = sim.run(qc, shots=1).result()
    return [res.data(0)[f'k{i}'] for i in range(len(trace['checkpoints']))]


def main():
    out = dict(environment=dict(python=platform.python_version(), numpy=np.__version__,
                                qiskit=qiskit.__version__, qiskit_aer=qiskit_aer.__version__,
                                machine=platform.machine(), system=platform.system()), runs=[])
    worst_D, worst_F = 0.0, 1.0
    for name, (nc, layers) in CONFIGS.items():
        kp = v10.page_step(nc)
        arms = {'HL_young': dict(release_steps=[0] * 3, settle=TSTAR, mix_on_join=TSTAR),
                'HL_old': dict(release_steps=[kp] * 3, settle=0, mix_on_join=TSTAR)}
        for arm, kw in arms.items():
            for sd in SEEDS:
                t0 = time.time()
                D, _, tr = v10.run_schedule(nc, layers, sd, trace=True, **kw)
                svs = replay(tr)
                N = tr['N']
                Dq = []
                for (n_g, rad), sv in zip(tr['checkpoints'], svs):
                    Dq.append(S2_qiskit(sv, tr['ref'], N) + S2_qiskit(sv, rad, N)
                              - S2_qiskit(sv, tr['ref'] + rad, N))
                Dq = np.array(Dq)
                psi_q = np.transpose(tr['psi'], list(range(N))[::-1]).reshape(-1)   # v10 -> Qiskit ordering
                F = float(abs(np.vdot(psi_q, np.asarray(svs[-1].data))) ** 2)
                dmax = float(np.max(np.abs(D - Dq)))
                worst_D, worst_F = max(worst_D, dmax), min(worst_F, F)
                out['runs'].append(dict(config=name, arm=arm, seed=sd, gates=len(tr['log']), steps=len(D),
                                        max_abs_D_diff_bits=dmax, final_state_fidelity=F,
                                        seconds=round(time.time() - t0, 1)))
                print(name, arm, sd, 'gates', len(tr['log']), 'max|dD| = %.2e bits' % dmax,
                      'fidelity = %.15f' % F, flush=True)
    out['worst_max_abs_D_diff_bits'] = worst_D
    out['worst_final_state_fidelity'] = worst_F
    print('WORST: max|dD| = %.2e bits, min fidelity = %.15f' % (worst_D, worst_F))
    json.dump(out, open('crossval_results.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
