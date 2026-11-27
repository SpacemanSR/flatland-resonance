"""Negative control for crossval.py: replay one run with a deliberate porting bug (qubit order of
two-qubit gates not reversed). The comparison must flag it, or the cross-validation proves nothing."""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator
import v10
from crossval import S2_qiskit

D, _, tr = v10.run_schedule(14, [1, 1, 1], 3001, [v10.page_step(14)] * 3, mix_on_join=4, trace=True)
N = tr['N']
qc = QuantumCircuit(N)
done = 0
for i, (n_g, _) in enumerate(tr['checkpoints']):
    for U, qs in tr['log'][done:n_g]:
        qc.append(UnitaryGate(U, check_input=False), list(qs))      # BUG: order not reversed
    done = n_g
    qc.save_statevector(label=f'k{i}')
res = AerSimulator(method='statevector').run(qc, shots=1).result()
Dq = [S2_qiskit(res.data(0)[f'k{i}'], tr['ref'], N) + S2_qiskit(res.data(0)[f'k{i}'], rad, N)
      - S2_qiskit(res.data(0)[f'k{i}'], tr['ref'] + rad, N) for i, (_, rad) in enumerate(tr['checkpoints'])]
psi_q = np.transpose(tr['psi'], list(range(N))[::-1]).reshape(-1)
F = abs(np.vdot(psi_q, np.asarray(res.data(0)[f'k{len(Dq)-1}'].data))) ** 2
print('with deliberate bug: max|dD| = %.3f bits, final fidelity = %.3e' % (np.max(np.abs(D - np.array(Dq))), F))
