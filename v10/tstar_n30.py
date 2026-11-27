"""Pilot: scrambling clock t* for the N = 30 design's 21-qubit hole (probe only; needs ~22 qubits).
Seeds 1000-1009. Writes tstar_n30.json."""
import json
import numpy as np
import v10

A = np.array([v10.tstar_probe(21, sd, max_layers=14)[0] for sd in range(1000, 1010)])
m = A.mean(0)
plat = m[-6:].mean()
ts = int(np.argmax(m >= plat - 0.05))
print('t* =', ts, 'steps; plateau', round(plat, 3))
json.dump(dict(curve=m.tolist(), tstar=ts, plateau=float(plat)), open('tstar_n30.json', 'w'))
