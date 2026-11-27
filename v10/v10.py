"""Flatland Resonance Model v10 -- headless engine for the filtered-unhooking trial.

State-vector simulation of an evaporating qubit hole with bound matter layers.
Implements Protocol sections 4 and 6 of the preregistration draft.

Register (qubit indices):
  core H   : thermal core, scrambled before step 0
  matter M : K layers, each Bell-paired with its own reference qubits
  ref R_M  : purifies M; never touched by gates
Radiation is a relabelling: an emitted qubit leaves the active hole and joins R.

Entropies are Renyi-2, in bits.
"""
import numpy as np

# Array backend for the state vector: numpy (CPU) by default, or cupy (NVIDIA GPU) via set_backend.
# Random draws always use numpy on the CPU, so a given seed gives the same circuit on either backend.
xp = np


def set_backend(name):
    """Select 'numpy' or 'cupy' for state-vector arithmetic."""
    global xp
    if name == 'numpy':
        xp = np
    elif name == 'cupy':
        import cupy
        xp = cupy
    else:
        raise ValueError(name)
    return xp


def haar4(rng):
    z = (rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    return q * (d / np.abs(d))


class Hole:
    def __init__(self, n_core, layer_sizes, seed, pre_scramble=6, trace=False):
        self.rng = np.random.default_rng(seed)
        self.log = [] if trace else None      # (matrix, qubits) for every gate, for cross-validation
        self.nc = n_core
        self.layers = []
        q = n_core
        for m in layer_sizes:
            self.layers.append(list(range(q, q + m)))
            q += m
        self.nm = q - n_core
        self.ref = list(range(q, q + self.nm))
        self.ref_layers = []
        r = q
        for m in layer_sizes:
            self.ref_layers.append(list(range(r, r + m)))
            r += m
        self.N = r
        psi = xp.zeros(2 ** self.N, dtype=xp.complex128)
        psi[0] = 1.0
        self.psi = psi.reshape((2,) * self.N)
        # Bell pairs matter <-> reference
        for mq, rq in zip(range(n_core, n_core + self.nm), self.ref):
            self._h(rq)
            self._cnot(rq, mq)
        self.active = list(range(n_core))      # core only; layers join on release
        self.bound = [list(L) for L in self.layers]
        self.rad = []
        for _ in range(pre_scramble):
            self.scramble()

    # --- gates -------------------------------------------------------
    def _apply1(self, U, a):
        if self.log is not None:
            self.log.append((np.array(U, dtype=complex), (a,)))
        self.psi = xp.moveaxis(xp.tensordot(xp.asarray(U), self.psi, axes=([1], [a])), 0, a)

    def _h(self, a):
        self._apply1(np.array([[1, 1], [1, -1]]) / np.sqrt(2), a)

    def _apply2(self, U, a, b):
        if self.log is not None:
            self.log.append((np.array(U, dtype=complex).reshape(4, 4), (a, b)))
        U = xp.asarray(U).reshape(2, 2, 2, 2)
        t = xp.tensordot(U, self.psi, axes=([2, 3], [a, b]))
        del self.psi                                   # free the old state before the move (memory)
        self.psi = xp.moveaxis(t, [0, 1], [a, b])

    def _cnot(self, c, t):
        U = np.eye(4, dtype=complex)[[0, 1, 3, 2]]
        self._apply2(U, c, t)

    def scramble(self):
        """One layer of Haar 2-qubit gates on a random perfect matching of the active hole."""
        qs = list(self.rng.permutation(self.active))
        for i in range(0, len(qs) - 1, 2):
            self._apply2(haar4(self.rng), int(qs[i]), int(qs[i + 1]))

    # --- dynamics ------------------------------------------------------
    def release(self, layer_idx, n=None):
        L = self.bound[layer_idx]
        take = L if n is None else L[:n]
        self.active += take
        self.bound[layer_idx] = L[len(take):]

    def emit(self):
        j = int(self.rng.integers(len(self.active)))
        q = self.active.pop(j)
        self.rad.append(q)

    # --- entropies -----------------------------------------------------
    def S2(self, A):
        A = sorted(set(A))
        if not A or len(A) == self.N:
            return 0.0
        if len(A) > self.N / 2:
            A = [q for q in range(self.N) if q not in A]
        B = [q for q in range(self.N) if q not in A]
        M = xp.ascontiguousarray(xp.transpose(self.psi, A + B)).reshape(2 ** len(A), -1)
        rho = M @ M.conj().T
        del M
        pur = float(xp.real(xp.vdot(rho, rho)))
        del rho
        return float(-np.log2(pur))

    def I2(self, A, B):
        return self.S2(A) + self.S2(B) - self.S2(list(A) + list(B))


def page_step(n_core):
    """First step at which a core-only hole is past its halfway point (Hayden-Preskill 'old')."""
    return n_core // 2 + 1


def run_strict(n_core, layer_sizes, seed, trace=False):
    """H_S sanity arm: last in, first out, no scrambling. All matter is in from the start; the emission
    order is layer K, two core qubits, layer K-1, two core qubits, ..., then the rest of the core.
    D(k) must show K turns, and the per-layer traces must rise in reverse layer order."""
    h = Hole(n_core, layer_sizes, seed, pre_scramble=0, trace=trace)
    for i in range(len(layer_sizes)):
        h.release(i)
    core = list(range(n_core))
    order = []
    for L in reversed(h.layers):
        order += L + [core.pop(0) for _ in range(min(2, len(core)))]
    order += core
    D, per, checkpoints = [], [], []

    def record():
        if trace:
            checkpoints.append((len(h.log), list(h.rad)))
        D.append(h.I2(h.ref, h.rad))
        per.append([h.I2(rl, h.rad) for rl in h.ref_layers])

    record()
    for q in order:
        h.active.remove(q)
        h.rad.append(q)
        record()
    if trace:
        return np.array(D), np.array(per), dict(log=h.log, checkpoints=checkpoints, N=h.N, ref=list(h.ref))
    return np.array(D), np.array(per)


def run_schedule(n_core, layer_sizes, seed, release_steps=None, gradual_width=0, settle=0, mix_on_join=None,
                 trace=False, strict=False):
    """One trial. release_steps[i] = step at which layer i releases (sharp) or starts (gradual).
    settle = scrambling layers applied after step-0 releases, before the first emission
    (pilot amendment 1: the ledger baseline's matter is scrambled in before evaporation starts).
    mix_on_join = scrambling layers applied right after any release step, before the next emission
    (pilot-2 amendment: Hayden-Preskill's instantaneous-scrambler idealization; one t*). If None, it
    equals settle at step 0 and 0 afterwards, which reproduces pilot 1 exactly.
    Returns D(k) for k = 0..n_steps and per-layer traces. With trace=True, also returns a dict with the
    full gate log and, for each recorded step, the gate count so far and the radiation qubits."""
    if strict:
        return run_strict(n_core, layer_sizes, seed, trace=trace)
    h = Hole(n_core, layer_sizes, seed, trace=trace)
    checkpoints = []
    n_hole_total = n_core + sum(layer_sizes)
    D, per = [], []
    k = 0

    def record():
        if trace:
            checkpoints.append((len(h.log), list(h.rad)))
        D.append(h.I2(h.ref, h.rad))
        per.append([h.I2(rl, h.rad) for rl in h.ref_layers])

    while True:
        released = False
        for i, t in enumerate(release_steps):
            n_before = len(h.bound[i])
            if gradual_width and h.bound[i]:
                m = layer_sizes[i]
                due = [t + round(j * gradual_width / max(m - 1, 1)) for j in range(m)]
                n_due = sum(1 for d in due if d <= k) - (m - len(h.bound[i]))
                if n_due > 0:
                    h.release(i, n_due)
            elif not gradual_width and t <= k and h.bound[i]:
                h.release(i)
            released = released or len(h.bound[i]) < n_before
        if k == 0:
            for _ in range(settle):
                h.scramble()
            record()
        elif released and mix_on_join:
            for _ in range(mix_on_join):
                h.scramble()
        if not h.active and not any(h.bound):
            break
        if not h.active:
            k += 1
            continue
        h.scramble()
        h.emit()
        k += 1
        record()
        if len(h.rad) == n_hole_total or k > 10 * n_hole_total:
            break
    if trace:
        return np.array(D), np.array(per), dict(log=h.log, checkpoints=checkpoints, N=h.N, psi=(h.psi.get() if hasattr(h.psi, 'get') else h.psi.copy()),
                                               ref=list(h.ref), ref_layers=[list(r) for r in h.ref_layers])
    return np.array(D), np.array(per)


def tstar_probe(n_hole, seed, max_layers=30):
    """Protocol section 4 clock: steps for I2(probe_ref : rest of hole) to saturate.
    Also returns the Hayden-Preskill variant I2(probe_ref : hole minus 2 random qubits)."""
    h = Hole(n_hole - 1, [1], seed)        # core = n_hole-1, one probe qubit as a layer
    h.release(0)
    probe = h.layers[0][0]
    pref = h.ref
    rest = [q for q in h.active if q != probe]
    rng = np.random.default_rng(seed + 7)
    drop = list(rng.choice(h.active, 2, replace=False))
    hp = [q for q in h.active if q not in drop]
    a, b = [h.I2(pref, rest)], [h.I2(pref, hp)]
    for _ in range(max_layers):
        h.scramble()
        a.append(h.I2(pref, rest))
        b.append(h.I2(pref, hp))
    return np.array(a), np.array(b)
