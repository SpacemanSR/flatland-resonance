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


def haar4(rng):
    z = (rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    return q * (d / np.abs(d))


class Hole:
    def __init__(self, n_core, layer_sizes, seed, pre_scramble=6):
        self.rng = np.random.default_rng(seed)
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
        psi = np.zeros(2 ** self.N, dtype=np.complex128)
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
        self.psi = np.moveaxis(np.tensordot(U, self.psi, axes=([1], [a])), 0, a)

    def _h(self, a):
        self._apply1(np.array([[1, 1], [1, -1]]) / np.sqrt(2), a)

    def _apply2(self, U, a, b):
        U = U.reshape(2, 2, 2, 2)
        t = np.tensordot(U, self.psi, axes=([2, 3], [a, b]))
        self.psi = np.moveaxis(t, [0, 1], [a, b])

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
        M = np.transpose(self.psi, A + B).reshape(2 ** len(A), -1)
        rho = M @ M.conj().T
        return float(-np.log2(np.real(np.vdot(rho, rho))))

    def I2(self, A, B):
        return self.S2(A) + self.S2(B) - self.S2(list(A) + list(B))


def page_step(n_core):
    """First step at which a core-only hole is past its halfway point (Hayden-Preskill 'old')."""
    return n_core // 2 + 1


def run_schedule(n_core, layer_sizes, seed, release_steps, gradual_width=0, settle=0, mix_on_join=None):
    """One trial. release_steps[i] = step at which layer i releases (sharp) or starts (gradual).
    settle = scrambling layers applied after step-0 releases, before the first emission
    (pilot amendment 1: the ledger baseline's matter is scrambled in before evaporation starts).
    mix_on_join = scrambling layers applied right after any release step, before the next emission
    (pilot-2 amendment: Hayden-Preskill's instantaneous-scrambler idealization; one t*). If None, it
    equals settle at step 0 and 0 afterwards, which reproduces pilot 1 exactly.
    Returns D(k) for k = 0..n_steps and per-layer traces."""
    h = Hole(n_core, layer_sizes, seed)
    n_hole_total = n_core + sum(layer_sizes)
    D, per = [], []
    k = 0

    def record():
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
