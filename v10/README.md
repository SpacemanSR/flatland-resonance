# v10: headless engine for the filtered-unhooking trial

A state-vector simulation of an evaporating qubit hole whose matter is held in bound layers. Each layer joins the scrambling dynamics only at its release step. The observable is D(k) = I2(R_M : R), the Rényi-2 mutual information, in bits, between the matter's reference qubits and the radiation after k emissions.

## Files

| File | What it does |
| --- | --- |
| `v10.py` | Engine: register, Haar scrambling layers, release schedules (sharp or gradual), emission, Rényi-2 entropies, the t* probe |
| `pilot.py` | Pilot: baseline (H_L) only, N = 15 and 20, seeds 1000–1029. Measures t* and applies turn rule v1. Writes `pilot_results.json` |
| `calibrate.py` | Calibrates turn rule v2 (peak prominence in the rise rate) on baseline runs only. Writes `calibration.json` |
| `tstar_n30.py` | Scrambling clock for the N = 30 design's 21-qubit hole (probe only, about 22 qubits) |
| `pilot_results.json` | Pilot 1 output as recorded on 26 Nov 2026 |
| `pilot2.py` | Pilot 2 (27 Nov 2026): baselines only for the redesigned trial (old-hole test arm, young-hole control). Recalibrates the turn rule and measures t* at the new hole sizes. Writes `pilot2_results.json` |
| `pilot2_results.json` | Pilot 2 output as recorded on 27 Nov 2026 |

## Reproduce

```
pip install -r requirements.txt
python3 pilot.py        # about 3 minutes
python3 calibrate.py    # about 4 minutes
python3 tstar_n30.py    # about 3 minutes
python3 pilot2.py       # about 20 minutes
```

All randomness comes from numpy's `default_rng` with the listed seeds, so runs are deterministic on a given numpy version.

## Pilot results (26 Nov 2026)

- The scrambling time is t* = 4 steps at N = 15, 20 and 30 (the 21-qubit hole).
- At the end of evaporation the radiation holds exactly 2 bits per matter qubit: 6.00 of 6 at N = 15, and 12.00 of 12 at N = 20.
- Turn rule v1 (a flat stretch, then a 3σ jump) found no turns in 60 of 60 baseline runs, because the rise in this toy is smooth. Before filing it was replaced by rule v2, a peak-prominence rule. At a prominence of 0.2 × 2m bits, rule v2 finds exactly one turn in 60 of 60 baseline runs.
- No filtered (H_F) run has been made. Confirmatory runs begin only after the protocol is filed.

## Redesign after the literature check (v10.1, 27 Nov 2026)

- **Young holes.** Hayden & Preskill (arXiv:0708.4025) show that information thrown into a young hole stays hidden until the halfway point. Layered release into a young hole therefore gives one turn whatever the gaps. The test arm now releases layers into an *old* hole (`page_step`), and a young-hole release schedule is kept as a control.
- **Mixing on release.** Every release is followed by t* = 4 scrambling layers before the next emission (`mix_on_join`), Hayden-Preskill's instantaneous-scrambler idealization. Without it, a just-released layer leaks through its first gate partner.
- **Default behaviour.** With `mix_on_join=None`, the engine reproduces pilot 1 exactly.
- **Pilot 2** (baselines only) gives exactly one turn in 180 of 180 baseline runs at prominence f = 0.3 x 2m bits.

## Compute

Measured: 0.08 s per run at N = 15 and 6.5 s per run at N = 20. Cost grows about 32× per 5 qubits, so the extrapolated figures are about 3.5 min per run at N = 25 and about 1.8 h per run at N = 30 on one CPU thread. At N = 30 the run also needs about 52 GB of memory at peak.
