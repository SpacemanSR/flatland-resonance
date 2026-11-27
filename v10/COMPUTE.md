# Compute plan for the confirmatory runs

## What the runs need

| Register | Runs | Measured time per run (2-core Xeon, numpy) | Memory per run |
| --- | --- | --- | --- |
| N = 20 | 700 | 5.5 s | under 1 GB |
| N = 25 | 1,000 | 707 s | about 2 GB |
| N = 30 | 300 | not measured; about 25 h extrapolated | about 50 GB at peak |

Cost grows about 128× per 5 qubits: the state vector doubles with each qubit, and the entropy step at the half split dominates. At that rate the N = 25 arm takes about 200 hours, and the N = 30 arm several thousand hours, on hardware like the development workspace. **Both arms therefore run on one GPU machine.** The first job there is `run_confirmatory.py --timing` for N25 and N30, which replaces these extrapolations with measurements.

GPU requirement: one NVIDIA data-center GPU with 80 GB of memory (A100 80GB or H100 class), CUDA 12, plus a host with at least 64 GB of RAM. At 30 qubits the state takes 16 GB in double precision; a gate's working copy and the half-split density matrix bring the peak to about 50 GB.

## How to run

```
docker build -f Dockerfile.gpu -t v10-gpu .
docker run --gpus all -v $PWD/results:/v10/results v10-gpu \
    python3 run_confirmatory.py --timing --register N30_m1 --backend cupy
# after the OSF filing:
docker run --gpus all -v $PWD/results:/v10/results v10-gpu \
    python3 run_confirmatory.py --register N30_m1 --backend cupy --osf 10.17605/OSF.IO/XXXXX
python3 analyze.py results
```

Before any confirmatory GPU run, check the backend: `crossval.py` must pass on the GPU machine with `v10.set_backend('cupy')`. The CuPy path is written but could not be tested in the development workspace, which has no GPU.

## Where (in order of preference)

1. **A co-author's institutional cluster or an NSF ACCESS allocation.** Both are at no cost to the project. An ACCESS Explore request needs a one- to two-paragraph summary, the PI's CV, a title and keywords. The PI must be a U.S.-based researcher at the graduate-student level or higher, so the co-author files it. Draft summary below.
2. **A commercial cloud instance** with one 80 GB GPU, if no allocation is available in time. Record the provider, instance type and region in the paper. The manifest records the GPU model and driver automatically.

## Draft ACCESS Explore project summary (for the co-author PI)

**Title:** A preregistered state-vector test of layered information release from evaporating black holes

**Summary.** We test, in a unitary qubit model of black-hole evaporation, whether matter released into an old (post-Page) hole in separate layers leaves separate signatures in the radiation's mutual information, and whether the release gap needed to separate them grows with layer size, as Hayden–Preskill retention predicts. The study is preregistered on OSF, with fixed decision rules. The code is open source and archived (doi:10.5281/zenodo.24602812) and has been cross-validated against Qiskit Aer to 10⁻¹⁴ bits. The confirmatory design is 2,000 exact state-vector runs at 20, 25 and 30 qubits. Runs at 25 and 30 qubits need a GPU with 80 GB of memory (16 GB state, about 50 GB peak). We request an Explore allocation on an ACCESS GPU resource for about 1,300 runs. A timing run will fix the exact GPU-hours.

**Keywords:** black hole information, Page curve, Hayden–Preskill, quantum scrambling, state-vector simulation, preregistration

## Independent rerun

After the confirmatory runs, a second person on a second machine reruns a fixed subset from the archived release: N = 25, m = 1, conditions H_L old, H_F old Δ = 1 and Δ = 4, and H_F young Δ = 5, seeds 1–10. Seeds fix every gate, so the D(k) traces must match to numerical precision. The paper reports the maximum difference, and both machines' manifests are archived.
