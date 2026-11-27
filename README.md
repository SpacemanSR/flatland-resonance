# Flatland Resonance Model

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24600507.svg)](https://doi.org/10.5281/zenodo.24600507)

An interactive model of spacetime as a superfluid, in nine versions: from a resonance field rising out of a flat source plane, to a 13-qubit black hole that evaporates, forms an island, and has a swallowed message teleported back out of its radiation with the Yoshida–Kitaev decoder.

A Findings page (top-right toggle) states the thesis, summarises the nine versions, and reports four results computed from the model's own engine, with what is published, what is ours and what is computed kept apart.

A Sources page (same toggle) cites the papers behind every step of the build and lists, separately, the design choices that are ours rather than the literature's.

Everything runs on the device. No server, no data collected.

## Research engine (v10)

The [`v10/`](v10/) folder holds a headless Python engine for a preregistered test of filtered unhooking: whether matter that decouples layer by layer leaves more than one turn in the radiation's mutual information. It includes the pilot scripts and results. [`v10/README.md`](v10/README.md) explains how to reproduce them.

## Install on a phone
Open the hosted link, then:
- iPhone (Safari): Share → Add to Home Screen
- Android (Chrome): menu → Install app (or Add to Home screen)

It then launches full-screen from its icon and works offline.

## Run locally
Serve this folder with any static server, for example `python3 -m http.server`, and open http://localhost:8000.

## License and citation
MIT licensed (see [`LICENSE`](LICENSE)). To cite this work, use [`CITATION.cff`](CITATION.cff) or the archived release, v10.0.0: [doi:10.5281/zenodo.24600507](https://doi.org/10.5281/zenodo.24600507).
