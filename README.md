# Underwater Gas-Leakage Passive Acoustic Localization (tank campaign)

Numpy port and analysis pipeline for a laboratory-scale passive-acoustic
localization study of underwater gas leakage with a compact 8-hydrophone
array (0.5 m spacing, planar ring), comparing four 2-D near-field
beamforming localizers:

- **CBF** — conventional (delay-and-sum) beamforming
- **MVDR** — adaptive beamforming (diagonal-loaded Capon)
- **MUSIC** — subspace method on the raw outer-product matrix
- **CSDM-MUSIC** — MUSIC on the cross-spectral density matrix (CSDM)

The localization is a direct 2-D position-grid search with a spherical-wave
(near-field) steering model over a 1/36 m grid, x ∈ [−3, 3] m,
y ∈ [0, 6] m, using a single-snapshot spectrum at the 23 analysis bins
1000:500:12000 Hz.

## Contents

```
code/
  beamforming_port.py  faithful numpy port of the four MATLAB localizers,
                       including the original quirks (non-conjugated vs
                       conjugated matrix forms, delta=1e-3 loading after a
                       1/8 scale, row-flipped MUSIC map, row-major argmax)
  val_hist8.py         per-condition geometry calibration (array re-mount
                       scale s ~ 1.20-1.26) against the archived per-event
                       error sheets
  run_315_tone.py      E1: full projector-campaign rerun (21 tone
                       conditions x 4 algorithms x <=20 events) with
                       per-event errors and event-averaged power-map metric
  run_e2_snr.py        E2: SNR-robustness sweep (band-matched noise
                       injected post hoc at the 23 analysis bins of the
                       zero-padded single-snapshot spectrum; SNR +20..-20 dB)
  fig4_snr.mjs         Fig. 4 generator: error-vs-SNR curves (pure
                       pgfplots LaTeX, no extra packages)
results/              per-event CSVs, summary JSONs and the calibration
                      log of the runs reported in the paper
```

The raw 8-channel wav recordings (24-bit PCM, 64 kHz, 10 ms bursts) are
proprietary tank-experiment data and are **not** included; the pipeline
reads them from a per-condition directory tree
`<freq>Hz <dist>/*.wav` (MATLAB `dir()` ordering is reproduced for
event-index compatibility). Note: in the source campaign three projector
conditions (5 kHz/2 m, 7 kHz/2 m, 7 kHz/3 m) retained only a single valid
recording each and 13 kHz/3 m is empty; the runners handle this.

## Reproducing

```bash
python3 code/val_hist8.py     # ~1.5 h: per-condition scale calibration
python3 code/run_315_tone.py  # E1 full rerun (all tone conditions)
E2_CONDS=5000Hz_3,3000Hz_2,11000Hz_1 python3 code/run_e2_snr.py  # E2 sweep
node code/fig4_snr.mjs results/e2_snr.csv outdir 5000Hz_3,3000Hz_2,11000Hz_1 4.5  # Fig. 4
```

Requires numpy/scipy (any recent version); PYTHONPATH must point at the
package directory if you keep them separate.

## Provenance notes

- The archived MATLAB reference results were reproduced **exactly
  per-event** (tolerance 2e-3 m) for CBF/MVDR in the calibrated conditions
  (e.g. 5 kHz/3 m: MVDR and CBF 20/20 events at s = 1.240; 7 kHz/1 m and
  9 kHz/1 m CBF 20/20 at s = 1.2375).
- The MUSIC-family localizers are reproduced only at the statistical
  level: the noise-subspace projector of a rank-1 single-snapshot matrix
  is basis-dependent inside its degenerate eigenvalue cluster, so MATLAB's
  `eig` and LAPACK's can return different (equally valid) eigenvector
  bases. Means/medians agree; individual events may differ.
- E2 injects noise directly at the 23 analysis bins of the zero-padded
  spectrum (conjugate-symmetric, i.e. equivalent to a real band-line
  sequence). The SNR is set on the array-averaged in-band bin power.

## License

MIT (code). The result CSVs/JSONs are provided for review reproducibility.
