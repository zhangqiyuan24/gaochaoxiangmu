"""Fig.2 draft: spatial spectra, 4 algorithms x 2 signal types.
Rows: transducer tone / needle-free bubble. Unified colorbar per row.
True position = white x, estimate = white o."""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.expanduser("~/vsip/results/e1_315")
ALGOS = [("cbf", "CBF"), ("mvdr", "MVDR"), ("music", "MUSIC"), ("music_csdm", "CSDM-MUSIC")]

fig, axes = plt.subplots(2, 4, figsize=(7.16, 3.6), constrained_layout=True)
rows_done = 0
for r, pattern in enumerate(["map_*__trans_5000Hz_1m_01.npz", "map_*__bubble_needle_free_1m_03.npz"]):
    maps = {}
    for a, _ in ALGOS:
        p = os.path.join(OUT, pattern.replace("map_*", f"map_{a}"))
        if os.path.exists(p):
            maps[a] = np.load(p)
    if not maps:
        continue
    db = {a: 10 * np.log10(m["mp_abs"] / m["mp_abs"].max() + 1e-12) for a, m in maps.items()}
    vmin = min(v.min() for v in db.values())
    vmax = 0.0
    for c, (a, label) in enumerate(ALGOS):
        ax = axes[r, c]
        m = maps[a]
        gx, gy = m["grid_x"], m["grid_y"]
        pc = ax.pcolormesh(gy, gx, db[a], cmap="viridis", vmin=vmin, vmax=vmax,
                           shading="auto", rasterized=True)
        ax.plot(m["seeps"][1], m["seeps"][0], marker="x", color="white", ms=6, mew=1.5, ls="none")
        ax.plot(m["est"][1], m["est"][0], marker="o", color="white", ms=6, mfc="none", mew=1.2, ls="none")
        ax.set_title(f"({'abcd'[c]}) {label}", fontsize=8)
        ax.set_xlim(0, 6); ax.set_ylim(-2, 2)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
        if c == 0:
            ax.set_ylabel(["Transducer 5 kHz, 1 m", "Bubble (needle-free), 1 m"][r] + "\n$x$ (m)", fontsize=8)
        if r == 1:
            ax.set_xlabel("y (m)", fontsize=8)
    fig.colorbar(pc, ax=axes[r, :].tolist(), shrink=0.9, pad=0.01,
                 label="Normalized spectrum (dB)")
    rows_done += 1

if rows_done:
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig2_spatial_spectra.{ext}"), dpi=300)
    print("saved fig2,", rows_done, "rows")
else:
    print("no maps yet")
