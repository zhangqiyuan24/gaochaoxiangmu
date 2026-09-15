#!/usr/bin/env python3
"""
Faithful numpy port of the four MATLAB beamforming localization scripts
(CBF / MVDR / MUSIC / MUSIC-CSDM) from the undergrad thesis project.

Reproduction quirks kept intentionally:
- csdm built WITHOUT conj for CBF & plain MUSIC (x' * x), WITH conj and a 1/8
  factor for MVDR & MUSIC-CSDM (x' * conj(x))  [1/Nch with Nch=8]
- frequency bins used directly as FFT indices (fft length == Fs == 64000)
- diagonal loading delta = 1e-3 added AFTER the 1/8 scaling
- plain MUSIC flips the map vertically (row reversal) before argmax
- error = euclidean distance to Seeps = (0, 4)
Grid: x in linspace(-2, 2, 145) (rows), y in linspace(0, 6, 217) (cols).
"""
import argparse
import csv
import glob
import os
import wave

import numpy as np

C0 = 1489.0
HPOS = np.array([
    [0, 0, 0.5],
    [0, 0, 1],
    [0.5, 0, 1.5],
    [1, 0, 1.5],
    [1.5, 0, 1],
    [1.5, 0, 0.5],
    [1, 0, 0],
    [0.5, 0, 0],
], dtype=float)
SEEPS = np.array([0.0, 4.0])
FREQS = np.arange(1000, 12000 + 1, 500)   # MATLAB 1000:500:12000 -> 23 bins
GRID_X = np.linspace(-2, 2, 145)          # R     (rows)
GRID_Y = np.linspace(0, 6, 217)           # Theta (cols)
DELTA = 1e-3
NSRC = 1
NCH = 8


def audioread(path):
    """MATLAB audioread equivalent for 16/24/32-bit PCM (normalized)."""
    with wave.open(path, "rb") as w:
        nch, fr, nf, sw = (w.getnchannels(), w.getframerate(),
                           w.getnframes(), w.getsampwidth())
        raw = w.readframes(nf)
    if sw == 2:
        y = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif sw == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        v = (b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)).astype(np.int64)
        v = np.where(v >= 1 << 23, v - (1 << 24), v)
        y = v.astype(np.float64) / float(1 << 23)
    else:
        y = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    return y.reshape(-1, nch), fr


# ---------------------------------------------------------------------------
# 2024-03-15 tank-campaign geometry (locked by per-event reproduction of the
# archived error sheets, ref_315; see val_hist5-8).  The array was re-mounted
# for this session: hydrophone positions are the 1031 layout, x-centered and
# uniformly rescaled by a per-condition factor s (~1.23-1.24, array re-plate
# between sessions); grid extends to +-3 m in x (archived map calibration).
# ---------------------------------------------------------------------------
GRID_X_315 = np.linspace(-3, 3, 217)          # rows, step 1/36 m
GRID_Y_315 = np.linspace(0, 6, 217)           # cols, step 1/36 m
_HC = HPOS - np.array([0.75, 0.0, 0.0])       # x-centered


def hpos_315(s=1.24):
    return _HC * s


def grid_delays_315(s=1.24):
    """(npts, 8) delays/c0 on the 315 grid for scale s."""
    xx, yy = np.meshgrid(GRID_X_315, GRID_Y_315, indexing="ij")
    pts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    d = np.linalg.norm(pts[:, None, :] - hpos_315(s)[None, :, :], axis=2)
    return d / C0


def matlab_wavs(dirpath, n=None):
    """Files in MATLAB dir() order: sorted by full basename (extension
    included), so 'bubble1.wav' < 'bubble10.wav' < 'bubble2.wav'."""
    import os as _os
    seen = sorted(_os.path.basename(p) for p in glob.glob(_os.path.join(dirpath, "*.wav")))
    return seen[:n] if n else seen


def spec_315(y, fs):
    """Single-snapshot spectra at the 23 analysis bins (fft length = Fs)."""
    return np.fft.fft(y, n=int(fs), axis=0)[FREQS - 1, :]


def _projector_matlab(q):
    """En @ En' for the 7 smallest eigenvalues, MATLAB-style: eigh for
    Hermitian matrices, otherwise eig with complex sort (primary |lambda|,
    tie-break angle) and RAW (non-orthonormal) eigenvectors."""
    if np.allclose(q, q.conj().T, atol=1e-10):
        w, v = np.linalg.eigh(q)
        en = v[:, :NCH - NSRC]
    else:
        w, v = np.linalg.eig(q)
        idx = np.lexsort((np.angle(w), np.abs(w)))
        en = v[:, idx[:NCH - NSRC]]
    return en @ en.conj().T


def maps_315(spec, s=1.24, tau=None):
    """All four algorithm maps for one single-snapshot spectrum.
    Returns {algo: abs-map (217,217)} with the MUSIC row-flip NOT applied
    (apply via argmax_315)."""
    if tau is None:
        tau = grid_delays_315(s)
    qc = np.einsum("fi,fj->fij", spec, spec.conj()) * (1.0 / NCH)   # mvdr/csdm
    qn = np.einsum("fi,fj->fij", spec, spec)                        # cbf/music
    out = {a: np.zeros(tau.shape[0], dtype=complex) for a in
           ("cbf", "mvdr", "music", "music_csdm")}
    for k in range(len(FREQS)):
        v = np.exp(-1j * 2 * np.pi * FREQS[k] * tau)
        qn_k = qn[k] + DELTA * np.eye(NCH)
        qc_k = qc[k] + DELTA * np.eye(NCH)
        out["cbf"] += np.einsum("pi,ij,pj->p", v.conj(), qn_k.conj().T, v)
        out["music"] += 1.0 / np.einsum("pi,ij,pj->p", v.conj(), _projector_matlab(qn_k), v)
        out["mvdr"] += 1.0 / np.einsum("pi,ij,pj->p", v.conj(), np.linalg.inv(qc_k), v)
        out["music_csdm"] += 1.0 / np.einsum("pi,ij,pj->p", v.conj(), _projector_matlab(qc_k), v)
    return {a: np.abs(m).reshape(len(GRID_X_315), len(GRID_Y_315))
            for a, m in out.items()}


def argmax_315(amap, algo):
    """Est (x, y) with MATLAB quirks: row-major first-max, MUSIC row-flip."""
    a = amap[::-1, :] if algo == "music" else amap
    ix, iy = np.unravel_index(np.argmax(a), a.shape)
    return GRID_X_315[ix], GRID_Y_315[iy]


def error_315(est, dist):
    """Error vs the 315-era truth (0, dist)."""
    return float(np.hypot(est[0], est[1] - dist))


def build_csdm(y, fs, conj, scale):
    """FFT of length Fs; CSDM per freq bin as 8x8 stacked array."""
    nfft = int(fs)
    spec = np.fft.fft(y, n=nfft, axis=0)          # (nfft, 8)
    bins = FREQS - 1                               # MATLAB 1-based index f -> 0-based f-1
    vecs = spec[bins, :]                           # (Lf, 8) complex
    if conj:
        q = np.einsum("fi,fj->fij", vecs, vecs.conj())
    else:
        q = np.einsum("fi,fj->fij", vecs, vecs)
    return q * scale


def grid_delays():
    """path_lengths/c0 for every grid point: (npts, 8)."""
    xx, yy = np.meshgrid(GRID_X, GRID_Y, indexing="ij")
    pts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    d = np.linalg.norm(pts[:, None, :] - HPOS[None, :, :], axis=2)
    return d / C0


def music_en_projector(q):
    """Noise-subspace projector En @ En' with an ORTHONORMAL basis
    (MATLAB eig on Hermitian matrices returns orthonormal vectors; raw
    np.linalg.eig does not inside a degenerate cluster, which makes the
    projector basis-dependent -- so we orthonormalize explicitly)."""
    n_en = NCH - NSRC
    if np.allclose(q, q.conj().T, atol=1e-10):
        w, v = np.linalg.eigh(q)          # orthonormal, ascending eigenvalues
        en = v[:, :n_en]                  # 7 smallest -> noise subspace
    else:
        w, v = np.linalg.eig(q)
        idx = np.argsort(w.real)
        en = v[:, idx[:n_en]]
        en, _ = np.linalg.qr(en)          # orthonormalize the cluster basis
    return en @ en.conj().T


def run_algo(y, fs, algo, tau):
    """Spatial spectrum map (l_R, l_T) for one file."""
    npts = tau.shape[0]
    out = np.zeros(npts, dtype=complex)
    if algo in ("mvdr", "music_csdm"):
        csdm = build_csdm(y, fs, conj=True, scale=1.0 / NCH)
    else:
        csdm = build_csdm(y, fs, conj=False, scale=1.0)

    for k in range(len(FREQS)):
        f = FREQS[k]
        v = np.exp(-1j * 2 * np.pi * f * tau)              # (npts, 8)
        q = csdm[k] + DELTA * np.eye(NCH)
        if algo == "cbf":
            m = np.einsum("pi,ij,pj->p", v.conj(), q.conj().T, v)
            out += m
        elif algo == "mvdr":
            qinv = np.linalg.inv(q)
            m = np.einsum("pi,ij,pj->p", v.conj(), qinv, v)
            out += 1.0 / m
        else:  # music / music_csdm
            p = music_en_projector(q)
            m = np.einsum("pi,ij,pj->p", v.conj(), p, v)
            out += 1.0 / m
    return out.reshape(len(GRID_X), len(GRID_Y))


def localize(mp, flip):
    """argmax + coordinate read, replicating the MATLAB plotting code."""
    a = np.abs(mp)
    if flip:
        a = a[::-1, :]
    ix, iy = np.unravel_index(np.argmax(a), a.shape)      # row-major first, like find()
    est = np.array([GRID_X[ix], GRID_Y[iy]])
    return est, float(np.linalg.norm(SEEPS - est)), a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.expanduser(
        "~/vsip/data/bubble_wave_1031/bubble wave-2023.10.31"))
    ap.add_argument("--out-dir", default=os.path.expanduser("~/vsip/results/e1_tone"))
    ap.add_argument("--algos", default="cbf,mvdr,music,music_csdm")
    ap.add_argument("--save-maps", default="dahuan_fang_8000hz_1.wav,"
                        "xiaohuan_sin_5000hz_1.wav,xiaohuan_sin_8000hz_1.wav")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tau = grid_delays()
    files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(args.data_dir, "*.wav")))
    save_maps = set(args.save_maps.split(","))

    rows = []
    for fn in files:
        y, fs = audioread(os.path.join(args.data_dir, fn))
        for algo in args.algos.split(","):
            mp = run_algo(y, fs, algo, tau)
            est, err, _ = localize(mp, flip=(algo == "music"))
            rows.append([algo, fn, est[0], est[1], err])
            if fn in save_maps:
                np.savez_compressed(
                    os.path.join(args.out_dir, f"map_{algo}__{os.path.splitext(fn)[0]}.npz"),
                    mp_abs=np.abs(mp), est=est, error=err,
                    grid_x=GRID_X, grid_y=GRID_Y, seeps=SEEPS)
        print(f"{fn}: done", flush=True)

    out_csv = os.path.join(args.out_dir, "e1_tone_errors.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", "file", "est_x_m", "est_y_m", "error_m"])
        w.writerows(rows)

    # summary
    import statistics as st
    print("\n== summary (mean error m) ==")
    for algo in args.algos.split(","):
        v = [r[4] for r in rows if r[0] == algo]
        print(f"{algo:12s} n={len(v)} mean={st.mean(v):.4f} std={st.stdev(v):.4f} "
              f"min={min(v):.4f} max={max(v):.4f}")
    print("reference    CBF mean=3.2480 | MVDR mean=1.7776 | "
          "MUSIC mean=2.3913 | MUSIC_CSDM mean=2.1743")


if __name__ == "__main__":
    main()
