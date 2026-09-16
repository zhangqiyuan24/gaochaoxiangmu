"""E1 full run over 2024.3.15 sessions -> writer-spec e1_errors.csv.
Transducer: wide band 1-12 kHz (band A, matches ported legacy default).
Bubbles:    per-sample band from bubble_fre.xls (col1=f_low, col2=f_high), 100 Hz steps.
Also saves one representative spatial-spectrum map per (signal_type, distance) for Fig.2."""
import csv
import glob
import os
import statistics as st
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/vsip/code"))
import beamforming_port as bp

BASE = os.path.expanduser("~/vsip/data/s315")
REF = os.path.expanduser("~/vsip/results/ref_315")
OUT = os.path.expanduser("~/vsip/results/e1_315_v2")
os.makedirs(OUT, exist_ok=True)
ALGOS = ["cbf", "mvdr", "music", "music_csdm"]


def find_wavs(d):
    """All unique event wavs under d: the folder itself plus one level of
    subfolders (数据/ 新建文件夹/ session 1-3/).  Files whose name does not
    contain 'bubble' (e.g. Amplitude_observation.wav) are skipped.  A
    subfolder wav whose basename already appeared (the migration left exact
    duplicate copies directly in the condition folder) is dropped; repeated
    basenames across session subfolders keep the first (session 1) file.
    Returns paths sorted lexicographically by basename, which replicates the
    original MATLAB dir() listing order used to build the bubble_fre tables.
    """
    roots = [d] + [os.path.join(d, s) for s in sorted(os.listdir(d))
                   if os.path.isdir(os.path.join(d, s))]
    seen, out = set(), []
    for root in roots:
        for p in sorted(glob.glob(os.path.join(root, "*.wav"))):
            b = os.path.basename(p)
            if "bubble" not in b.lower():
                continue
            if b in seen:
                continue
            seen.add(b)
            out.append(p)
    return sorted(out, key=os.path.basename)


def event_num(p):
    """Event number parsed from the wav basename (bubbleN.wav / bubble NN fcHz .wav)."""
    import re
    m = re.search(r"(\d+)", os.path.basename(p))
    return int(m.group(1)) if m else 0


def run_one(path, algo, seeps):
    bp.SEEPS = np.asarray(seeps, dtype=float)
    y, fs = bp.audioread(path)
    mp = bp.run_algo(y, fs, algo, bp.grid_delays())
    est, err, amp = bp.localize(mp, flip=(algo == "music"))
    return est, err, amp


def median_filtered(vals):
    v = list(vals)
    med, sd = st.median(v), st.pstdev(v)
    keep = [x for x in v if x - med <= sd]
    return st.median(keep) if keep else med


rows, maps_done = [], set()


def process_set(files, signal_type, dist, freqs_of, tag, rep_pick=0):
    """freqs_of(i) -> np.array of freq bins for file index i."""
    for i, p in enumerate(files):
        fr = freqs_of(i)
        if fr is None or len(fr) == 0:
            continue
        bp.FREQS = np.asarray(fr, dtype=int)
        sid = f"{tag}_ev{event_num(p):02d}"
        for a in ALGOS:
            est, err, amp = run_one(p, a, [0.0, float(dist)])
            rows.append([a, signal_type, dist, sid, round(err, 6),
                         round(est[0], 6), round(est[1], 6),
                         f"{fr[0]}-{fr[-1]}"])
            key = (signal_type, dist)
            if i == rep_pick and key not in maps_done:
                np.savez_compressed(
                    os.path.join(OUT, f"map_{a}__{sid}.npz"),
                    mp_abs=amp, est=est, error=err,
                    grid_x=bp.GRID_X, grid_y=bp.GRID_Y,
                    seeps=[0.0, float(dist)])
        if key not in maps_done and i == rep_pick:
            maps_done.add(key)
            print(f"  map saved for {key} ({sid})", flush=True)


# ---- transducer ----
for fc in (1000, 3000, 5000, 7000, 9000, 11000, 13000):
    band = np.arange(1000, 12000 + 1, 500) if fc <= 11000 else \
        np.arange(fc - 2000, fc + 2001, 500)
    for dist in (1, 2, 3):
        d = os.path.join(BASE, "bubble wave-2024.3.15 1", f"{fc}Hz {dist}")
        files = find_wavs(d)
        if files:
            process_set(files, "transducer", dist, lambda i, b=band: b,
                        f"trans_{fc}Hz_{dist}m")
            print(f"transducer {fc}Hz {dist}m: {len(files)} done", flush=True)

# ---- bubbles: per-sample band from bubble_fre ----
def fre_table(cond):
    p = os.path.join(REF, f"bubble_wave-2024.3.15_2__{cond}__bubble_fre__Sheet1.csv")
    if not os.path.exists(p):
        return []
    with open(p) as f:
        r = [x for x in csv.reader(f) if x and x[0].strip()]
    out = []
    for x in r:
        try:
            flo, fhi = float(x[0]), float(x[1])
        except (ValueError, IndexError):
            continue
        if flo <= 0 or fhi <= 0:
            continue  # zero-padding rows, not real measurements
        out.append((int(flo), int(fhi)))
    return out


for cond, stype in (("针头_1", "bubble_needle"), ("针头_2", "bubble_needle"),
                    ("针头_3", "bubble_needle"), ("无针头_1", "bubble_needle_free"),
                    ("无针头_2", "bubble_needle_free"), ("无针头_3", "bubble_needle_free")):
    dist = int(cond[-1])
    d = os.path.join(BASE, "bubble wave-2024.3.15 2", cond.replace("_", " "))
    files = find_wavs(d)
    tbl = fre_table(cond)
    if not files:
        continue

    def freqs_of(i, t=tbl, n=len(files)):
        if i < len(t):
            lo, hi = t[i]
            fr = np.arange(lo, hi + 1, 100)
            return fr if len(fr) >= 2 else np.arange(lo - 100, hi + 101, 100)
        print(f"  !! {cond}: event {i+1} ({os.path.basename(files[i])}) has no "
              f"bubble_fre row ({len(t)} valid rows) - skipped", flush=True)
        return None

    process_set(files, stype, dist, freqs_of, f"{stype}_{dist}m",
                rep_pick=min(2, len(files) - 1))
    print(f"{stype} {dist}m: {len(files)} wavs, fre rows {len(tbl)}", flush=True)

with open(os.path.join(OUT, "e1_errors.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["algorithm", "signal_type", "distance_m", "sample_id",
                "error_m", "est_x_m", "est_y_m", "freq_band_Hz"])
    w.writerows(rows)

print(f"\n{len(rows)} rows -> e1_errors.csv")
print("\n== medians (filtered) by condition ==")
conds = sorted({(r[1], r[2]) for r in rows})
for stype, dist in conds:
    line = f"{stype:18s} {dist}m: "
    for a in ALGOS:
        v = [r[4] for r in rows if r[0] == a and r[1] == stype and r[2] == dist]
        line += f"{a}={median_filtered(v):.3f}(n={len(v)}) " if v else f"{a}=- "
    print(line, flush=True)
