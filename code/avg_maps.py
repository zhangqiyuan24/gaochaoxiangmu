"""Averaged power-map validation vs 误差真值表 (Table II values).
For each of the 15 conditions x 4 algorithms: average |map| over all clips,
then localize once. Compare with ground-truth table.
Usage: python3 avg_maps.py [quick|full]"""
import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/vsip/code"))
import beamforming_port as bp

BASE = os.path.expanduser("~/vsip/data/s315")
REF = os.path.expanduser("~/vsip/results/ref_315")

# ground truth (m): cond -> {algo: err}
GT = {
    ("trans", 5000, 1): dict(cbf=1.083, mvdr=5.060, music=2.522, music_csdm=4.117),
    ("trans", 5000, 2): dict(cbf=1.103, mvdr=1.064, music=0.920, music_csdm=0.996),
    ("trans", 5000, 3): dict(cbf=0.639, mvdr=0.194, music=1.698, music_csdm=0.958),
    ("trans", 7000, 1): dict(cbf=4.250, mvdr=2.275, music=2.925, music_csdm=2.318),
    ("trans", 7000, 2): dict(cbf=1.287, mvdr=1.479, music=1.500, music_csdm=1.017),
    ("trans", 7000, 3): dict(cbf=0.639, mvdr=0.418, music=2.426, music_csdm=2.318),
    ("trans", 9000, 1): dict(cbf=2.222, mvdr=2.445, music=2.398, music_csdm=1.584),
    ("trans", 9000, 2): dict(cbf=1.683, mvdr=1.722, music=1.711, music_csdm=1.772),
    ("trans", 9000, 3): dict(cbf=2.416, mvdr=1.945, music=2.364, music_csdm=0.536),
    ("bub_free", 0, 1): dict(cbf=0.420, mvdr=0.485, music=0.362, music_csdm=0.150),
    ("bub_free", 0, 2): dict(cbf=1.058, mvdr=1.250, music=2.005, music_csdm=0.848),
    ("bub_free", 0, 3): dict(cbf=3.000, mvdr=3.001, music=3.001, music_csdm=3.062),
    ("bub_needle", 0, 1): dict(cbf=0.306, mvdr=0.139, music=0.528, music_csdm=0.512),
    ("bub_needle", 0, 2): dict(cbf=0.694, mvdr=0.283, music=1.120, music_csdm=1.187),
    ("bub_needle", 0, 3): dict(cbf=2.763, mvdr=2.195, music=2.005, music_csdm=2.659),
}
ALGOS = ["cbf", "mvdr", "music", "music_csdm"]


def fre_table(cond):
    p = os.path.join(REF, f"bubble_wave-2024.3.15_2__{cond}__bubble_fre__Sheet1.csv")
    if not os.path.exists(p):
        return []
    with open(p) as f:
        r = [x for x in csv.reader(f) if x and x[0].strip()]
    out = []
    for x in r:
        try:
            out.append((int(float(x[0])), int(float(x[1]))))
        except (ValueError, IndexError):
            pass
    return out


def avg_map(files, freqs_list, dist, variant):
    """Average |maps| over files -> localized error, per algo."""
    bp.SEEPS = np.array([0.0, float(dist)])
    sums = {a: None for a in ALGOS}
    for p, fr in zip(files, freqs_list):
        if fr is None:
            continue
        bp.FREQS = np.asarray(fr, dtype=int)
        y, fs = bp.audioread(p)
        for a in ALGOS:
            mp = np.abs(bp.run_algo(y, fs, a, bp.grid_delays()))
            sums[a] = mp if sums[a] is None else sums[a] + mp
    errs = {}
    for a in ALGOS:
        if sums[a] is None:
            continue
        est, err, amp = bp.localize(sums[a], flip=(a == "music" and variant.get("flip", True)))
        errs[a] = (err, est)
    return errs, sums


def cond_variant(kind, dist, fc, variant):
    """freqs per file under a variant setting."""
    return variant


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "quick"
    quick_only = mode == "quick"
    # variants: (name, trans_band, bubble_band_mode, music_flip)
    VAR = []
    for tb_name, tb in (("wide", lambda fc: np.arange(1000, 12001, 500)),
                        ("pm100", lambda fc: np.arange(fc - 100, fc + 101, 20)),
                        ("pm500", lambda fc: np.arange(fc - 500, fc + 501, 100))):
        VAR.append((f"trans={tb_name},bub=per,f=on", tb, "per", True))
        VAR.append((f"trans={tb_name},bub=uni,f=on", tb, "uni", True))
    VAR.append(("trans=wide,bub=per,f=off", lambda fc: np.arange(1000, 12001, 500), "per", False))

    conds = list(GT.keys())
    if quick_only:
        conds = [("trans", 5000, 1), ("trans", 9000, 3), ("bub_free", 0, 1), ("bub_needle", 0, 1), ("bub_needle", 0, 3)]
    for vname, tb, bub_mode, flip in VAR:
        variant = {"flip": flip}
        hits, total, delta = 0, 0, 0.0
        for key in conds:
            kind, fc, dist = key
            if kind == "trans":
                d = os.path.join(BASE, "bubble wave-2024.3.15 1", f"{fc}Hz {dist}")
                files = sorted(glob.glob(os.path.join(d, "*.wav")))
                freqs = [tb(fc)] * len(files)
            else:
                cond = ("针头" if kind == "bub_needle" else "无针头") + f"_{dist}"
                d = os.path.join(BASE, "bubble wave-2024.3.15 2", cond.replace("_", " "))
                files = sorted(glob.glob(os.path.join(d, "*.wav")))
                tbl = fre_table(cond)
                if bub_mode == "per":
                    freqs = []
                    for i in range(len(files)):
                        if i < len(tbl):
                            lo, hi = tbl[i]
                            fr = np.arange(lo, hi + 1, 100)
                            freqs.append(fr if len(fr) >= 2 else np.arange(lo - 100, hi + 101, 100))
                        else:
                            freqs.append(None)
                else:
                    if kind == "bub_needle":
                        freqs = [np.arange(3000, 5001, 100)] * len(files)
                    else:
                        freqs = [np.arange(700, 1501, 100)] * len(files)
            if not files:
                continue
            errs, _ = avg_map(files, freqs, dist, variant)
            for a in ALGOS:
                if a in errs:
                    e, est = errs[a]
                    g = GT[key][a]
                    total += 1
                    delta += abs(e - g)
                    if abs(e - g) < 0.03:
                        hits += 1
        print(f"{vname:34s} hits={hits:2d}/{total:2d} meanabs={delta/max(total,1):.3f}", flush=True)


if __name__ == "__main__":
    main()
