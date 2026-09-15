"""E1 v3: full rerun of the 2024-03-15 projector (tone) campaign with the
locked 315-era construction (see beamforming_port.py header).

Per condition (21: 1/3/5/7/9/11/13 kHz x 1/2/3 m):
  - scale s_c from ~/vsip/results/val_hist8_report.json (fallback 1.24)
  - up to 20 surviving wavs in MATLAB dir order
  - 4 algorithms x per-event est/err  -> e1_errors.csv (writer columns)
  - per-event match vs archived ref_315 sheets (tol 2e-3)
  - event-averaged power-map metric (raw mean and per-event [0,1]-normalized
    mean) vs the truth-table single values for 5/7/9 kHz
Outputs: ~/vsip/results/e1_v3/{e1_errors.csv,e1_v3_report.json}
"""
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/vsip/code"))
import beamforming_port as bp

BASE = os.path.expanduser("~/vsip/data/s315/bubble wave-2024.3.15 1")
REF = os.path.expanduser("~/vsip/results/ref_315")
OUT = os.path.expanduser("~/vsip/results/e1_v3")
os.makedirs(OUT, exist_ok=True)

# truth-table section-1 single values (avg power-map metric), 5/7/9 kHz only
TT = {  # (freq, dist): {ALGO: err}
    (5000, 1): {"CBF": 1.083, "MVDR": 5.060, "MUSIC": 2.522, "MUSIC_CSDM": 4.117},
    (7000, 1): {"CBF": 4.250, "MVDR": 2.275, "MUSIC": 2.925, "MUSIC_CSDM": 2.318},
    (9000, 1): {"CBF": 2.222, "MVDR": 2.418, "MUSIC": 3.833, "MUSIC_CSDM": 2.588},
    (5000, 2): {"CBF": 1.103, "MVDR": 1.064, "MUSIC": 0.920, "MUSIC_CSDM": 0.996},
    (7000, 2): {"CBF": 1.287, "MVDR": 1.479, "MUSIC": 1.500, "MUSIC_CSDM": 1.017},
    (9000, 2): {"CBF": 1.683, "MVDR": 1.722, "MUSIC": 1.831, "MUSIC_CSDM": 1.038},
    (5000, 3): {"CBF": 0.639, "MVDR": 0.194, "MUSIC": 1.698, "MUSIC_CSDM": 0.958},
    (7000, 3): {"CBF": 0.639, "MVDR": 0.418, "MUSIC": 2.426, "MUSIC_CSDM": 2.318},
    (9000, 3): {"CBF": 2.416, "MVDR": 1.945, "MUSIC": 2.341, "MUSIC_CSDM": 0.536},
}
ALGOS = [("CBF", "cbf"), ("MVDR", "mvdr"), ("MUSIC", "music"), ("MUSIC_CSDM", "music_csdm")]

scales = {}
rep_path = os.path.expanduser("~/vsip/results/val_hist8_report.json")
if os.path.exists(rep_path):
    with open(rep_path) as f:
        scales = {k: v["s"] for k, v in json.load(f).items()}


def arch(cname, A):
    p = os.path.join(REF, f"bubble_wave-2024.3.15_1__{cname}__{A}_error__Sheet1.csv")
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return [float(x[0]) for x in csv.reader(f) if x and x[0].strip()]


rows = [["algorithm", "signal_type", "distance_m", "sample_id",
         "error_m", "est_x_m", "est_y_m", "freq_band_Hz"]]
report = {}
only = [c for c in os.environ.get("E1_CONDS", "").split(",") if c]
conds = []
for d in sorted(glob.glob(os.path.join(BASE, "*Hz *"))):
    b = os.path.basename(d)
    freq, dist = b.split("Hz ")
    cname = f"{freq}Hz_{dist}"
    if only and cname not in only:
        continue
    conds.append((cname, d, int(freq), int(dist)))

for cname, d, freq, dist in conds:
    s = scales.get(cname, 1.24)
    tau = bp.grid_delays_315(s)
    wavs = bp.matlab_wavs(d, 20)
    arch_all = {A: arch(cname, A) for A, _ in ALGOS}
    cond_rows = []
    maps_sum = {a: None for _, a in ALGOS}
    maps_norm_sum = {a: None for _, a in ALGOS}
    per_event = {A: [] for A, _ in ALGOS}
    n_cmp = min(len(wavs), 20)
    for i in range(n_cmp):
        y, fs = bp.audioread(os.path.join(d, wavs[i]))
        mp_all = bp.maps_315(bp.spec_315(y, fs), s, tau)
        for A, a in ALGOS:
            mp = mp_all[a]
            ex, ey = bp.argmax_315(mp, a)
            e = bp.error_315((ex, ey), dist)
            per_event[A].append(e)
            cond_rows.append([A, "projector", dist, wavs[i], f"{e:.6f}",
                              f"{ex:.6f}", f"{ey:.6f}", "1000-12000"])
            maps_sum[a] = mp if maps_sum[a] is None else maps_sum[a] + mp
            nm = mp / mp.max()
            maps_norm_sum[a] = nm if maps_norm_sum[a] is None else maps_norm_sum[a] + nm
    stats = {}
    for A, a in ALGOS:
        aa = arch_all[A]
        n = min(n_cmp, len(aa)) if aa else n_cmp
        errs = per_event[A]
        ok = sum(abs(errs[i] - aa[i]) < 2e-3 for i in range(n)) if aa else 0
        stats[A] = {
            "n": len(errs),
            "match_archived": f"{ok}/{n}" if aa else "no-archive",
            "mean": round(float(np.mean(errs)), 4),
            "median": round(float(np.median(errs)), 4),
            "arch_mean": round(float(np.mean(aa[:len(errs)])), 4) if aa else None,
            "arch_median": round(float(np.median(aa[:len(errs)])), 4) if aa else None,
        }
        # averaged power-map metric
        for tag, mm in (("avgmap_raw", maps_sum[a] / len(errs)),
                        ("avgmap_norm", maps_norm_sum[a] / len(errs))):
            ex, ey = bp.argmax_315(mm, a)
            stats[A][tag] = round(bp.error_315((ex, ey), dist), 4)
        if (freq, dist) in TT:
            stats[A]["truth_table"] = TT[(freq, dist)][A]
    rows += cond_rows
    report[cname] = {"scale": s, "stats": stats}
    print(f"{cname} s={s}: " + " | ".join(
        f"{A} match={stats[A]['match_archived']} mu={stats[A]['mean']} "
        f"arch_mu={stats[A]['arch_mean']} avg_raw={stats[A]['avgmap_raw']} "
        f"avg_norm={stats[A]['avgmap_norm']}"
        + (f" tt={stats[A]['truth_table']}" if (freq, dist) in TT else "")
        for A, _ in ALGOS), flush=True)

with open(os.path.join(OUT, "e1_errors.csv"), "w", newline="") as f:
    csv.writer(f).writerows(rows)
with open(os.path.join(OUT, "e1_v3_report.json"), "w") as f:
    json.dump(report, f, ensure_ascii=False, indent=1)
print("DONE", flush=True)
