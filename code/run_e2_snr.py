"""E2 v3: SNR-robustness sweep on the validated projector conditions.

Design (per writer spec, main.tex E2 placeholder):
  - post-processing injection of band-matched additive noise into the
    original 8-channel recordings (analysis-band white: complex Gaussian
    added at the 23 analysis FFT bins 1000:500:12000 Hz of the zero-padded
    single-snapshot spectrum, conjugate-symmetric <=> a real band-line
    noise sequence; the pipeline consumes exactly these bins)
  - in-band SNR set on the array-averaged bin power:
        mean_ch sum_bins |N|^2  =  mean_ch sum_bins |X|^2 / 10^(snr/10)
  - SNR points: clean (no injection) and +20..-20 dB step 5
  - common random numbers: identical noisy record for all four algorithms
    at each (condition, event, SNR); seed derived from that triple
  - failure criterion err > 0.5 m vs truth (0, dist); critical SNR =
    smallest injected SNR keeping the event-mean error <= 0.5 m
Outputs: ~/vsip/results/e2_v3/{e2_snr.csv,e2_snr_events.csv,e2_summary.json}
"""
import csv
import json
import os
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.expanduser("~/vsip/code"))
import beamforming_port as bp

BASE = os.path.expanduser("~/vsip/data/s315/bubble wave-2024.3.15 1")
OUT = os.path.expanduser("~/vsip/results/e2_v3")
os.makedirs(OUT, exist_ok=True)

CONDS = os.environ.get("E2_CONDS", "5000Hz_3,3000Hz_2,7000Hz_3").split(",")
SNRS = [20, 15, 10, 5, 0, -5, -10, -15, -20]
ALGOS = [("CBF", "cbf"), ("MVDR", "mvdr"), ("MUSIC", "music"), ("MUSIC_CSDM", "music_csdm")]

scales = {}
rep_path = os.path.expanduser("~/vsip/results/val_hist8_report.json")
if os.path.exists(rep_path):
    with open(rep_path) as f:
        scales = {k: v["s"] for k, v in json.load(f).items()}


def band_noise_spec(rng, nch, sig_bin_power, snr_db):
    """Complex (23, nch) noise added directly at the 23 analysis bins of the
    zero-padded (n = Fs) single-snapshot spectrum.  Equivalent to adding a
    real band-line noise sequence (conjugate-symmetric DFT) and re-running
    the FFT: the localization pipeline sees exactly these bins.  Scaled so
    the array-averaged bin power = sig_bin_power / 10^(snr/10)."""
    N = (rng.standard_normal((len(bp.FREQS), nch))
         + 1j * rng.standard_normal((len(bp.FREQS), nch))) / np.sqrt(2)
    r = 10.0 ** (snr_db / 10.0)
    p_n = np.mean(np.sum(np.abs(N) ** 2, axis=0))
    N *= np.sqrt(sig_bin_power / r / p_n)
    return N


ev_rows = [["condition", "algorithm", "snr_db", "sample_id",
            "error_m", "est_x_m", "est_y_m"]]
sum_rows = [["condition", "algorithm", "snr_db", "n",
             "mean_err_m", "std_err_m", "median_err_m", "success_rate_le_0.5m"]]
summary = {}

for cname in CONDS:
    freq, dist = cname.split("Hz_")
    dist = int(dist)
    d = os.path.join(BASE, f"{freq}Hz {dist}")
    s = scales.get(cname, 1.24)
    tau = bp.grid_delays_315(s)
    wavs = bp.matlab_wavs(d, 20)
    acc = {(A, tag): [] for A, _ in ALGOS for tag in ["clean"] + SNRS}
    for i, w in enumerate(wavs):
        y, fs = bp.audioread(os.path.join(d, w))
        spec = bp.spec_315(y, fs)
        sig_pow = float(np.mean(np.sum(np.abs(spec) ** 2, axis=0)))
        for tag in ["clean"] + SNRS:
            if tag == "clean":
                sp = spec
            else:
                rng = np.random.default_rng([zlib.crc32(cname.encode()), i, int(tag) + 100])
                sp = spec + band_noise_spec(rng, spec.shape[1], sig_pow, float(tag))
            mp_all = bp.maps_315(sp, s, tau)
            for A, a in ALGOS:
                ex, ey = bp.argmax_315(mp_all[a], a)
                e = bp.error_315((ex, ey), dist)
                acc[(A, tag)].append(e)
                ev_rows.append([cname, A, "clean" if tag == "clean" else tag,
                                w, f"{e:.6f}", f"{ex:.6f}", f"{ey:.6f}"])
    cond_sum = {}
    for A, _ in ALGOS:
        cond_sum[A] = {}
        for tag in ["clean"] + SNRS:
            e = np.array(acc[(A, tag)])
            row = [cname, A, "clean" if tag == "clean" else tag, len(e),
                   f"{e.mean():.4f}", f"{e.std():.4f}", f"{np.median(e):.4f}",
                   f"{np.mean(e <= 0.5):.3f}"]
            sum_rows.append(row)
            cond_sum[A][str(tag)] = {"mean": round(float(e.mean()), 4),
                                     "std": round(float(e.std()), 4),
                                     "median": round(float(np.median(e)), 4),
                                     "success": round(float(np.mean(e <= 0.5)), 3)}
        clean_ok = cond_sum[A]["clean"]["mean"] <= 0.5
        crit = None
        if clean_ok:
            for tag in SNRS:                       # descending SNR
                if cond_sum[A][str(tag)]["mean"] <= 0.5:
                    crit = tag
                else:
                    break
        cond_sum[A]["critical_snr_db"] = crit
    summary[cname] = {"scale": s, "n_events": len(wavs), "algos": cond_sum}
    print(f"{cname} s={s} n={len(wavs)}: " + " | ".join(
        f"{A} clean_mu={cond_sum[A]['clean']['mean']} crit={cond_sum[A]['critical_snr_db']}"
        for A, _ in ALGOS), flush=True)

with open(os.path.join(OUT, "e2_snr.csv"), "w", newline="") as f:
    csv.writer(f).writerows(sum_rows)
with open(os.path.join(OUT, "e2_snr_events.csv"), "w", newline="") as f:
    csv.writer(f).writerows(ev_rows)
with open(os.path.join(OUT, "e2_summary.json"), "w") as f:
    json.dump(summary, f, ensure_ascii=False, indent=1)
print("DONE", flush=True)
