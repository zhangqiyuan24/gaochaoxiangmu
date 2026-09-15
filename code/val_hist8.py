"""val_hist v8: per-condition scale calibration + full validation.
Phase 1: for each of the 21 tone conditions, sweep s in 1.200..1.260 step
.0025 on the first 3 events (mvdr+cbf), pick best-matching s (ties -> closest
to 1.24).  Phase 2: at the chosen s, run all events x 4 algos and report
per-event matches (mvdr/cbf) + mean/median comparison (music/music_csdm).
"""
import csv
import glob
import json
import os
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.expanduser("~/vsip/code"))
import beamforming_port as bp

BASE = os.path.expanduser("~/vsip/data/s315/bubble wave-2024.3.15 1")
REF = os.path.expanduser("~/vsip/results/ref_315")
BAND = np.arange(1000, 12001, 500)
GY = np.linspace(0, 6, 217)
GX = np.linspace(-3, 3, 217)
xx, yy = np.meshgrid(GX, GY, indexing="ij")
PTS = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
Hc = bp.HPOS - np.array([0.75, 0, 0])


def audioread2(path):
    with wave.open(path, "rb") as w:
        nch, fr, nf, sw = w.getnchannels(), w.getframerate(), w.getnframes(), w.getsampwidth()
        raw = w.readframes(nf)
    if sw == 2:
        y = np.frombuffer(raw, dtype="<i2").astype(np.float64).reshape(-1, nch) / 32768.0
    elif sw == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        v = (b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)).astype(np.int64)
        v = np.where(v >= 1 << 23, v - (1 << 24), v)
        y = v.astype(np.float64).reshape(-1, nch) / float(1 << 23)
    else:
        y = np.frombuffer(raw, dtype="<i4").astype(np.float64).reshape(-1, nch) / 2147483648.0
    return y, fr


def arch(cname, A):
    p = os.path.join(REF, f"bubble_wave-2024.3.15_1__{cname}__{A}_error__Sheet1.csv")
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return [float(x[0]) for x in csv.reader(f) if x and x[0].strip()]


VCACHE = {}


def vecs_of(s):
    if s not in VCACHE:
        h = Hc * s
        tau = np.linalg.norm(PTS[:, None, :] - h[None, :, :], axis=2) / 1489.0
        VCACHE[s] = [np.exp(-1j * 2 * np.pi * f * tau) for f in BAND]
    return VCACHE[s]


def run(spec, s, algo):
    vv = vecs_of(s)
    conj = algo in ("mvdr", "music_csdm")
    scale = 1.0 / 8 if algo in ("mvdr", "music_csdm") else 1.0
    q = np.einsum("fi,fj->fij", spec, spec.conj() if conj else spec) * scale
    out = np.zeros(len(PTS), dtype=complex)
    for k in range(len(BAND)):
        v = vv[k]
        qq = q[k] + 1e-3 * np.eye(8)
        if algo == "cbf":
            out += np.einsum("pi,ij,pj->p", v.conj(), qq.conj().T, v)
        elif algo == "mvdr":
            qi = np.linalg.inv(qq)
            out += 1.0 / np.einsum("pi,ij,pj->p", v.conj(), qi, v)
        else:
            if np.allclose(qq, qq.conj().T, atol=1e-10):
                w, evec = np.linalg.eigh(qq)
                en = evec[:, :7]
            else:
                w, evec = np.linalg.eig(qq)
                idx = np.lexsort((np.angle(w), np.abs(w)))
                en = evec[:, idx[:7]]
            pp = en @ en.conj().T
            out += 1.0 / np.einsum("pi,ij,pj->p", v.conj(), pp, v)
    a = np.abs(out).reshape(len(GX), len(GY))
    if algo == "music":
        a = a[::-1, :]
    ix, iy = np.unravel_index(np.argmax(a), a.shape)
    return GX[ix], GY[iy]


conds = []
for d in sorted(glob.glob(os.path.join(BASE, "*Hz *"))):
    b = os.path.basename(d)
    freq, dist = b.split("Hz ")
    conds.append((f"{freq}Hz_{dist}", d, int(dist)))

SS = [round(1.200 + 0.0025 * i, 4) for i in range(25)]
report = {}
for cname, d, dist in conds:
    wavs = sorted(os.path.basename(w) for w in glob.glob(os.path.join(d, "*.wav")))
    a_mv, a_cb = arch(cname, "MVDR"), arch(cname, "CBF")
    n = min(3, len(wavs), len(a_mv), len(a_cb))
    if n == 0:
        continue
    specs = []
    for w in wavs[:n]:
        y, fs = audioread2(os.path.join(d, w))
        specs.append(np.fft.fft(y, n=int(fs), axis=0)[BAND - 1, :])
    best = (-1, 1.24)
    for s in SS:
        m = sum(abs(np.hypot(*run(sp, s, "mvdr")) - dist - a_mv[i]) < 2e-3
                for i, sp in enumerate(specs))
        m += sum(abs(np.hypot(*run(sp, s, "cbf")) - dist - a_cb[i]) < 2e-3
                 for i, sp in enumerate(specs))
        if m > best[0] or (m == best[0] and abs(s - 1.24) < abs(best[1] - 1.24)):
            best = (m, s)
    s_c = best[1]
    # phase 2: full validation at s_c
    wavs_all = wavs[: min(len(wavs), 20)]
    na = min(len(a_mv), 20)
    stats = {}
    samples = {}
    for A, algo in (("MVDR", "mvdr"), ("CBF", "cbf"), ("MUSIC", "music"), ("MUSIC_CSDM", "music_csdm")):
        aa = arch(cname, A)
        if not aa:
            continue
        errs = []
        ok = 0
        for i, w in enumerate(wavs_all[: min(len(wavs_all), len(aa))]):
            y, fs = audioread2(os.path.join(d, w))
            ex, ey = run(np.fft.fft(y, n=int(fs), axis=0)[BAND - 1, :], s_c, algo)
            e = float(np.hypot(ex, ey - dist))
            errs.append(e)
            ok += abs(e - aa[i]) < 2e-3
        stats[A] = (ok, len(errs), round(float(np.mean(errs)), 4), round(float(np.median(errs)), 4),
                    round(float(np.mean(aa[:len(errs)])), 4), round(float(np.median(aa[:len(errs)])), 4))
    report[cname] = {"s": s_c, "phase1": int(best[0]), "stats": stats}
    print(f"{cname}: s={s_c} (p1 {best[0]}/{2*n}) " +
          " ".join(f"{A} {v[0]}/{v[1]} mine_mu={v[2]}/md={v[3]} arch_mu={v[4]}/md={v[5]}"
                   for A, v in stats.items()), flush=True)

with open(os.path.expanduser("~/vsip/results/val_hist8_report.json"), "w") as f:
    json.dump(report, f, ensure_ascii=False, indent=1)
print("DONE", flush=True)
