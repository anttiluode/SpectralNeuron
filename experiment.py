"""
experiment.py
==================================================================
Two measurements, both designed so they can come back NEGATIVE.

PART A -- Dissociation battery
    Both units are calibrated to fire the same number of times on the
    one stimulus that flatters the fork (loud, on-band). Then we feed
    six stimuli and tabulate spikes. If the two units track each other
    across all six, the spectral framing added nothing. If they
    dissociate -- fork fires on quiet-in-band, bucket fires on
    loud-out-of-band and on broadband noise -- then "spectral address"
    means the unit answers to WHICH frequency is present, not HOW MUCH
    signal.

PART B -- Multiplexing over one wire ("spectral islands")
    Three forks at different frequencies share a single summed input
    plus noise, each frequency carrying its own slowly-varying message.
    Question: does each fork's firing track ONLY its own channel's
    message and ignore the others? That is frequency-division
    multiplexing (old idea, radio) rebuilt from integrate-fire-reset
    units. We measure channel crosstalk and compare against three
    buckets, which can only see total power and so cannot separate.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spectral_neuron import (
    integrate_bucket, integrate_fork, calibrate_threshold,
)

rng = np.random.default_rng(7)

# ---- shared timebase ----
FS = 2000.0          # Hz
DT = 1.0 / FS
T = 2.0              # seconds per stimulus
N = int(T * FS)
tvec = np.arange(N) * DT

# ---- fork tuning ----
F0 = 40.0
W0 = 2 * np.pi * F0
ZETA = 0.04          # Q = 1/(2 zeta) = 12.5  -> fairly narrow band
G = 1.0

BUCKET_TAU = 0.05
TARGET = 20          # spikes on the reference stimulus

def sine(freq, amp):
    return amp * np.sin(2 * np.pi * freq * tvec)

# ----------------------------------------------------------------------
# PART A
# ----------------------------------------------------------------------
def part_a():
    # reference = loud, on-band (best case for the fork). Calibrate BOTH here.
    ref = sine(F0, 1.0)

    fork_kw = dict(w0=W0, zeta=ZETA, g=G)
    bkt_kw  = dict(tau=BUCKET_TAU)

    fork_thr = calibrate_threshold(integrate_fork, fork_kw, ref, DT, TARGET)
    bkt_thr  = calibrate_threshold(integrate_bucket, bkt_kw, ref, DT, TARGET)
    fork_kw["thresh"] = fork_thr
    bkt_kw["thresh"]  = bkt_thr

    noise_hi = lambda s: rng.normal(0, s, N)

    battery = {
        "1 on-band  quiet  (40Hz, A=0.2)":      sine(F0, 0.2),
        "2 off-band quiet  (120Hz, A=0.2)":     sine(120, 0.2),
        "3 on-band  LOUD   (40Hz, A=1.0) [ref]": sine(F0, 1.0),
        "4 off-band LOUD   (120Hz, A=1.0)":     sine(120, 1.0),
        "5 broadband noise (sigma=1.0)":        noise_hi(1.0),
        "6 quiet on-band in noise (0.2 + n0.8)": sine(F0, 0.2) + noise_hi(0.8),
    }

    rows = []
    traces = {}
    for name, I in battery.items():
        fsp, fenv = integrate_fork(I, DT, **fork_kw)
        bsp, btr  = integrate_bucket(I, DT, **bkt_kw)
        rows.append((name, fsp.size, bsp.size))
        traces[name] = (I, fenv, fork_thr, btr, bkt_thr)

    return rows, traces, (fork_thr, bkt_thr)


# ----------------------------------------------------------------------
# PART B  -- three channels, one wire
# ----------------------------------------------------------------------
def slow_envelope(seed, lo=0.0, hi=1.0):
    """A smooth, slowly-varying positive message in [lo,hi]."""
    r = np.random.default_rng(seed)
    raw = r.normal(0, 1, N)
    # smooth with a long moving average (~150 ms)
    k = int(0.15 * FS)
    ker = np.ones(k) / k
    s = np.convolve(raw, ker, mode="same")
    s = (s - s.min()) / (np.ptp(s) + 1e-9)
    return lo + (hi - lo) * s

def part_b():
    freqs = [25.0, 45.0, 75.0]
    msgs  = [slow_envelope(s) for s in (11, 22, 33)]

    # one shared wire: sum of the three modulated carriers + noise
    wire = np.zeros(N)
    for f, m in zip(freqs, msgs):
        wire += m * np.sin(2 * np.pi * f * tvec)
    wire += rng.normal(0, 0.6, N)        # shared-medium noise

    # calibrate each fork on its own loud on-band carrier
    fork_units = []
    for f in freqs:
        ref = 1.0 * np.sin(2 * np.pi * f * tvec)
        kw = dict(w0=2 * np.pi * f, zeta=ZETA, g=G)
        kw["thresh"] = calibrate_threshold(integrate_fork, kw, ref, DT, 60)
        fork_units.append(kw)

    # three buckets, identical, calibrated on the same reference power
    ref_pow = 1.0 * np.sin(2 * np.pi * freqs[1] * tvec)
    bkw = dict(tau=BUCKET_TAU)
    bkw["thresh"] = calibrate_threshold(integrate_bucket, bkw, ref_pow, DT, 60)

    # spike-rate (binned) for each unit on the shared wire
    nbins = 40
    edges = np.linspace(0, N, nbins + 1).astype(int)
    centers = 0.5 * (edges[:-1] + edges[1:]) * DT

    def binned_rate(spikes):
        r = np.zeros(nbins)
        for b in range(nbins):
            r[b] = np.sum((spikes >= edges[b]) & (spikes < edges[b + 1]))
        return r

    fork_rates, bucket_rates = [], []
    for kw in fork_units:
        sp, _ = integrate_fork(wire, DT, **kw)
        fork_rates.append(binned_rate(sp))
    for _ in freqs:
        sp, _ = integrate_bucket(wire, DT, **bkw)
        bucket_rates.append(binned_rate(sp))   # identical buckets -> identical

    # downsample each message to the bins for correlation
    msg_binned = []
    for m in msgs:
        mb = np.array([m[edges[b]:edges[b + 1]].mean() for b in range(nbins)])
        msg_binned.append(mb)

    def corr(a, b):
        a = a - a.mean(); b = b - b.mean()
        d = np.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / d) if d > 0 else 0.0

    # crosstalk matrix: fork i  vs  message j
    fork_xtalk = np.array([[corr(fork_rates[i], msg_binned[j])
                            for j in range(3)] for i in range(3)])
    bucket_xtalk = np.array([[corr(bucket_rates[i], msg_binned[j])
                              for j in range(3)] for i in range(3)])

    return dict(freqs=freqs, centers=centers, msg_binned=msg_binned,
                fork_rates=fork_rates, bucket_rates=bucket_rates,
                fork_xtalk=fork_xtalk, bucket_xtalk=bucket_xtalk, wire=wire)


# ----------------------------------------------------------------------
def make_figure(rowsA, tracesA, B, path):
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(4, 3, hspace=0.55, wspace=0.3)

    # --- A: spike-count bars across the battery ---
    axA = fig.add_subplot(gs[0, :])
    names = [r[0] for r in rowsA]
    fork_c = [r[1] for r in rowsA]
    bkt_c  = [r[2] for r in rowsA]
    xpos = np.arange(len(names))
    axA.bar(xpos - 0.2, fork_c, 0.4, label="Fork (resonant)", color="#1f77b4")
    axA.bar(xpos + 0.2, bkt_c, 0.4, label="Bucket (energy LIF)", color="#d62728")
    axA.set_xticks(xpos)
    axA.set_xticklabels([n.split("(")[0].strip() for n in names],
                        rotation=20, ha="right", fontsize=8)
    axA.set_ylabel("spikes in 2 s")
    axA.set_title("PART A  -  both units calibrated to fire equally on stimulus 3 "
                  "(loud, on-band), then diverge", fontsize=10)
    axA.legend(fontsize=8)

    # --- A: two example envelope traces ---
    for col, key in enumerate(["1 on-band  quiet  (40Hz, A=0.2)",
                               "4 off-band LOUD   (120Hz, A=1.0)",
                               "6 quiet on-band in noise (0.2 + n0.8)"]):
        ax = fig.add_subplot(gs[1, col])
        I, fenv, fthr, btr, bthr = tracesA[key]
        ax.plot(tvec, fenv, color="#1f77b4", lw=0.8, label="fork ring")
        ax.axhline(fthr, color="#1f77b4", ls=":", lw=0.8)
        ax.plot(tvec, btr, color="#d62728", lw=0.8, label="bucket fill")
        ax.axhline(bthr, color="#d62728", ls=":", lw=0.8)
        ax.set_title(key.split("(")[0].strip(), fontsize=8)
        ax.set_xlabel("s", fontsize=7)
        if col == 0:
            ax.legend(fontsize=6)

    # --- B: shared wire + recovered channels ---
    axw = fig.add_subplot(gs[2, :])
    axw.plot(tvec, B["wire"], color="#777", lw=0.4)
    axw.set_title("PART B  -  one shared wire: 3 carriers (25/45/75 Hz) "
                  "+ noise, each carrying its own slow message", fontsize=10)
    axw.set_xlabel("s", fontsize=7)
    axw.set_ylabel("amplitude")

    colors = ["#2ca02c", "#9467bd", "#ff7f0e"]
    for i, f in enumerate(B["freqs"]):
        ax = fig.add_subplot(gs[3, i])
        # normalise for overlay
        mr = B["msg_binned"][i]; mr = (mr - mr.min()) / (np.ptp(mr) + 1e-9)
        fr = B["fork_rates"][i]; fr = fr / (fr.max() + 1e-9)
        ax.plot(B["centers"], mr, color="k", lw=1.4, label="true message")
        ax.plot(B["centers"], fr, color=colors[i], lw=1.4, label="fork rate")
        ax.set_title(f"channel {int(f)} Hz   "
                     f"(self r={B['fork_xtalk'][i,i]:.2f})", fontsize=8)
        ax.set_xlabel("s", fontsize=7)
        if i == 0:
            ax.legend(fontsize=6)

    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"figure -> {path}")


if __name__ == "__main__":
    rowsA, tracesA, (fthr, bthr) = part_a()

    print("\n" + "=" * 64)
    print("PART A  -  dissociation battery")
    print("(both calibrated to ~20 spikes on stimulus 3: loud, on-band)")
    print("=" * 64)
    print(f"{'stimulus':42s}{'fork':>6s}{'bucket':>8s}")
    print("-" * 64)
    for name, fc, bc in rowsA:
        print(f"{name:42s}{fc:6d}{bc:8d}")
    print("=" * 64)

    B = part_b()
    print("\n" + "=" * 64)
    print("PART B  -  multiplexing over one wire")
    print("fork crosstalk matrix  (rows = fork i, cols = message j)")
    print("diagonal = listens to own channel, off-diagonal = leakage")
    print("=" * 64)
    fx = B["fork_xtalk"]
    hdr = "          " + "".join(f"msg{int(f):>3d}Hz" for f in B["freqs"])
    print(hdr)
    for i, f in enumerate(B["freqs"]):
        print(f"fork{int(B['freqs'][i]):>4d}Hz " +
              "".join(f"{fx[i,j]:>8.2f}" for j in range(3)))
    print("-" * 64)
    print("bucket crosstalk matrix  (all buckets identical -> all see total power)")
    bx = B["bucket_xtalk"]
    for i, f in enumerate(B["freqs"]):
        print(f"bkt {int(B['freqs'][i]):>4d}Hz " +
              "".join(f"{bx[i,j]:>8.2f}" for j in range(3)))
    print("=" * 64)

    diag = np.diag(fx).mean()
    offdiag = (fx.sum() - np.trace(fx)) / 6.0
    print(f"\nfork:   mean self-correlation {diag:.2f}   "
          f"mean crosstalk {offdiag:.2f}   ratio {diag/abs(offdiag) if offdiag else float('inf'):.1f}x")
    bdiag = np.diag(bx).mean()
    boff = (bx.sum() - np.trace(bx)) / 6.0
    print(f"bucket: mean self-correlation {bdiag:.2f}   "
          f"mean crosstalk {boff:.2f}")

    make_figure(rowsA, tracesA, B, "spectral_address_results.png")
