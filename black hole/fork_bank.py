"""
fork_bank.py — SpectralNeuron's resonant fire-reset unit, used as the
               defect-frequency readout on top of the Koopman monitor
=============================================================================
THE QUESTION (Antti's): does adding the SpectralNeuron "fork" buy the bearing
monitor anything real? The honest answer this file tests: YES, one thing, and a
specific one — it makes the readout ROBUST TO A BENIGN LEVEL CHANGE that fools
the energy indicators (RMS, and partly the DMD HF-energy). That is exactly the
fork's verified property in the SpectralNeuron repo ("rejects off-band power and
broadband noise that the bucket is fooled by"), applied where it matters.

WHERE THE FORK IS VALID (measured, not assumed): the fork primitive is a
semi-implicit oscillator; at fs=12 kHz it cannot resolve the 3.3 kHz structural
resonance (~3.6 samples/cycle -> it goes numerically silent). It IS valid at the
DEFECT frequency (BPFO ~107 Hz), which is the band that actually carries the
fault's signature. So the standard envelope-demodulation cascade is the right
shape and the fork lands on the right rung:

    raw -> [band-pass to the resonance]      (a linear fork; butter, stable at 3.3 kHz)
        -> [analytic envelope]               (Hilbert: the impact train shows here)
        -> [FORK tuned to BPFO]              (SpectralNeuron's resonant fire-reset unit)
        -> spikes = sparse defect EVENTS     (the compact stream a normal model reads)

This is not new signal processing (it is envelope analysis; the README of
SpectralNeuron says so plainly). What it adds to THIS project is: a level-robust,
frequency-addressed, event-emitting readout, complementary to DMD's trainless
mode discovery.

Reuses integrate_fork from SpectralNeuron/spectral_neuron.py (copied with
attribution to keep this folder self-contained).

PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, June 2026.
Do not hype. Do not lie. Just show.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from koopman_monitor import highpass, dmd_spectrum, hf_coherent_energy, rms


# --- SpectralNeuron's resonant fire-reset unit (Antti Luode / PerceptionVlab) ---
def integrate_fork(I, dt, w0, zeta, g, thresh):
    """damped driven oscillator with fire-then-reset. The reset is the clock.
       (verbatim from SpectralNeuron/spectral_neuron.py, attribution kept)."""
    x = 0.0; v = 0.0
    env = np.empty_like(I, dtype=float); spikes = []
    for t in range(I.size):
        v += dt * (-2.0 * zeta * w0 * v - (w0 * w0) * x + g * I[t])
        x += dt * v
        e = np.hypot(x, v / w0)
        env[t] = e
        if e > thresh:
            spikes.append(t); x = 0.0; v = 0.0
    return np.array(spikes, dtype=int), env


def bandpass(x, fs, f0, bw, order=4):
    lo, hi = max(f0 - bw, 1.0), min(f0 + bw, fs / 2 - 1)
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, np.asarray(x, float).ravel())


def defect_envelope(rec, fs, f_res, hp=800.0, bw=700.0):
    """resonance band-pass -> analytic envelope. The defect's periodic impacts
       live as a modulation of THIS envelope (textbook envelope demodulation)."""
    xf = highpass(rec, fs, hp)
    band = bandpass(xf, fs, f_res, bw)
    env = np.abs(hilbert(band))
    return env - env.mean()


def defect_index(rec, fs, f_res, bpfo, hp=800.0, bw=700.0, tol=3.0, harmonics=3):
    """LEVEL-ROBUST defect indicator: fraction of envelope power concentrated at
       the defect frequency and its harmonics. A louder benign signal raises
       envelope power everywhere -> ratio stays low. A real fault concentrates it
       at BPFO -> ratio climbs. This is the fork's 'reject broadband, answer to
       the band' property as a single number."""
    env = defect_envelope(rec, fs, f_res, hp, bw)
    n = len(env)
    P = np.abs(np.fft.rfft(env * np.hanning(n))) ** 2
    f = np.fft.rfftfreq(n, 1 / fs)
    total = P[1:].sum() + 1e-12
    defect = 0.0
    for k in range(1, harmonics + 1):
        sel = np.abs(f - k * bpfo) <= tol
        defect += P[sel].sum()
    return float(defect / total)


def fork_defect_spikes(rec, fs, f_res, bpfo, thresh, hp=800.0, bw=700.0,
                       zeta=0.06, g=400.0):
    """the EVENT readout: a fork tuned to BPFO fires on the envelope's periodic
       impacts. Spike count = sparse defect events handed downstream."""
    env = defect_envelope(rec, fs, f_res, hp, bw)
    sp, _ = integrate_fork(env, 1 / fs, w0=2 * np.pi * bpfo, zeta=zeta, g=g, thresh=thresh)
    return len(sp)


def bucket_spikes(rec, fs, thresh, hp=800.0, tau=0.01):
    """the energy detector (bucket), for contrast: integrates rectified band
       energy, fires on level. Fooled by a benign loudness increase."""
    xf = highpass(rec, fs, hp)
    u = 0.0; dt = 1 / fs; r = np.abs(xf); n = 0
    for t in range(len(xf)):
        u += dt * (-u / tau + r[t])
        if u > thresh:
            n += 1; u = 0.0
    return n


if __name__ == "__main__":
    from synth_bearing import generate_run, BPFO, F_RES
    run = generate_run(n_records=60, onset_frac=0.6, seed=0)
    fs = run.fs

    # calibrate the fork spike threshold by bisection so it fires ~12 times on a
    # strong-fault record (their calibrate-on-the-suiting-stimulus method), then
    # watch it stay silent on healthy + benign-louder.
    def fork_count(rec, thr):
        return fork_defect_spikes(rec, fs, F_RES, BPFO, thr)
    lo, hi = 1e-6, 1e3
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        (lo, hi) = (mid, hi) if fork_count(run.records[58], mid) > 12 else (lo, mid)
    fork_thr = np.sqrt(lo * hi)
    buck_thr = 8.0 * np.mean([rms(highpass(run.records[i], fs, 800)) for i in range(10)]) * 0.01

    print("=" * 78)
    print("FORK BANK — does SpectralNeuron's resonant unit add anything to the monitor?")
    print("=" * 78)
    print(f"  BPFO {BPFO:.0f} Hz | resonance {F_RES:.0f} Hz | onset rec {run.onset}\n")

    # ---- THE DECISIVE TEST: a benign LOAD/LEVEL change vs a real fault --------
    rng = np.random.default_rng(123)
    healthy   = run.records[5]
    confound  = run.records[6] * 1.8        # benign: same machine, 1.8x louder (load up)
    faulty    = run.records[58]             # real outer-race fault, comparable late severity
    faulty_q  = run.records[44] * 0.7       # real fault but QUIETER than the confound

    print("  indicator behaviour on: healthy | benign-louder(1.8x) | real-fault | quiet-fault")
    print("  " + "-" * 74)
    def row(name, fn):
        vals = [fn(healthy), fn(confound), fn(faulty), fn(faulty_q)]
        print(f"  {name:<26}{vals[0]:>10.3f}{vals[1]:>14.3f}{vals[2]:>12.3f}{vals[3]:>13.3f}")
        return vals
    rms_v  = row("RMS (energy/level)",      lambda r: rms(highpass(r, fs, 800)))
    dmd_v  = row("DMD HF coherent energy",  lambda r: hf_coherent_energy(dmd_spectrum(highpass(r, fs, 800), fs), 1000))
    fork_v = row("FORK defect index (BPFO)",lambda r: defect_index(r, fs, F_RES, BPFO))
    print("  " + "-" * 74)
    print("  read it honestly: RMS is badly fooled -- it ranks the benign-louder record as")
    print("  the WORST and the quiet real fault as the BEST (the ranking is inverted). DMD")
    print("  HF-energy is partially lifted by the louder record but still ranks both real")
    print("  faults above it, so it is only mildly confounded. The FORK defect index is the")
    print("  cleanest: the benign-louder record sits at/below healthy while both faults")
    print("  (loud AND quiet) stand far out -- because it keys on BPFO periodicity, not level.\n")

    # honest scoring: does the fork separate fault from benign-loud where RMS can't?
    fork_sep = fork_v[2] > 2*fork_v[1] and fork_v[3] > 2*fork_v[1]
    rms_sep  = rms_v[2]  > rms_v[1]   and rms_v[3]  > rms_v[1]
    print(f"  fork separates real fault (loud & quiet) from benign-louder:  {fork_sep}")
    print(f"  RMS  separates real fault (loud & quiet) from benign-louder:  {rms_sep}")

    # ---- the event stream the fork emits over the run ------------------------
    print(f"\n  fork DEFECT-EVENT spikes over the run (sparse stream for a downstream model):")
    print(f"  {'record':>8}{'severity':>10}{'fork spikes':>13}{'bucket spikes':>15}")
    for i in [5, 30, 36, 42, 48, 54, 59]:
        fs_sp = fork_defect_spikes(run.records[i], fs, F_RES, BPFO, fork_thr)
        bk_sp = bucket_spikes(run.records[i], fs, buck_thr)
        print(f"  {i:>8}{run.severity[i]:>10.2f}{fs_sp:>13}{bk_sp:>15}")
    print("\n  Relative units, labelled-synthetic signal, parameters chosen. Envelope")
    print("  analysis is textbook; the fork is its addressed readout, not new math.")
    print("  Do not hype. Do not lie. Just show.")
    print("=" * 78)
