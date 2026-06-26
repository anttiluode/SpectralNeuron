"""
synth_bearing.py — a physically-motivated run-to-failure bearing signal
=======================================================================
LABELLED SYNTHETIC (stated plainly, per the ledger): this is NOT measured data.
It is the textbook outer-race-defect model, used here only to give the instrument
a real SIGNAL TYPE with a KNOWN fault-onset record, so detection can be scored
honestly. The real-data loaders (loaders.py) drop the same monitor onto CWRU /
NASA-IMS / MIMII without changing a line of koopman_monitor.py.

THE PHYSICS (standard rolling-element bearing model):
  healthy record = structural resonances excited by broadband noise + a little
                   shaft-rate content + sensor noise. The HF resonance rings and
                   DECAYS between random kicks -> it is noise-damped (rho < 1).
  faulty record  = above PLUS a periodic impact train at the outer-race defect
                   frequency BPFO. Each impact rings the HF resonance; because the
                   kicks now arrive every 1/BPFO seconds, the resonance is kept
                   alive -> it becomes a SUSTAINED mode (rho -> 1). Impact
                   amplitude ramps from onset to end-of-life. This is exactly the
                   impulse-train-modulating-a-resonance signature kurtosis and
                   envelope analysis are built for -- a fair, not rigged, test.

Each "record" mimics a periodic condition-monitoring snapshot (like NASA-IMS,
which logs ~1 s of vibration every ~10 min): record index ~ machine age.

PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, June 2026.
Do not hype. Do not lie. Just show.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

FS = 12000.0          # Hz, matches CWRU drive-end sampling
SHAFT_HZ = 29.95      # ~1797 rpm, the classic CWRU speed
BPFO_FACTOR = 3.585   # representative outer-race factor for a 9-ball bearing
BPFO = SHAFT_HZ * BPFO_FACTOR     # ~107 Hz outer-race defect frequency
F_RES = 3300.0        # a structural resonance the defect impacts ring (Hz)
ZETA = 0.05           # damping ratio of that resonance per impact


@dataclass
class Run:
    records: np.ndarray      # (n_records, n_samples)
    severity: np.ndarray     # 0..1 fault amplitude per record (0 = healthy)
    onset: int               # first faulty record index
    fs: float
    bpfo: float
    f_res: float


def _impact_ring(n, fs, f_res=F_RES, zeta=ZETA):
    """one decaying-sinusoid impact response (a struck resonance)."""
    t = np.arange(n) / fs
    wr = 2 * np.pi * f_res
    return np.exp(-zeta * wr * t) * np.sin(wr * np.sqrt(max(1 - zeta**2, 0.0)) * t)


def _record(severity, fs, n, rng):
    t = np.arange(n) / fs
    # background: shaft harmonics (low-freq) + broadband-excited resonance + noise
    x = 0.6 * np.sin(2*np.pi*SHAFT_HZ*t) + 0.3*np.sin(2*np.pi*2*SHAFT_HZ*t)
    # healthy HF: the resonance gets random small kicks and decays between them
    ring = _impact_ring(int(0.02*fs), fs)
    for _ in range(rng.poisson(8)):
        k = rng.integers(0, n - len(ring))
        x[k:k+len(ring)] += 0.15 * rng.standard_normal() * ring
    x += 0.25 * rng.standard_normal(n)                       # sensor noise

    # fault: periodic impacts at BPFO (with slip jitter), ringing the resonance
    if severity > 0:
        period = fs / BPFO
        jitter = 0.01 * period
        k = rng.uniform(0, period)
        while k < n - len(ring):
            kk = int(k + rng.normal(0, jitter))
            if 0 <= kk < n - len(ring):
                amp = severity * (1.0 + 0.2 * rng.standard_normal())
                x[kk:kk+len(ring)] += amp * ring
            k += period
    return x


def generate_run(n_records=60, onset_frac=0.6, n_samples=4096, fs=FS, seed=0):
    """run-to-failure: healthy until `onset`, then severity ramps 0->1."""
    rng = np.random.default_rng(seed)
    onset = int(n_records * onset_frac)
    recs, sev = [], []
    for i in range(n_records):
        if i < onset:
            s = 0.0
        else:
            s = (i - onset) / max(n_records - onset - 1, 1)   # 0 -> 1 ramp
        recs.append(_record(s, fs, n_samples, rng))
        sev.append(s)
    return Run(records=np.array(recs), severity=np.array(sev),
               onset=onset, fs=fs, bpfo=BPFO, f_res=F_RES)


if __name__ == "__main__":
    r = generate_run()
    print(f"generated {len(r.records)} records of {r.records.shape[1]} samples @ {r.fs:.0f} Hz")
    print(f"BPFO = {r.bpfo:.1f} Hz, resonance = {r.f_res:.0f} Hz, fault onset at record {r.onset}")
