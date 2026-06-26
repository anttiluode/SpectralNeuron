"""
koopman_monitor.py — the HKT temporal core, lifted off the webcam and turned
                     into a trainless streaming machine-health instrument
=============================================================================
This is the README's own next step #1+#2, made literal: "freeze the front end ->
make it a measurement" and "point it at a spinning fan, a vibrating beam, a motor
-- its modal eigenvalues are its health signature, and a mode drifting in
frequency or damping is a fault appearing BEFORE it breaks."

THE INSTRUMENT (no learning, no labels, no GPU):
  - a 1-D sensor stream (vibration/acoustic) is Takens delay-embedded into a
    Hankel matrix -- the dendritic delay line of the geometric neuron, on a
    scalar signal where it is the only way to get a state at all;
  - per window, EXACT DMD fits a one-step operator A = X2 X1^+ (SVD-truncated,
    the same X2 X1^+ as HKT, done properly) and we read its eigenvalues:
        f_i   = angle(lambda_i) * fs / (2*pi)      [Hz]   -- the rotation rate
        rho_i = |lambda_i|                          [-]    -- the persistence
    This is the eigen-constellation, on a machine instead of a face.
  - the health indicator is the persistence of the HIGH-FREQUENCY structure:
    a developing fault re-excites a structural resonance every defect period, so
    a high-frequency mode that was noise-damped (rho < 1) climbs toward rho -> 1.
    That rising rho is the fault becoming a sustained ringing mode.

WHY DMD AND NOT JUST AN FFT: the FFT tells you the frequency content; DMD tells
you each component's frequency AND its damping (rho), from a windowed stream,
trainless. Damping is the part that moves first and the part a spectrogram hides.

HONEST DESIGN CHOICE (a knob, not a rig): each window is mean-removed and
std-normalised before the fit, so the DMD indicator reports STRUCTURE (what
frequencies persist), NOT LEVEL. Level is the cheap baseline's job (RMS). This
makes DMD a complementary structural-change detector, deliberately blind to a
pure volume change -- which is the honest division of labour, and is stated so.

Grounding: exact DMD (Schmid 2010; Tu et al. 2014); Hankel/delay-coordinate DMD
== Takens embedding (Takens 1981; Brunton et al. HAVOK 2017); vibration-based
condition monitoring and operational modal analysis are established fields. The
contribution here is not the math; it is the trainless, streaming, interpretable
instrument form of it, carried over from the HKT line, and run against the
obvious simple baseline with the honest result reported.

PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, June 2026.
Do not hype. Do not lie. Just show.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from scipy.signal import butter, filtfilt

# --------------------------------------------------------------------------- #
#  Standard bearing-analysis front end: remove the deterministic shaft content
#  so the fault's high-frequency resonance is not buried. This is textbook
#  pre-processing (you band-pass to the resonance before any indicator); the
#  cutoff is a documented knob, not a rig. Applied identically to DMD AND to the
#  baselines, so the comparison stays fair.
# --------------------------------------------------------------------------- #
def highpass(x, fs, cut=800.0, order=4):
    b, a = butter(order, cut / (fs / 2), btype="high")
    return filtfilt(b, a, np.asarray(x, float).ravel())


# --------------------------------------------------------------------------- #
#  Takens / Hankel delay embedding  (the dendrite, on a scalar stream)
# --------------------------------------------------------------------------- #
def hankel(x: np.ndarray, d: int) -> np.ndarray:
    """delay-embed a 1-D signal into d-dim snapshots.
       returns H of shape (d, N) where column j is [x[j], ..., x[j+d-1]]^T."""
    x = np.asarray(x, float).ravel()
    N = len(x) - d + 1
    if N <= 2:
        raise ValueError(f"signal too short ({len(x)}) for delay {d}")
    # stride trick -> no copy of the big matrix
    s = x.strides[0]
    return np.lib.stride_tricks.as_strided(x, shape=(d, N), strides=(s, s)).copy()


# --------------------------------------------------------------------------- #
#  Exact DMD on one window  ->  eigenvalues (frequency, persistence)
# --------------------------------------------------------------------------- #
@dataclass
class Spectrum:
    freqs:  np.ndarray     # Hz, folded to [0, fs/2]
    rho:    np.ndarray     # |lambda|, persistence (1 = undamped/sustained)
    energy: np.ndarray     # |b_i| * ||phi_i||  -- how much of the window each mode carries
    lam:    np.ndarray     # raw complex eigenvalues
    resid:  float          # one-step reconstruction residual ||X2 - A X1|| / ||X2||


def dmd_spectrum(window: np.ndarray, fs: float, delay: int = 64,
                 rank: int = 20, normalise: bool = True) -> Spectrum:
    """fit one window's Koopman operator and read its constellation, WITH the
       proper DMD mode amplitudes -- so a high-|lambda| NOISE mode (which carries
       no energy) is not mistaken for a coherent fault mode (high |lambda| AND
       high energy). Reading |lambda| alone was the bug; energy is the fix."""
    w = np.asarray(window, float).ravel()
    w = w - w.mean()
    if normalise:
        sd = w.std()
        if sd > 1e-12:
            w = w / sd                      # concentrate variance -> a coherent fault
                                            # mode grabs a large amplitude; noise spreads thin
    H = hankel(w, delay)
    X1, X2 = H[:, :-1], H[:, 1:]

    U, S, Vt = np.linalg.svd(X1, full_matrices=False)        # exact DMD
    r = int(min(rank, np.sum(S > S[0] * 1e-10))) if S[0] > 0 else 1
    r = max(r, 1)
    Ur, Sr, Vr = U[:, :r], S[:r], Vt[:r, :].conj().T
    invSr = np.diag(1.0 / Sr)
    Atil = Ur.conj().T @ X2 @ Vr @ invSr                     # r x r reduced operator
    lam, Wv = np.linalg.eig(Atil)

    Phi = X2 @ Vr @ invSr @ Wv                               # exact DMD modes (d x r)
    b, *_ = np.linalg.lstsq(Phi, X1[:, 0].astype(complex), rcond=None)   # amplitudes
    energy = np.abs(b) * np.linalg.norm(Phi, axis=0)         # per-mode carried energy

    A_full = Ur @ Atil @ Ur.conj().T
    resid = float(np.linalg.norm(X2 - A_full @ X1) / (np.linalg.norm(X2) + 1e-12))

    freqs = np.abs(np.angle(lam)) * fs / (2 * np.pi)
    return Spectrum(freqs=freqs, rho=np.abs(lam), energy=energy, lam=lam, resid=resid)


# --------------------------------------------------------------------------- #
#  Health indicators read off one spectrum (energy-weighted -- the fix)
# --------------------------------------------------------------------------- #
def hf_coherent_energy(sp: Spectrum, f_min: float = 1000.0) -> float:
    """headline indicator: PERSISTENT high-frequency energy = sum over HF modes of
       (carried energy x persistence). A developing fault re-excites a structural
       resonance into a sustained, coherent mode -> a high-energy, high-rho HF mode
       appears and this climbs. Healthy HF is incoherent noise: spread thin, damped."""
    m = sp.freqs >= f_min
    return float((sp.energy[m] * sp.rho[m]).sum()) if np.any(m) else 0.0


def dominant_hf_mode(sp: Spectrum, f_min: float = 1000.0):
    """the interpretable readout: (frequency, persistence) of the strongest HF mode.
       This is the 'which fault' a bare amplitude indicator cannot give you."""
    m = sp.freqs >= f_min
    if not np.any(m):
        return 0.0, 0.0
    i = np.argmax(sp.energy[m] * sp.rho[m])
    return float(sp.freqs[m][i]), float(sp.rho[m][i])


# --------------------------------------------------------------------------- #
#  The cheap simple baselines we must beat (or honestly fail to beat)
# --------------------------------------------------------------------------- #
def rms(x):     x = np.asarray(x, float); return float(np.sqrt(np.mean(x * x)))
def crest(x):
    x = np.asarray(x, float); r = rms(x)
    return float(np.max(np.abs(x)) / (r + 1e-12))
def kurtosis(x):
    """Fisher kurtosis (0 for Gaussian). The classic early bearing-fault tell:
       impacts make the signal spiky -> kurtosis climbs. Strong, simple rival."""
    x = np.asarray(x, float); x = x - x.mean(); v = x.var()
    return float(np.mean(x ** 4) / (v * v + 1e-18) - 3.0) if v > 1e-18 else 0.0


# --------------------------------------------------------------------------- #
#  A streaming monitor: self-calibrates on the first healthy records, then alarms
# --------------------------------------------------------------------------- #
@dataclass
class Alarm:
    record: int
    indicator: str
    value: float
    threshold: float
    freq_hz: float        # the offending mode's frequency, when it is a DMD alarm
    rho: float


@dataclass
class StreamMonitor:
    fs: float
    delay: int = 64
    rank: int = 20
    f_min: float = 1000.0           # HF band for the DMD fault indicator
    hp_cut: float = 800.0           # standard shaft-removal high-pass
    baseline_records: int = 10      # known-good warm-up (self-calibration)
    n_sigma: float = 5.0            # alarm = baseline mean + n_sigma * std
    persistence: int = 2            # require this many consecutive crossings

    # state
    _baseline: list = field(default_factory=list)
    _thr: dict = field(default_factory=dict)
    _streak: dict = field(default_factory=dict)
    n_seen: int = 0
    history: dict = field(default_factory=lambda: {"hf_energy": [], "resid": [],
                                                   "kurtosis": [], "rms": [],
                                                   "crest": [], "peak_f": [], "peak_rho": []})

    _IND = ("hf_energy", "kurtosis", "crest", "resid")     # indicators that alarm

    def _features(self, window):
        xf = highpass(window, self.fs, self.hp_cut)        # standard front end
        sp = dmd_spectrum(xf, self.fs, self.delay, self.rank, normalise=True)
        pf, pr = dominant_hf_mode(sp, self.f_min)
        return {
            "hf_energy": hf_coherent_energy(sp, self.f_min),
            "resid":     sp.resid,
            "kurtosis":  kurtosis(xf),     # baselines on the SAME filtered signal
            "rms":       rms(xf),
            "crest":     crest(xf),
            "peak_f":    pf,
            "peak_rho":  pr,
        }

    def push(self, window) -> list[Alarm]:
        """feed one record; returns any alarms raised this record."""
        f = self._features(window)
        for k in self.history:
            self.history[k].append(f[k])
        rec = self.n_seen
        self.n_seen += 1
        alarms = []

        if rec < self.baseline_records:
            self._baseline.append(f)
            if rec == self.baseline_records - 1:
                for k in self._IND:
                    vals = np.array([b[k] for b in self._baseline])
                    self._thr[k] = float(vals.mean() + self.n_sigma * (vals.std() + 1e-9))
                    self._streak[k] = 0
            return alarms

        for k in self._IND:
            if f[k] > self._thr[k]:
                self._streak[k] += 1
                if self._streak[k] == self.persistence:
                    alarms.append(Alarm(record=rec, indicator=k, value=f[k],
                                        threshold=self._thr[k],
                                        freq_hz=f["peak_f"], rho=f["peak_rho"]))
            else:
                self._streak[k] = 0
        return alarms


# --------------------------------------------------------------------------- #
#  The seam to a "normal" model: turn an alarm into a triage prompt / summary.
#  The instrument does the cheap continuous watching; a normal model only ever
#  sees a compact, interpretable EVENT -- never the raw stream.
# --------------------------------------------------------------------------- #
def summarize_alarm(a: Alarm, fs: float) -> str:
    """deterministic, runnable triage line (no LLM needed)."""
    if a.indicator in ("hf_energy",):
        return (f"[record {a.record}] DMD: a high-frequency mode near "
                f"{a.freq_hz:.0f} Hz has become sustained (persistence rho={a.rho:.3f}, "
                f"baseline thr {a.threshold:.3f}). A structural resonance is being "
                f"re-excited periodically -> incipient impact-type fault. "
                f"Inspect the band around {a.freq_hz:.0f} Hz and the defect-frequency "
                f"harmonics below it.")
    if a.indicator == "kurtosis":
        return (f"[record {a.record}] kurtosis={a.value:.1f} (thr {a.threshold:.1f}): "
                f"the signal has turned impulsive -> impacts present.")
    if a.indicator == "resid":
        return (f"[record {a.record}] DMD one-step residual={a.value:.3f}: the stream "
                f"became less linearly-predictable (non-stationary surprise).")
    return (f"[record {a.record}] {a.indicator}={a.value:.3f} crossed {a.threshold:.3f}.")


def triage_prompt(alarms: list[Alarm], fs: float, machine: str = "bearing") -> str:
    """format an event packet for a downstream model to describe / prioritise.
       This is the coupling to 'normal' AI: it reads EVENTS, not the waveform."""
    lines = "\n".join(summarize_alarm(a, fs) for a in alarms)
    return (f"You are a maintenance triage assistant. A trainless vibration monitor "
            f"on a {machine} raised these events (frequencies in Hz, rho is modal "
            f"persistence in [0,1]):\n{lines}\n\n"
            f"State the single most likely fault, its urgency (watch / plan / "
            f"act now), and the one measurement to confirm it. Be terse.")

# To actually call a model here you would POST triage_prompt(...) to your LLM of
# choice; left as a string seam so this file stays trainless and runnable offline.
