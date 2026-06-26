"""
loaders.py — drop the monitor onto the real datasets
====================================================
The build sandbox could not download these (network-restricted), so they are
unvalidated here; the loaders are written to the published formats so that on a
machine with the data, `StreamMonitor` runs unchanged.

  CWRU  (Case Western Reserve bearing data) — MATLAB .mat per condition, keys
        like 'X097_DE_time' (drive-end) / '..._FE_time' (fan-end), fs 12k or 48k.
        https://engineering.case.edu/bearingdatacenter
  IMS   (NASA/IMS run-to-failure) — directories of ASCII files, each file one
        ~1 s snapshot every ~10 min, columns = channels, fs 20 kHz. The canonical
        "fault appears before it breaks" dataset.
  MIMII (industrial machine sound) — 16 kHz WAV, normal/ vs abnormal/ folders.
        Use channel 0; the monitor treats it as a 1-D stream.

Each loader yields (record_array, fs) so you can:
    mon = StreamMonitor(fs=fs, ...)
    for rec, fs in load_ims(dir): mon.push(rec)

PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, June 2026.
Do not hype. Do not lie. Just show.
"""
from __future__ import annotations
import os, glob
import numpy as np


def load_cwru(mat_path, channel="DE", fs=12000.0, window=4096, hop=None):
    """yield windows from one CWRU .mat file. channel in {'DE','FE','BA'}."""
    from scipy.io import loadmat
    m = loadmat(mat_path)
    key = next((k for k in m if k.endswith(f"_{channel}_time")), None)
    if key is None:
        raise KeyError(f"no *_{channel}_time key in {mat_path}; keys={list(m)}")
    x = np.asarray(m[key]).ravel().astype(float)
    hop = hop or window
    for i in range(0, len(x) - window + 1, hop):
        yield x[i:i + window], fs


def load_ims(run_dir, channel=0, fs=20000.0, max_files=None):
    """yield one window per IMS snapshot file (each file ~1 s of vibration).
       record index over files == machine age == the run-to-failure axis."""
    files = sorted(glob.glob(os.path.join(run_dir, "*")))
    files = [f for f in files if os.path.isfile(f)]
    if max_files:
        files = files[:max_files]
    for f in files:
        data = np.loadtxt(f)
        x = data[:, channel] if data.ndim == 2 else data
        yield x.astype(float), fs


def load_mimii(wav_path, channel=0):
    """one WAV -> (signal, fs). MIMII is 16 kHz, 8-channel; channel 0 by default."""
    from scipy.io import wavfile
    fs, data = wavfile.read(wav_path)
    x = data[:, channel] if data.ndim == 2 else data
    x = x.astype(float)
    if np.issubdtype(data.dtype, np.integer):
        x /= np.iinfo(data.dtype).max
    return x, float(fs)


# bearing defect frequencies from geometry + shaft speed (Hz) — what to tune forks to
def defect_freqs(shaft_hz, n_balls, ball_dia, pitch_dia, contact_angle_deg=0.0):
    import math
    r = (ball_dia / pitch_dia) * math.cos(math.radians(contact_angle_deg))
    return {
        "BPFO": n_balls / 2 * shaft_hz * (1 - r),
        "BPFI": n_balls / 2 * shaft_hz * (1 + r),
        "BSF":  pitch_dia / (2 * ball_dia) * shaft_hz * (1 - r * r),
        "FTF":  shaft_hz / 2 * (1 - r),
    }


if __name__ == "__main__":
    # CWRU 6205-2RS SKF drive-end bearing geometry, ~1797 rpm
    print(defect_freqs(shaft_hz=29.95, n_balls=9, ball_dia=0.3126, pitch_dia=1.537))
