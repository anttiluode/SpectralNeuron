"""
run_demo.py — point the trainless instrument at a run-to-failure bearing and
              score it honestly against the obvious simple detectors
=============================================================================
Streams a labelled run-to-failure signal through StreamMonitor, which
self-calibrates on the first healthy records and then alarms. Reports, for each
indicator: the record it first confirmed an alarm, the DETECTION DELAY after the
true fault onset, and the LEAD TIME before end-of-life (the "caught it before it
broke" number). Then plots the constellation health indicator next to kurtosis
and RMS, with the onset marked.

The point is the comparison, not a victory: kurtosis is a strong, cheap rival on
impulsive faults. Whatever the numbers say, they print.

PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, June 2026.
Do not hype. Do not lie. Just show.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from synth_bearing import generate_run
from koopman_monitor import StreamMonitor, summarize_alarm, triage_prompt

def first_confirmed(mon_alarms, indicator):
    for a in mon_alarms:
        if a.indicator == indicator:
            return a
    return None

if __name__ == "__main__":
    run = generate_run(n_records=60, onset_frac=0.6, n_samples=4096, seed=0)
    fs, onset, eol = run.fs, run.onset, len(run.records) - 1

    mon = StreamMonitor(fs=fs, delay=64, rank=20, f_min=500.0,
                        baseline_records=10, n_sigma=5.0, persistence=2)
    all_alarms = []
    for rec in run.records:
        all_alarms.extend(mon.push(rec))

    print("=" * 74)
    print("KOOPMAN MONITOR — run-to-failure bearing, trainless detection")
    print("=" * 74)
    print(f"  {len(run.records)} records @ {fs:.0f} Hz | BPFO {run.bpfo:.0f} Hz | "
          f"resonance {run.f_res:.0f} Hz")
    print(f"  true fault onset: record {onset} | end-of-life: record {eol}")
    print(f"  monitor self-calibrated on records 0..{mon.baseline_records-1} (healthy)\n")

    print(f"  {'indicator':<12}{'1st alarm':>10}{'delay after onset':>19}{'lead before EOL':>18}")
    order = ["hf_energy", "hf_energy", "kurtosis", "crest", "resid"]
    names = {"hf_energy": "DMD energy", "hf_energy": "DMD energy", "kurtosis": "kurtosis",
             "crest": "crest", "resid": "DMD resid"}
    for ind in order:
        a = first_confirmed(all_alarms, ind)
        if a is None:
            print(f"  {names[ind]:<12}{'(none)':>10}{'-':>19}{'-':>18}")
        else:
            print(f"  {names[ind]:<12}{a.record:>10}{a.record - onset:>19}{eol - a.record:>18}")
    print("\n  (delay after onset: records AFTER the fault began before it was confirmed;")
    print("   lead before EOL: records of warning BEFORE end-of-life — the useful number.)\n")

    # the triage event packet the instrument would hand a 'normal' model
    dmd_alarms = [a for a in all_alarms if a.indicator == "hf_energy"][:1]
    if dmd_alarms:
        print("  --- event handed to a downstream model (the coupling seam) ---")
        print("  " + summarize_alarm(dmd_alarms[0], fs).replace("\n", "\n  "))
        print()

    # ----------------------------- figure -----------------------------------
    H = mon.history
    x = np.arange(len(run.records))
    fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

    ax[0].plot(x, run.severity, color="#b03030", lw=2)
    ax[0].axvline(onset, color="k", ls="--", lw=1)
    ax[0].set_ylabel("fault severity\n(ground truth)")
    ax[0].set_title("Run-to-failure bearing: trainless Koopman monitor vs simple baselines")
    ax[0].text(onset + 0.4, 0.05, "fault onset", fontsize=8)

    thr_e = mon._thr["hf_energy"]
    ax[1].plot(x, H["hf_energy"], color="#1f6feb", lw=1.8, label="DMD HF persistence (rho)")
    ax[1].axhline(thr_e, color="#1f6feb", ls=":", lw=1, label="self-cal threshold")
    ax[1].axvline(onset, color="k", ls="--", lw=1)
    a = first_confirmed(all_alarms, "hf_energy")
    if a: ax[1].axvline(a.record, color="#1f6feb", lw=1, alpha=0.4)
    ax[1].set_ylabel("DMD HF rho")
    ax[1].legend(fontsize=8, loc="lower right")

    thr_k = mon._thr["kurtosis"]
    ax[2].plot(x, H["kurtosis"], color="#2a9d4a", lw=1.8, label="kurtosis (baseline)")
    ax[2].axhline(thr_k, color="#2a9d4a", ls=":", lw=1)
    ax[2].plot(x, np.array(H["rms"]) / max(H["rms"]) * max(H["kurtosis"]),
               color="#888", lw=1, alpha=0.7, label="RMS (scaled, baseline)")
    ax[2].axvline(onset, color="k", ls="--", lw=1)
    ak = first_confirmed(all_alarms, "kurtosis")
    if ak: ax[2].axvline(ak.record, color="#2a9d4a", lw=1, alpha=0.4)
    ax[2].set_ylabel("kurtosis / RMS")
    ax[2].set_xlabel("record index  (≈ machine age)")
    ax[2].legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig("monitor_run.png", dpi=120)
    print("  figure written: monitor_run.png")
    print("=" * 74)
