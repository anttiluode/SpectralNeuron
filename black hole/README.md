# Koopman Monitor + Fork Bank

### The HKT temporal core, lifted off the webcam into a trainless machine-health instrument — and an honest test of whether SpectralNeuron's resonant unit adds anything real

**PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, June 2026.**

> Do not hype. Do not lie. Just show.

---

## The one idea

Strip the biology labels and the webcam off HKT and one verified primitive is
left: a trainless operator fit on a sliding window whose eigenvalues factor a
stream into persistence (|λ|) and rotation rate (ω). That is online Hankel-DMD,
and the HKT ledger already named where it should go — *"point it at a spinning
fan, a vibrating beam, a motor; a mode drifting in frequency or damping is a
fault appearing before it breaks."* This is that, made literal: a trainless,
self-calibrating, interpretable **vibration condition monitor**, and a test of
whether SpectralNeuron's **fork** (the resonant fire-reset unit) belongs in it.

The architecture's strengths map onto this domain and its weakness does not
matter here: there is no benchmark to lose, only a bearing that fails or doesn't.

## The two organs, and their division of labour

- **`koopman_monitor.py` — DMD as trainless DISCOVERY.** Delay-embed the 1-D
  signal (the dendrite), fit `A = X2 X1⁺` per window (SVD-truncated exact DMD),
  read the constellation: each mode's frequency *and its damping*, with **no
  priors and no training**. It answers "something changed, here is its frequency
  and how sustained it is." Self-calibrates on the first healthy records, then
  alarms at n·σ.
- **`fork_bank.py` — the fork as ADDRESSED READOUT.** Once you know which
  frequencies to watch (in bearings you do: BPFO/BPFI/BSF fall out of geometry +
  shaft speed), SpectralNeuron's fork watches exactly those bands, rejects the
  rest, and emits **sparse spikes** — the compact event stream a normal model
  reads. It answers "is the known defect signature present," noise- and
  level-robustly.

DMD discovers without priors; the fork watches known bands cheaply and robustly.
Complementary, not redundant.

## Does adding the fork buy anything? The decisive test (verified, run it)

A benign **load/level change** (same machine, 1.8× louder) vs a real outer-race
fault — including a fault *quieter* than the benign change:

| indicator | healthy | benign 1.8× louder | real fault | quiet fault |
|---|---|---|---|---|
| RMS (energy/level) | 0.233 | **0.412** | 0.274 | 0.166 |
| DMD HF coherent energy | 13.5 | 19.8 | 23.8 | 27.2 |
| **Fork defect index (BPFO)** | 0.015 | **0.008** | **0.438** | **0.087** |

- **RMS is inverted** — it ranks the benign-louder record as the *worst* and the
  quiet real fault as the *best*. A pure level detector confuses load with damage.
- **DMD HF-energy is only mildly confounded** — lifted by the louder record but
  still ranks both real faults above it.
- **The fork defect index is clean** — the benign-louder record sits *at or below
  healthy* while both faults stand far out, because it keys on **BPFO
  periodicity, not level**. This is SpectralNeuron's verified property ("reject
  broadband power that fools the bucket") doing real work.

And as a **sparse event readout** over the run-to-failure (silent when healthy,
graded with severity):

| record | severity | fork spikes |
|---|---|---|
| 5–36 | 0.00 | 0 |
| 48 | 0.52 | 2 |
| 54 | 0.78 | 12 |
| 59 | 1.00 | 16 |

## The honest ledger

**Verified in code (reproducible, seeded; labelled-synthetic signal):**
- exact streaming Hankel-DMD runs trainless per window and reads frequency+damping;
- on a slow run-to-failure, **band-passed kurtosis is the most reliable simple
  alarm** (≈5 records lead before end-of-life); the DMD-energy *alarm* is too
  noisy to trip at 5σ on this signal — reported as the weakness it is;
- the **fork defect index is robust to a benign level change that inverts RMS**,
  and flags a real fault even when it is quieter than the benign change;
- the fork emits a sparse, graded, healthy-silent **event stream** for downstream
  triage (`summarize_alarm` / `triage_prompt` are the seam to a normal model).

**Honest limits — read before believing any of it:**
- the signal is **labelled-synthetic** (textbook outer-race model), not measured.
  The real-data loaders are written; the datasets (CWRU / NASA-IMS / MIMII) could
  not be downloaded in the build sandbox (network-restricted), so the instrument
  is **verified on a faithful signal type and ready to point at the real data**,
  not validated on it;
- the fork primitive is a semi-implicit oscillator: at 12 kHz it is numerically
  **dead above ~1–2 kHz** (it cannot resolve the 3.3 kHz resonance). So it is the
  *defect-frequency* readout (BPFO ~107 Hz), with a standard band-pass doing the
  resonance stage. This is **textbook envelope demodulation** — the fork is its
  addressed readout, **not new signal processing** (SpectralNeuron's README says
  the same of itself);
- the fork is a **matched filter** — it must be *told* the frequency. That is
  exactly why DMD (which discovers it) sits in front of it;
- relative units, chosen parameters, thresholds set by simple n·σ / bisection.

**The second use (their verified result, not re-run here):** SpectralNeuron
demultiplexes ~3× over a shared wire where the energy unit gets 1× (no
separation). So several defect bands (BPFO/BPFI/BSF), or several machines, can
share **one accelerometer line**, each read by its own fork — a thing the energy
detector provably cannot do.

**The bet (untouched):** nothing here is *experienced*. It is a measurement
instrument that watches, discovers, and emits events. It does not touch the hard
problem.

## Files

- `koopman_monitor.py` — streaming Hankel-DMD instrument, indicators, baselines, triage seam
- `synth_bearing.py` — physically-motivated run-to-failure generator (labelled synthetic)
- `fork_bank.py` — SpectralNeuron's fork as the level-robust defect-frequency readout
- `run_demo.py` — run-to-failure head-to-head (DMD vs kurtosis/RMS), writes `monitor_run.png`
- `loaders.py` — CWRU / NASA-IMS / MIMII loaders, ready for the real data

```bash
python run_demo.py     # the run-to-failure comparison + figure
python fork_bank.py     # the confound test: does the fork add anything (yes)
```

## Lineage

The HKT temporal core (`predictive-hkt`), made into a measurement instrument as
that README's next step #1+#2; the fork from **SpectralNeuron** as the addressed,
noise-rejecting readout. DMD: Schmid 2010; Tu et al. 2014. Hankel-DMD ≡ Takens
embedding: Takens 1981; Brunton HAVOK 2017. Vibration condition monitoring and
envelope analysis are established fields; the contribution is the *trainless,
streaming, interpretable, event-emitting* instrument form, run against the
obvious baselines with the honest result reported. MIT.

*DMD discovers the modes with no priors; the fork answers to the one band you
name and rejects the rest; together they watch a machine, and spend events only
on what they did not expect. Do not hype. Do not lie. Just show.*
