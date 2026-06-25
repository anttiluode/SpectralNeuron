# Spectral Address: does a resonant neuron beat an energy neuron?

A small, falsifiable test of one idea pulled out of a long theory thread:
**if a neuron fires on resonance instead of raw integrated charge, does it
gain a "spectral address" — does it answer to *which* frequency is present
rather than *how much* signal — and can many such units share one wire
without their messages crossing?**

Two single units, same skeleton (integrate -> threshold -> fire -> **reset**),
differing only in *what* they integrate:

- **Bucket** (standard leaky integrate-and-fire, energy form): integrates the
  rectified magnitude of its input. A power detector, blind to frequency.
- **Fork** (resonant): a damped driven oscillator that rings up only when the
  input carries energy near its own natural frequency, fires when the ring
  crosses threshold, then empties.

The arrow of time in both comes from the **reset** — fire, then empty — not
from any carved structural asymmetry. No axon-initial-segment "spillway" is
needed to make time run one way; the one-way valve is the discharge itself.
That was the specific claim being checked, and it holds: both units are
perfectly causal with nothing but an ordered fill-fire-reset loop.

## Run

```
python3 experiment.py
```

Produces a numbers table and `spectral_address_results.png`.

## What it measured

### Part A — matched-power discrimination (the real result)

Both units calibrated to fire ~20 times on the *same* reference: a loud,
on-band tone. Then six stimuli:

| stimulus | fork | bucket |
|---|---|---|
| on-band quiet (40 Hz, A=0.2) | 0 | 0 |
| off-band quiet (120 Hz, A=0.2) | 0 | 0 |
| **on-band LOUD (40 Hz, A=1.0)** | **20** | **21** |
| **off-band LOUD (120 Hz, A=1.0)** | **0** | **20** |
| broadband noise (σ=1.0) | 0 | 33 |
| quiet on-band in noise (0.2 + n0.8) | 0 | 20 |

The headline is the third and fourth rows. **Same amplitude, same power, and
the fork fires 20 vs 0 while the bucket fires 21 vs 20.** At matched power the
fork discriminates frequency and the bucket cannot tell the two tones apart.
And on broadband noise the bucket fires 33 times (all false alarms) while the
fork stays silent. So the "spectral address" is real in the sense that matters:
the fork rejects off-band energy and noise that the power detector is fooled by.

### The correction (what I had wrong)

Two turns ago I — and Gemini — said the fork "fires on frequency match *even
when raw amplitude is low*." **The data says that's false.** Rows 1 and 6: the
fork fires **0** on a quiet on-band tone. A resonator's ring amplitude scales
with drive amplitude, so a quiet in-band signal, judged against a threshold
set by a loud reference, never crosses. The fork is **band-selective, not
loudness-blind.** The accurate claim is narrower than the one we reached for:
*among signals of comparable amplitude it picks the one in its band, and it
rejects off-band power however loud — but it still has an amplitude floor.*
This is exactly the kind of overclaim the whole thread exists to catch, and the
measurement caught it.

### Part B — multiplexing over one wire ("spectral islands")

Three carriers (25 / 45 / 75 Hz) summed onto a single noisy wire, each carrying
its own slowly-varying message. Can each fork recover only its own channel?

Crosstalk (correlation of each unit's firing rate with each message):

```
fork:   mean self-correlation 0.62   mean crosstalk 0.20   ratio 3.1x
bucket: mean self-correlation 0.32   mean crosstalk 0.32   ratio 1.0x
```

The forks demultiplex: each tracks its own message ~3x better than its
neighbours'. The buckets fail completely — every bucket sees total power, so
all three produce the identical trace and correlate equally with everything
(1.0x, no separation). So a shared medium *can* carry several frequency-addressed
channels read out by integrate-fire-reset units, and a power-integrating unit
*cannot* separate them. That is the spectral-islands idea, working.

**But the separation is finite, not magic: 3x, with real leakage (off-diagonal
up to 0.34).** A brain-scale version would need narrower bands (higher Q), wider
frequency spacing, or it pays in crosstalk. The number is the point — not "it
works" or "it doesn't," but *how cleanly*, measured.

## What this is and isn't

- **Is:** a clean demonstration that a resonant fire-reset unit is
  frequency-addressable and an energy-integrating one isn't, with the
  separation quantified (3x) and the failure of the power detector quantified
  (1x). And confirmation that the time-arrow needs only the reset, not a
  structural asymmetry.
- **Isn't:** novel signal processing. The fork is a band-pass filter; the
  multiplexing is frequency-division multiplexing, which radios have done for a
  century. The only thing rebuilt here is FDM out of neuron-flavoured
  primitives.
- **Says nothing about** real brains, theta/gamma, the AIS, holography, or
  binding. It tests one mechanism in isolation: resonance-gating vs
  energy-gating, single unit and three units. The biology stays a hypothesis.

The honest one-line version: *a neuron that fires on resonance has a usable
spectral address — band-selective, noise-rejecting, and demultiplexable about
3x — but it is not amplitude-blind, and none of this is new math; it is FDM
with a reset for a clock.*
