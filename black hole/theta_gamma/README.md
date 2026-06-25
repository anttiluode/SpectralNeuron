# ThetaGamma_Mycelial — does decoupling stability from plasticity kill the Markov loop?

**PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki.**

> Do not hype. Do not lie. Just show.

---

## The question

Last night `mycelial_learned_gates.py` made one decay gate learnable, the optimizer
drove it to the floor in the oscillating regions, switched off its own
inhibition-of-return, and the **D4↔A3 loop came back**. The proposed fix (from
`TheMaturingGate`) was to stop forcing one variable to do two jobs: keep a **fast**
trace that may flush *and* a **slow** trace that is **floored** so it cannot forget
the structure that breaks loops — the "theta/gamma dual state".

The claim under test, in Gemini's words, was that this *"definitively kills the 1957
Markov loop"*. This repo checks it, with one control Gemini's plan left out.

## The control that decides it

"Decoupling kills the loop" only means something if it beats the **cheap** fix:
a *single* trace with a floor, no decoupling at all. And a learnable model has a
second escape hatch besides decay→0 — it can drive the *inhibition strength* to
zero (trace persists but does nothing). `TheMaturingGate`'s real fix was a floor on
the **inhibition** (iota→0.57), so we floor **both** decay and inhibition.

Four conditions, same data, same objective, generation measured the same way:

| | condition | what it is |
|---|---|---|
| **A** | bare order-2 Markov | the baseline that loops by law |
| **B** | single gate, no floors | reproduce last night's failure |
| **C** | single gate, **floored** decay + inhibition | the cheap fix — no decoupling |
| **D** | dual trace: fast (free) + slow (floored) | Gemini's "two vectors" |

The decisive comparison is **C vs D**, and the judge is **not loop% alone** —
breaking loops by wandering off the learned transitions is the trivial, meaningless
way. The honest metric is *loops broken while staying in-grammar*.

## Run

```bash
pip install torch numpy matplotlib
python theta_gamma_mycelial.py
```

## What it found

```
condition                 loop%  off-gram%  distinct  train loss
----------------------------------------------------------------------
A bare Markov             16.6%       0.0%       7.0         n/a
B single free             13.4%       0.1%       7.0      0.0823
C single floored          12.0%       0.0%       7.0      0.1134
D dual decoupled           9.8%       7.7%       8.5      0.2947
```

**1. Nothing "definitively kills" the loop.** Best-to-worst is 16.6 → 9.8% — about a
40% reduction, which is roughly what the *fixed* IOR knob already did in the original
probe (15.4 → 8.4%). The learnable floored versions **recovered** the fixed knob's
win; they did not exceed it.

**2. The decisive C-vs-D comparison goes _against_ decoupling.** D has the lowest
loop% (9.8) — but bought it with **7.7% off-grammar**: it broke loops partly by
*leaving the learned transitions*, which is the cheap way the metric exists to catch.
C broke loops nearly as well (12.0%) at **zero grammar cost**. At matched fidelity the
**floored single trace is the better fix.** Decoupling did not earn its keep here.

**3. The floored single also wins on long-range memory.** In the MI panel, **C
(green) carries the most correlation out to lag ~19** — more than the dual model,
which sits near bare Markov at long lags. D spent its structure wandering off-grammar
instead of building reach. So the extra trace did not buy more memory; it bought less.

**4. The floor costs fit, exactly as predicted — and D pays the most.** Train loss
climbs 0.082 → 0.113 → 0.295. That rising number *is* the plasticity–stability
tradeoff made concrete: a floored gate fits the oscillatory training melodies less
tightly because it refuses to forget. The dual model pays the most fit AND leaves
grammar most — it is the worst-behaved of the three trained models, not the best.

**5. D really does run two timescales (panel 3) — it just doesn't help.** The fast
gate swings the full 0–0.6 range, the slow gate stays pinned near its 0.85 floor. The
architecture is genuinely dual; the *benefit* of being dual is absent. Mechanism real,
payoff not.

## The honest verdict

The week's lesson, one more time, now inside this experiment: **breaking loops is
trivial if you allow off-grammar; doing it while staying in-grammar is the only thing
that counts** — and the simplest floored trace does that best. The "two vectors /
theta-gamma dual state / active inference engine" framing is not supported by these
numbers. The keeper is smaller and real: **a floor under the inhibition trace
(decay + strength) recovers the fixed-knob loop-breaking inside a learnable model,
stays perfectly in-grammar, and carries the most long-range correlation of any
condition.** That is condition **C**, and it needed no decoupling.

A note worth keeping: the script's own first auto-verdict judged on loop% alone and
printed *"the second timescale earns its keep"* — then the off-grammar column
overturned it. Even the automated judge rounded up; the second metric caught it. That
is why the second metric is there.

## Honest limits — read before believing any of it

- **4 toy melodies (~40 notes), 6 seeds, one chosen set of floor values** (slow decay
  0.85, slow inhibition floor 2.0). "Grammar" is not statistically measurable at this
  data size; what is measurable is the *dynamical* property of loop-breaking-in-grammar.
- D's 7.7% off-grammar is partly a consequence of *my* floor choice — a strong slow
  inhibition floor pushes it off-manifold. A gentler floor might bring D's off-grammar
  down. The clean finding is only that **as configured, decoupling did not beat the
  floored single**; it is not a proof that no dual configuration ever could.
- `theta` / `gamma` here are **variable names** for two decay timescales. Nothing here
  demonstrates a neural oscillation, phase–amplitude coupling, or any specific circuit.
  The biology stays an analogy, as in every piece of this line.
- This is FDM-flavoured / SSM-flavoured sequence modelling in neuron dress. The
  mechanism (decaying trace as extra state, inhibition-of-return, floored gate) is
  known. The only thing built here is the controlled four-way comparison that lets the
  decoupling claim fail — and it did.

## The one test that would actually move it

The open question from the night before is still open and this didn't answer it: is
the elevated long-range MI **content** (real grammar — motif return) or **refractory
spacing** (mechanical — revisit interval and its harmonics)? Split the MI by whether it
sits only at the recurrence interval and harmonics, or also at content lags independent
of spacing. That single measurement separates "rhythm engine" from "phrase engine", and
no amount of architecture-shuffling substitutes for it.

---

## Lineage

Built on the Mycelial Cortex / `TheMaturingGate` line. The dual-trace idea and the
theta/gamma framing are Antti Luode's; the four-way controlled comparison and this
document were built with Claude (Opus 4.8). The result corrects the framing it set out
to confirm — which is the point of running it. MIT.
