"""
spectral_neuron.py
==================================================================
Two single-unit models, deliberately stripped to the bone, so the
ONLY thing that differs between them is *what they integrate*.

  BucketNeuron  (standard "leaky integrate-and-fire", energy form)
      Integrates the rectified magnitude of its input. It is a
      power detector: it fires when enough energy has arrived,
      blind to the spectral content of that energy.

          u' = -u/tau + |I(t)|
          if u > thresh:  spike;  u <- 0

  ForkNeuron  (resonant / "tuning fork")
      A damped, driven harmonic oscillator. It only rings up when
      the input carries energy near its own natural frequency w0.
      It fires when the ring envelope crosses threshold, then
      empties. It is a band-addressed detector.

          x'' + 2*zeta*w0 * x' + w0^2 * x = g * I(t)
          envelope = sqrt(x^2 + (x'/w0)^2)
          if envelope > thresh:  spike;  x <- 0, x' <- 0

Both share the SAME skeleton: integrate something, cross a
threshold, fire, RESET. The reset is the one-way valve -- the
arrow of time falls out of "fire then empty", with no structural
asymmetry carved anywhere. That is the point being tested, not
assumed.

Nothing here is novel as signal processing: the fork is a band-pass
filter, the bucket is an energy integrator. The experiment exists
to MEASURE how far apart they actually fall, and whether the gap is
usable in noise -- not to claim the gap is a discovery.
"""

import numpy as np


def integrate_bucket(I, dt, tau, thresh):
    """Leaky integrator of rectified input. Returns (spike_times_idx, trace)."""
    u = 0.0
    trace = np.empty_like(I)
    spikes = []
    rectified = np.abs(I)
    for t in range(I.size):
        u += dt * (-u / tau + rectified[t])
        trace[t] = u
        if u > thresh:
            spikes.append(t)
            u = 0.0
    return np.array(spikes, dtype=int), trace


def integrate_fork(I, dt, w0, zeta, g, thresh):
    """Damped driven oscillator. Returns (spike_times_idx, envelope_trace)."""
    x = 0.0
    v = 0.0
    env = np.empty_like(I)
    spikes = []
    for t in range(I.size):
        # semi-implicit (symplectic) Euler -- stable for stiff w0
        v += dt * (-2.0 * zeta * w0 * v - (w0 * w0) * x + g * I[t])
        x += dt * v
        e = np.hypot(x, v / w0)          # instantaneous ring amplitude
        env[t] = e
        if e > thresh:
            spikes.append(t)
            x = 0.0
            v = 0.0
    return np.array(spikes, dtype=int), env


def calibrate_threshold(integrate_fn, kwargs, I, dt, target_spikes,
                        lo=1e-6, hi=1e6, iters=40):
    """
    Find the threshold that makes a unit fire ~target_spikes times on the
    reference stimulus I. Spike count is monotone-decreasing in threshold,
    so we bisect. This is how we keep the comparison fair: both units are
    tuned to fire the SAME amount on the one stimulus that suits the fork
    best (loud, on-band), then we watch them diverge on everything else.
    """
    def count(th):
        kw = dict(kwargs)
        kw["thresh"] = th
        sp, _ = integrate_fn(I, dt, **kw)
        return sp.size

    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        c = count(mid)
        if c > target_spikes:
            lo = mid          # too many spikes -> raise threshold
        else:
            hi = mid
    return np.sqrt(lo * hi)
