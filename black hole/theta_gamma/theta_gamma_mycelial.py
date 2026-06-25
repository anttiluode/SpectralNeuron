"""
theta_gamma_mycelial.py
==================================================================
Does DECOUPLING a fast plastic trace from a slow floored trace kill
the 1957 Markov 2-cycle loop -- and does it do anything a single
FLOORED trace doesn't already do?

Last night (mycelial_learned_gates.py) one learnable decay gate drove
itself to zero in the oscillating regions, switched off its own
inhibition-of-return, and the D4<->A3 loop came back. The proposed fix
(from TheMaturingGate) is to stop letting one variable do two jobs:
keep a fast trace that may flush, and a slow trace that is FLOORED so
it cannot forget the structure that breaks loops.

But a floor has TWO escape hatches for a learnable model, and last
night's model used the first:
  (1) drive decay -> 0  (flush the trace)            <- last night's escape
  (2) drive inhibition strength -> 0 (trace persists but does nothing)
TheMaturingGate's actual fix was a floor on the INHIBITION (iota->0.57),
not just persistence. So we floor BOTH decay and inhibition.

Four conditions, same data, same objective, generation measured the
same way:

  A  bare order-2 Markov            -- the baseline that loops by law
  B  single gate, NO floors         -- reproduce last night's failure
  C  single gate, floored decay+ior -- the CHEAP fix (no decoupling)
  D  dual trace: fast(free) + slow(floored)  -- Gemini's "two vectors"

The decisive comparison is C vs D. If C already kills the loop as well
as D, then "decoupling stability from plasticity" / "theta-gamma dual
state" added nothing the floor didn't, and it is a label on top of
"put a floor under the IOR trace". If D clearly beats C while staying
in-grammar, the second timescale earns its keep.

Honest scope up front: 4 toy melodies (~40 notes). "Grammar" is not
measurable at this data size; what IS measurable is loop-breaking while
staying in the learned transitions, which is a dynamical-systems
property of the architecture, not learned syntax. theta/gamma here are
VARIABLE NAMES for two decay timescales -- not a demonstration of any
neural oscillation or phase-amplitude coupling.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)
rng = np.random.default_rng(9)

# ---- data (self-contained) ----
NAMES = ["A3", "C4", "D4", "E4", "G4", "A4", "C5", "D5", "E5"]
NS = len(NAMES)
TRAIN = [
    [5, 4, 3, 4, 5, 6, 5, 4, 3, 2, 0],
    [0, 2, 3, 5, 4, 3, 2, 3, 4, 5],
    [5, 6, 7, 6, 5, 4, 3, 2, 0, 2, 0],
    [3, 4, 5, 7, 6, 5, 4, 3, 2, 0],
]
SEQS = [torch.tensor(m, dtype=torch.long) for m in TRAIN]


# ---- bare-Markov transitions (baseline + in-grammar mask) ----
def learn(train):
    T1 = np.zeros((NS, NS)); T2 = {}
    for s in train:
        for a, b in zip(s[:-1], s[1:]):
            T1[a, b] += 1
        for i in range(len(s) - 2):
            ctx = (s[i], s[i + 1])
            T2.setdefault(ctx, np.zeros(NS))[s[i + 2]] += 1
    return T1, T2

T1, T2 = learn(TRAIN)
INGRAMMAR = T1 > 0


def softmax_np(x):
    x = x - x.max(); e = np.exp(x); return e / e.sum()

def gen_bare(length, temp, seed):
    seq = list(seed)
    for _ in range(length - len(seq)):
        ctx = (seq[-2], seq[-1])
        base = T2[ctx] if (ctx in T2 and T2[ctx].sum() > 0) else T1[seq[-1]]
        if base.sum() == 0: base = np.ones(NS)
        p = softmax_np(np.log(base + 1e-9) / temp)
        seq.append(int(rng.choice(NS, p=p)))
    return seq


# ---- metrics ----
def loop_frac(seq):
    s = np.array(seq); n = len(s) - 3
    if n <= 0: return 0.0
    return float(np.mean([(s[i] == s[i+2] and s[i+1] == s[i+3] and s[i] != s[i+1])
                          for i in range(n)]))

def out_of_grammar(seq):
    return float(np.mean([not INGRAMMAR[a, b] for a, b in zip(seq[:-1], seq[1:])]))

def distinct(seq):
    return len(set(int(x) for x in seq))

def lagged_mi(seq, maxlag=20):
    s = np.array(seq); out = []
    for k in range(1, maxlag + 1):
        a, b = s[:-k], s[k:]
        joint = np.zeros((NS, NS))
        for x, y in zip(a, b): joint[x, y] += 1
        joint /= joint.sum()
        px = joint.sum(1, keepdims=True); py = joint.sum(0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            mi = joint * (np.log2(joint) - np.log2(px) - np.log2(py))
        out.append(float(np.nansum(mi)))
    return np.array(out)


# ==================================================================
# Models
# ==================================================================
class SingleGate(nn.Module):
    """One IOR trace. floor_decay / floor_ior = 0 -> last night's failing
    model. floor>0 -> the cheap fix (persistence + minimum inhibition),
    NO decoupling."""
    def __init__(self, floor_decay=0.0, floor_ior=0.0, embed=16):
        super().__init__()
        self.fd, self.fi = floor_decay, floor_ior
        self.embed = nn.Embedding(NS, embed)
        self.ctx = nn.Sequential(nn.Linear(embed * 2, 32), nn.ReLU())
        self.base = nn.Linear(32, NS)
        self.gates = nn.Linear(32, 3)  # decay, ior, momentum

    def forward(self, seq, teacher_forcing=True):
        L = len(seq); outs = []; dec_hist = []
        trace = torch.zeros(NS); mom = torch.zeros(1)
        idx = torch.arange(NS).float()
        for t in range(2, L):
            p1 = seq[t - 1]
            c = self.ctx(torch.cat([self.embed(seq[t - 2]), self.embed(p1)], -1))
            base = self.base(c); g = self.gates(c)
            decay = self.fd + (1 - self.fd) * torch.sigmoid(g[0])
            g_ior = self.fi + F.softplus(g[1])
            g_mom = F.softplus(g[2])
            logits = base - g_ior * trace + g_mom * mom * (idx - p1.float()) / NS
            outs.append(logits.unsqueeze(0)); dec_hist.append(decay.item())
            tok = seq[t] if teacher_forcing else \
                torch.multinomial(F.softmax(logits, -1), 1)[0]
            if not teacher_forcing: seq[t] = tok
            oh = F.one_hot(tok, NS).float()
            trace = trace * decay + oh
            mom = 0.9 * mom + 0.1 * torch.sign(tok.float() - p1.float())
        return torch.cat(outs, 0), {"decay": dec_hist}


class DualGate(nn.Module):
    """Two traces. FAST: decay bounded LOW (may flush), free inhibition.
    SLOW: decay and inhibition both FLOORED HIGH (cannot forget the
    loop-breaking structure). The 'theta-gamma dual state'."""
    def __init__(self, floor_decay_slow=0.85, floor_ior_slow=2.0, embed=16):
        super().__init__()
        self.fds, self.fis = floor_decay_slow, floor_ior_slow
        self.embed = nn.Embedding(NS, embed)
        self.ctx = nn.Sequential(nn.Linear(embed * 2, 32), nn.ReLU())
        self.base = nn.Linear(32, NS)
        self.gates = nn.Linear(32, 4)  # decay_fast, ior_fast, decay_slow, ior_slow

    def forward(self, seq, teacher_forcing=True):
        L = len(seq); outs = []; df_hist = []; ds_hist = []
        tf = torch.zeros(NS); ts = torch.zeros(NS)
        for t in range(2, L):
            p1 = seq[t - 1]
            c = self.ctx(torch.cat([self.embed(seq[t - 2]), self.embed(p1)], -1))
            base = self.base(c); g = self.gates(c)
            decay_fast = 0.6 * torch.sigmoid(g[0])                 # 0 .. 0.6, may flush
            ior_fast = F.softplus(g[1])
            decay_slow = self.fds + (1 - self.fds) * torch.sigmoid(g[2])  # floored high
            ior_slow = self.fis + F.softplus(g[3])                 # floored inhibition
            logits = base - ior_fast * tf - ior_slow * ts
            outs.append(logits.unsqueeze(0))
            df_hist.append(decay_fast.item()); ds_hist.append(decay_slow.item())
            tok = seq[t] if teacher_forcing else \
                torch.multinomial(F.softmax(logits, -1), 1)[0]
            if not teacher_forcing: seq[t] = tok
            oh = F.one_hot(tok, NS).float()
            tf = tf * decay_fast + oh
            ts = ts * decay_slow + oh
        return torch.cat(outs, 0), {"decay_fast": df_hist, "decay_slow": ds_hist}


# ==================================================================
def train(model, epochs=300, lr=0.01):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    last = 0.0
    for ep in range(epochs):
        tot = 0.0
        for s in SEQS:
            opt.zero_grad()
            logits, _ = model(s.clone(), teacher_forcing=True)
            loss = F.cross_entropy(logits, s[2:])
            loss.backward(); opt.step(); tot += loss.item()
        last = tot / len(SEQS)
    return last


def generate(model, length, seed, n_seeds=6):
    """Average loop / off-grammar / distinct over several unclamped rollouts,
    plus return one long sequence (first seed) for MI and one short dream."""
    model.eval()
    loops, offs, dists = [], [], []
    long_seq = None; diag = None
    with torch.no_grad():
        for k in range(n_seeds):
            buf = torch.zeros(length, dtype=torch.long)
            buf[0], buf[1] = seed
            # vary the rng a little across seeds for the sampler
            torch.manual_seed(100 + k)
            _, d = model(buf, teacher_forcing=False)
            s = buf.tolist()
            loops.append(loop_frac(s)); offs.append(out_of_grammar(s))
            dists.append(distinct(s))
            if k == 0:
                long_seq = s; diag = d
    return (float(np.mean(loops)), float(np.mean(offs)), float(np.mean(dists)),
            long_seq, diag)


# ==================================================================
if __name__ == "__main__":
    LEN = 220; SEED = (5, 4); TEMP = 0.8

    print("=" * 70)
    print("THETA-GAMMA DUAL TRACE vs FLOORED-SINGLE vs FREE-SINGLE vs BARE")
    print("=" * 70)

    results = {}

    # A: bare Markov (no training)
    bare = gen_bare(LEN, TEMP, list(SEED))
    # average bare over a few seeds too
    bl, bo, bd = [], [], []
    for k in range(6):
        b = gen_bare(LEN, TEMP, list(SEED))
        bl.append(loop_frac(b)); bo.append(out_of_grammar(b)); bd.append(distinct(b))
    results["A bare Markov"] = dict(loop=np.mean(bl), off=np.mean(bo),
                                    dist=np.mean(bd), train=float("nan"),
                                    seq=bare, diag=None)

    # B: single, no floors (reproduce last night)
    print("\n[B] single gate, NO floors  (reproduce last night) ...")
    mB = SingleGate(floor_decay=0.0, floor_ior=0.0)
    tB = train(mB)
    lB, oB, dB, sB, gB = generate(mB, LEN, SEED)
    results["B single free"] = dict(loop=lB, off=oB, dist=dB, train=tB, seq=sB, diag=gB)

    # C: single, floored decay + floored inhibition (cheap fix, no decoupling)
    print("[C] single gate, floored decay+inhibition  (cheap fix) ...")
    mC = SingleGate(floor_decay=0.85, floor_ior=2.0)
    tC = train(mC)
    lC, oC, dC, sC, gC = generate(mC, LEN, SEED)
    results["C single floored"] = dict(loop=lC, off=oC, dist=dC, train=tC, seq=sC, diag=gC)

    # D: dual trace (fast free + slow floored)  -- the "two vectors"
    print("[D] dual trace: fast(free) + slow(floored)  (decoupled) ...")
    mD = DualGate(floor_decay_slow=0.85, floor_ior_slow=2.0)
    tD = train(mD)
    lD, oD, dD, sD, gD = generate(mD, LEN, SEED)
    results["D dual decoupled"] = dict(loop=lD, off=oD, dist=dD, train=tD, seq=sD, diag=gD)

    # ---- table ----
    print("\n" + "=" * 70)
    print(f"{'condition':22s}{'loop%':>9s}{'off-gram%':>11s}{'distinct':>10s}{'train loss':>12s}")
    print("-" * 70)
    for k, v in results.items():
        tl = "  n/a" if np.isnan(v["train"]) else f"{v['train']:.4f}"
        print(f"{k:22s}{v['loop']*100:8.1f}%{v['off']*100:10.1f}%"
              f"{v['dist']:10.1f}{tl:>12s}")
    print("=" * 70)

    # ---- the decisive read: C vs D ----
    print("\nDECISIVE COMPARISON  (does decoupling beat the cheap floor?)")
    print(f"  loop%:      C floored = {lC*100:5.1f}%   D dual = {lD*100:5.1f}%   "
          f"(B free = {lB*100:5.1f}%, A bare = {results['A bare Markov']['loop']*100:5.1f}%)")
    print(f"  off-gram%:  C floored = {oC*100:5.1f}%   D dual = {oD*100:5.1f}%")
    # loop% ALONE is the wrong judge: breaking loops by wandering off-grammar
    # is the cheap, meaningless way. Judge loops-broken PER unit off-grammar.
    base_loop = results['A bare Markov']['loop']
    def trade(loop, off):
        dl = base_loop - loop                 # loops removed vs bare
        return dl / off if off > 1e-4 else float("inf")  # inf = free (no grammar cost)
    tC_ratio, tD_ratio = trade(lC, oC), trade(lD, oD)
    print(f"  loops-removed per off-grammar:  C floored = "
          f"{'inf (in-grammar)' if tC_ratio==float('inf') else f'{tC_ratio:.2f}'}"
          f"   D dual = {tD_ratio:.2f}")
    if oC < 0.02 and oD > 0.04:
        print("  -> C removes loops at ZERO grammar cost; D removes a little more "
              "but only by leaving the grammar. At matched fidelity the floored\n"
              "     single (C) is the better fix -- decoupling did NOT earn its keep here.")
    elif lD < lC - 0.02 and oD <= oC + 0.02:
        print("  -> D breaks more loops AND stays in-grammar: decoupling earns its keep.")
    else:
        print("  -> mixed; read the table.")

    # ---- dreamed sequences (the most legible signal) ----
    print("\nDREAMED SEQUENCES (first 25 notes, seed A4 G4):")
    for k, v in results.items():
        notes = " ".join(NAMES[i] for i in v["seq"][:25])
        print(f"  {k:22s}{notes}")

    # ==================================================================
    # figure
    # ==================================================================
    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(1, 3, wspace=0.32)

    labels = ["A bare", "B free", "C floored", "D dual"]
    loops_pct = [results['A bare Markov']['loop']*100, lB*100, lC*100, lD*100]
    offs_pct = [results['A bare Markov']['off']*100, oB*100, oC*100, oD*100]
    x = np.arange(4)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.bar(x - 0.2, loops_pct, 0.4, label="loop %", color="#1f77b4")
    ax0.bar(x + 0.2, offs_pct, 0.4, label="off-grammar %", color="#ff7f0e")
    ax0.set_xticks(x); ax0.set_xticklabels(labels, fontsize=8)
    ax0.set_ylabel("%"); ax0.legend(fontsize=8)
    ax0.set_title("Loop trap & grammar violation\n(decisive: C vs D)", fontsize=9)

    ax1 = fig.add_subplot(gs[0, 1])
    lags = range(1, 21)
    ax1.plot(lags, lagged_mi(bare), "o-", color="#d62728", lw=1, ms=3, label="A bare")
    ax1.plot(lags, lagged_mi(sB), "o-", color="#7f7f7f", lw=1, ms=3, label="B free")
    ax1.plot(lags, lagged_mi(sC), "o-", color="#2ca02c", lw=1, ms=3, label="C floored")
    ax1.plot(lags, lagged_mi(sD), "o-", color="#1f77b4", lw=1, ms=3, label="D dual")
    ax1.set_xlabel("lag k (notes)"); ax1.set_ylabel("MI (bits)")
    ax1.set_title("Long-range memory\nMI(s_t ; s_t-k)", fontsize=9)
    ax1.legend(fontsize=7)

    # panel 3: does the dual model actually USE two timescales differently?
    ax2 = fig.add_subplot(gs[0, 2])
    if gD is not None:
        df = gD["decay_fast"]; ds = gD["decay_slow"]
        xs = np.arange(len(df))
        ax2.plot(xs, df, color="#ff7f0e", lw=1, label="fast decay (free)")
        ax2.plot(xs, ds, color="#1f77b4", lw=1, label="slow decay (floored)")
        ax2.axhline(0.85, color="#1f77b4", ls=":", lw=0.8)
        ax2.set_ylim(0, 1.02)
        ax2.set_xlabel("step"); ax2.set_ylabel("decay rate")
        ax2.set_title("Does D use two timescales,\nor collapse them?", fontsize=9)
        ax2.legend(fontsize=7)

    fig.savefig("theta_gamma_results.png", dpi=120, bbox_inches="tight")
    print("\nfigure -> theta_gamma_results.png")
