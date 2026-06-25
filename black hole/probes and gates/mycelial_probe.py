"""
mycelial_probe.py
==================================================================
Does the mycelial drift (a slow priming tide + directional skew)
add structure a plain Markov chain lacks -- or is it the same chain
in fungal clothing?

Same learned transitions for both generators. The ONLY difference is
that the mycelial one carries two slow extra-state variables beyond
the last-k symbols a Markov chain sees:
  - trace[s]: a decaying activation memory (inhibition-of-return:
    discourage recently visited states -> pressure to escape loops)
  - mom: a directional momentum in pitch-index space (the "skew")

The decisive test is a TRADEOFF, because breaking loops is trivial
(just add noise). The question is whether the drift can reduce the
2-cycle loop trap WHILE keeping its transitions inside the learned
grammar. If it can, it adds something the memoryless chain cannot.
If escaping loops costs proportional out-of-grammar transitions,
it is just noise with a story.
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mycelial_music import learn, TRAIN, NS, NAMES

rng = np.random.default_rng(9)
T1, T2, _ = learn(TRAIN)
INGRAMMAR = T1 > 0                      # transitions that exist in training

def softmax(x):
    x = x - x.max(); e = np.exp(x); return e / e.sum()

def gen_bare(length, temp, seed):
    seq = list(seed)
    for _ in range(length - len(seq)):
        ctx = (seq[-2], seq[-1])
        base = T2[ctx] if (ctx in T2 and T2[ctx].sum() > 0) else T1[seq[-1]]
        if base.sum() == 0: base = np.ones(NS)
        p = softmax(np.log(base + 1e-9) / temp)
        seq.append(int(rng.choice(NS, p=p)))
    return seq

def gen_mycelial(length, temp, seed, gamma_ior, gamma_mom=0.3, decay=0.85):
    seq = list(seed)
    trace = np.zeros(NS); mom = 0.0
    idx = np.arange(NS)
    for _ in range(length - len(seq)):
        ctx = (seq[-2], seq[-1])
        base = T2[ctx] if (ctx in T2 and T2[ctx].sum() > 0) else T1[seq[-1]]
        if base.sum() == 0: base = np.ones(NS)
        logits = np.log(base + 1e-9) / temp
        logits -= gamma_ior * trace                       # priming tide (IOR)
        logits += gamma_mom * mom * (idx - seq[-1]) / NS  # directional skew
        p = softmax(logits)
        nxt = int(rng.choice(NS, p=p))
        trace *= decay; trace[nxt] += 1.0                 # update tide
        mom = 0.9 * mom + 0.1 * np.sign(nxt - seq[-1])    # update skew
        seq.append(nxt)
    return seq

def loop_frac(seq):
    """fraction of length-4 windows that are XYXY oscillation (the 1957 trap)"""
    s = np.array(seq); n = len(s) - 3
    if n <= 0: return 0.0
    return np.mean([(s[i]==s[i+2] and s[i+1]==s[i+3] and s[i]!=s[i+1])
                    for i in range(n)])

def out_of_grammar(seq):
    """fraction of transitions never seen in training"""
    return np.mean([not INGRAMMAR[a, b] for a, b in zip(seq[:-1], seq[1:])])

def distinct(seq):
    return len(set(seq))

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
        out.append(np.nansum(mi))
    return np.array(out)

# ---------- run ----------
LEN = 1500; TEMP = 0.8; SEED = [5, 4]
bare = gen_bare(LEN, TEMP, SEED)

print("=" * 66)
print("MYCELIAL DRIFT vs PLAIN MARKOV  (same transitions, same temp)")
print("=" * 66)
print(f"{'generator':28s}{'loop%':>9s}{'off-grammar%':>14s}{'distinct':>10s}")
print("-" * 66)
print(f"{'bare order-2 Markov':28s}{loop_frac(bare)*100:8.1f}%"
      f"{out_of_grammar(bare)*100:13.1f}%{distinct(bare):10d}")

sweep = [0.5, 1.0, 2.0, 4.0]
myc_seqs = {}
for g in sweep:
    m = gen_mycelial(LEN, TEMP, SEED, gamma_ior=g)
    myc_seqs[g] = m
    print(f"{'mycelial IOR=%.1f' % g:28s}{loop_frac(m)*100:8.1f}%"
          f"{out_of_grammar(m)*100:13.1f}%{distinct(m):10d}")
print("=" * 66)

# the decisive read: does loop% drop FASTER than off-grammar% rises?
print("\nTRADEOFF (vs bare):  loops removed  per  unit off-grammar added")
base_loop = loop_frac(bare); base_off = out_of_grammar(bare)
for g in sweep:
    dl = base_loop - loop_frac(myc_seqs[g])
    do = out_of_grammar(myc_seqs[g]) - base_off
    ratio = dl / do if do > 1e-6 else float("inf")
    print(f"  IOR={g:.1f}:  -{dl*100:5.1f}% loops   +{do*100:5.1f}% off-grammar"
          f"   ratio {ratio:6.2f}")

# ---------- figure ----------
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].plot(range(1, 21), lagged_mi(bare), "o-", label="bare Markov", color="#d62728")
for g in [1.0, 4.0]:
    ax[0].plot(range(1, 21), lagged_mi(myc_seqs[g]), "o-",
               label=f"mycelial IOR={g}", lw=1)
shuf = np.array(bare); rng.shuffle(shuf)
ax[0].plot(range(1, 21), lagged_mi(list(shuf)), "k:", lw=1, label="shuffled floor")
ax[0].set_xlabel("lag k (notes)"); ax[0].set_ylabel("mutual information (bits)")
ax[0].set_title("Long-range memory: MI(s_t ; s_t-k)\n(high even-lag MI in bare = it is LOOPING)",
                fontsize=9); ax[0].legend(fontsize=7)

loops = [loop_frac(bare)*100] + [loop_frac(myc_seqs[g])*100 for g in sweep]
offs  = [out_of_grammar(bare)*100] + [out_of_grammar(myc_seqs[g])*100 for g in sweep]
labels = ["bare"] + [f"IOR{g}" for g in sweep]
x = np.arange(len(labels))
ax[1].bar(x - 0.2, loops, 0.4, label="loop %", color="#1f77b4")
ax[1].bar(x + 0.2, offs, 0.4, label="off-grammar %", color="#ff7f0e")
ax[1].set_xticks(x); ax[1].set_xticklabels(labels, fontsize=8)
ax[1].set_ylabel("%"); ax[1].legend(fontsize=8)
ax[1].set_title("The tradeoff: do loops fall faster than style breaks?", fontsize=9)
fig.tight_layout(); fig.savefig("mycelial_probe_results.png", dpi=120, bbox_inches="tight")
print("\nfigure -> mycelial_probe_results.png")
