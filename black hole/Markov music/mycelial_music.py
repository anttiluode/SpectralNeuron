"""
mycelial_music.py
==================================================================
The honest build of "fork bank + mycelial sequence memory for music."

Stripped of poetry it is two known pieces wired together:
  (1) a fork bank = a constant-Q-ish filterbank reading raw audio into
      pitch states  (analysis / clamp ON)
  (2) a directional Markov transition memory that learns which state
      follows which, and dreams new sequences by walking the chain
      with a temperature knob  (generation / clamp OFF)

That is a spectrogram front-end + a Markov chain. Markov-chain music
is from 1957 (Illiac Suite). Nothing here is new as DSP or as
generation. What it demonstrates is the TRANSFER: the same
resonant-front-end + transition-manifold + clamp-on/clamp-off shape
as the visual loop, now running on sound.

Produces:
  - fork analysis: recovers the pitch sequence of a synthesized melody
    from raw audio (proves the front end, using the measured resonator)
  - transition_matrix.png : the learned directional skew (asymmetric)
  - dreamed_pianoroll.png  : a generated melody
  - train_example.wav, dream_warm.wav, dream_wild.wav : audio to hear
"""

import numpy as np, wave, struct
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spectral_neuron import integrate_fork

rng = np.random.default_rng(5)
FS = 8000
DT = 1.0 / FS

# ---- A-minor pentatonic alphabet (9 states) ----
NAMES = ["A3","C4","D4","E4","G4","A4","C5","D5","E5"]
FREQ  = np.array([220.0,261.63,293.66,329.63,392.0,440.0,523.25,587.33,659.25])
NS = len(FREQ)

# ---- hand-written training melodies (pentatonic -> pleasant), with a
#      deliberate style: mostly stepwise, occasional leap, resolves to A ----
#      (pitch_index, beats)
TRAIN = [
    [(5,1),(4,1),(3,1),(4,1),(5,2),(6,1),(5,1),(4,2),(3,1),(2,1),(0,2)],
    [(0,1),(2,1),(3,1),(5,1),(4,1),(3,1),(2,2),(3,1),(4,1),(5,2)],
    [(5,1),(6,1),(7,1),(6,1),(5,2),(4,1),(3,1),(2,1),(0,2),(2,1),(0,2)],
    [(3,1),(4,1),(5,1),(7,1),(6,2),(5,1),(4,1),(3,2),(2,1),(0,2)],
]
BEAT = 0.28  # seconds per beat

# ---------- synthesis ----------
def adsr(n):
    e = np.ones(n)
    a = int(0.02*FS); r = int(0.06*FS)
    if a>0: e[:a] = np.linspace(0,1,a)
    if r>0 and r<n: e[-r:] = np.linspace(1,0,r)
    return e

def synth(melody):
    out = []
    for idx, beats in melody:
        n = int(beats*BEAT*FS)
        t = np.arange(n)*DT
        f = FREQ[idx]
        wave_ = (np.sin(2*np.pi*f*t)
                 + 0.4*np.sin(2*np.pi*2*f*t)
                 + 0.2*np.sin(2*np.pi*3*f*t))
        out.append(wave_*adsr(n)*0.5)
    return np.concatenate(out)

def write_wav(path, sig):
    s = sig/ (np.max(np.abs(sig))+1e-9)
    pcm = (s*32767).astype(np.int16)
    with wave.open(path,"w") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(FS)
        w.writeframes(struct.pack("<%dh"%len(pcm), *pcm))

# ---------- FORK ANALYSIS: raw audio -> pitch states ----------
def fork_transcribe(audio, melody):
    """Run a fork at each scale pitch; per known note-window pick the
    fork that rang loudest. Recovers the symbolic sequence from audio.
    (Onsets are given -- this proves the resonant front end, not onset
    detection.)"""
    sig = audio - audio.mean()
    envs = []
    for f in FREQ:
        kw = dict(w0=2*np.pi*f, zeta=0.02, g=1.0, thresh=1e9)  # no firing, just ring
        _, e = integrate_fork(sig, DT, **kw)
        envs.append(e)
    envs = np.array(envs)               # [NS, T]
    recovered, true = [], []
    pos = 0
    for idx, beats in melody:
        n = int(beats*BEAT*FS)
        seg = envs[:, pos:pos+n]
        win = seg[:, int(0.2*n):int(0.8*n)]   # steady middle of the note
        recovered.append(int(np.argmax(win.mean(axis=1))))
        true.append(idx)
        pos += n
    acc = np.mean(np.array(recovered)==np.array(true))
    return recovered, true, acc

# ---------- MYCELIAL TRANSITION MEMORY ----------
def learn(train):
    T1 = np.zeros((NS,NS))                       # 1st order pitch
    T2 = {}                                      # 2nd order pitch
    D1 = {}                                      # duration | pitch
    for mel in train:
        idxs = [m[0] for m in mel]; durs=[m[1] for m in mel]
        for a,b in zip(idxs[:-1], idxs[1:]):
            T1[a,b]+=1
        for a,b,c in zip(idxs[:-2],idxs[1:-1],idxs[2:]):
            T2.setdefault((a,b),np.zeros(NS))[c]+=1
        for p,d in zip(idxs,durs):
            D1.setdefault(p,{}).setdefault(d,0)
            D1[p][d]+=1
    return T1,T2,D1

def sample(counts, temp):
    c = np.asarray(counts,dtype=float)
    if c.sum()==0: return rng.integers(NS)
    logits = np.log(c+1e-9)/max(temp,1e-3)
    p = np.exp(logits-logits.max()); p/=p.sum()
    return int(rng.choice(len(c), p=p))

def dream(T1,T2,D1, seed, length=18, temp=0.7):
    seq=list(seed)
    for _ in range(length-len(seed)):
        ctx=(seq[-2],seq[-1])
        if ctx in T2 and T2[ctx].sum()>0:
            nxt=sample(T2[ctx],temp)
        else:
            nxt=sample(T1[seq[-1]],temp)
        seq.append(nxt)
    # durations from per-pitch distribution
    mel=[]
    for p in seq:
        if p in D1:
            ds=list(D1[p]); ws=np.array([D1[p][d] for d in ds],float)
            d=ds[sample(ws,temp)]
        else: d=1
        mel.append((p,d))
    return mel

# ==================================================================
if __name__=="__main__":
    # 1) synthesize a training melody and transcribe it back with forks
    audio0 = synth(TRAIN[0])
    write_wav("train_example.wav", audio0)
    rec,tru,acc = fork_transcribe(audio0, TRAIN[0])
    print("="*64)
    print("FORK FRONT-END  (raw audio -> pitch states)")
    print("="*64)
    print("true     :", " ".join(NAMES[i] for i in tru))
    print("recovered:", " ".join(NAMES[i] for i in rec))
    print(f"pitch recovery accuracy: {acc*100:.0f}%")

    # 2) learn the transition memory
    T1,T2,D1 = learn(TRAIN)

    print("\n"+"="*64)
    print("DIRECTIONAL SKEW  (transition asymmetry: A->B vs B->A)")
    print("="*64)
    pairs=[(5,4),(2,0),(5,6),(3,4)]
    for a,b in pairs:
        print(f"  {NAMES[a]}->{NAMES[b]}: {T1[a,b]:.0f}    "
              f"{NAMES[b]}->{NAMES[a]}: {T1[b,a]:.0f}")
    asym = np.abs(T1-T1.T).sum()/ (T1.sum()+1e-9)
    print(f"  overall asymmetry index: {asym:.2f}  (0 = symmetric)")

    # 3) dream new melodies at two temperatures, render audio
    warm = dream(T1,T2,D1, seed=[5,4], length=18, temp=0.5)
    wild = dream(T1,T2,D1, seed=[5,4], length=18, temp=1.3)
    write_wav("dream_warm.wav", synth(warm))
    write_wav("dream_wild.wav", synth(wild))

    print("\n"+"="*64)
    print("DREAMED (clamp OFF) — same seed A4->G4, two temperatures")
    print("="*64)
    print("warm(0.5):", " ".join(NAMES[p] for p,_ in warm))
    print("wild(1.3):", " ".join(NAMES[p] for p,_ in wild))

    # 4) direction sensitivity: rising vs falling seed
    up   = dream(T1,T2,D1, seed=[0,2], length=10, temp=0.4)
    down = dream(T1,T2,D1, seed=[7,6], length=10, temp=0.4)
    print("\nrising seed A3->D4 continues :", " ".join(NAMES[p] for p,_ in up))
    print("falling seed D5->C5 continues:", " ".join(NAMES[p] for p,_ in down))

    # ---- figures ----
    fig,ax=plt.subplots(1,2,figsize=(13,5))
    im=ax[0].imshow(T1,cmap="magma")
    ax[0].set_xticks(range(NS)); ax[0].set_xticklabels(NAMES,rotation=90,fontsize=7)
    ax[0].set_yticks(range(NS)); ax[0].set_yticklabels(NAMES,fontsize=7)
    ax[0].set_title("Learned transition skew  (row=from, col=to)\n"
                    "asymmetry = the arrow of time, no AIS needed",fontsize=9)
    ax[0].set_xlabel("to"); ax[0].set_ylabel("from")
    fig.colorbar(im,ax=ax[0],fraction=0.046)

    # piano roll of warm dream
    t=0
    for p,d in warm:
        ax[1].barh(p, d*BEAT, left=t, height=0.6, color="#1f77b4")
        t+=d*BEAT
    ax[1].set_yticks(range(NS)); ax[1].set_yticklabels(NAMES,fontsize=7)
    ax[1].set_xlabel("seconds"); ax[1].set_title("Dreamed melody (temp 0.5)",fontsize=9)
    fig.tight_layout(); fig.savefig("mycelial_music_results.png",dpi=120,bbox_inches="tight")
    print("\nfigure -> mycelial_music_results.png")
