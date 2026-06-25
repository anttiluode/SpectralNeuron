"""
mycelial_learned_gates.py
==================================================================
The "Mamba" Upgrade: Data-Dependent Mycelial Drift.

Instead of a fixed metabolic decay (e.g., 0.85) and a fixed IOR penalty,
this model learns to dynamically gate its own memory. It uses gradient
descent to learn when to hold the metabolic trace, and when to flush 
it based on the musical context (the "grammar").

If the plot shows the decay rate fluctuating based on the melody
(e.g., dropping near the end of a phrase to flush the buffer), 
it proves the system learned syntax, not just a mechanical metronome.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- 1. Data Setup (From previous script) ----
NAMES = ["A3","C4","D4","E4","G4","A4","C5","D5","E5"]
NS = len(NAMES)

# Toy melodies (pitch indices only for sequence modeling)
TRAIN = [
    [5, 4, 3, 4, 5, 6, 5, 4, 3, 2, 0],
    [0, 2, 3, 5, 4, 3, 2, 3, 4, 5],
    [5, 6, 7, 6, 5, 4, 3, 2, 0, 2, 0],
    [3, 4, 5, 7, 6, 5, 4, 3, 2, 0],
]

# Prepare tensors
seqs = [torch.tensor(m, dtype=torch.long) for m in TRAIN]

# ---- 2. The Upgraded Model (Data-Dependent Gates) ----
class LearnedMycelialCortex(nn.Module):
    def __init__(self, vocab_size=NS, embed_dim=16):
        super().__init__()
        self.vocab_size = vocab_size
        
        # Order-2 context embedding
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.context_net = nn.Sequential(
            nn.Linear(embed_dim * 2, 32),
            nn.ReLU()
        )
        
        # The Base Markov Transition (Learned)
        self.base_logits = nn.Linear(32, vocab_size)
        
        # THE UPGRADE: The Data-Dependent Metabolic Knobs
        # Outputs 3 values: [decay_gate, ior_strength, momentum_strength]
        self.gates = nn.Linear(32, 3)
        
    def forward(self, seq, teacher_forcing=True):
        # seq shape: [Length]
        L = len(seq)
        logits_out = []
        gate_history = []
        
        # Biological state variables
        trace = torch.zeros(self.vocab_size)
        mom = torch.zeros(1)
        
        # We need 2 notes of context to start
        for t in range(2, L):
            prev2 = seq[t-2]
            prev1 = seq[t-1]
            
            # 1. Read Context
            emb2 = self.embed(prev2)
            emb1 = self.embed(prev1)
            ctx = self.context_net(torch.cat([emb2, emb1], dim=-1))
            
            # 2. Base Guess
            base_l = self.base_logits(ctx)
            
            # 3. Dynamic Metabolic Gating (The "Mamba" mechanism)
            g = self.gates(ctx)
            decay = torch.sigmoid(g[0])      # Bound between 0 and 1
            g_ior = F.softplus(g[1])         # Ensure positive penalty
            g_mom = F.softplus(g[2])         # Ensure positive directional push
            
            gate_history.append(decay.item())
            
            # 4. Apply the Mycelial Skew
            idx_tensor = torch.arange(self.vocab_size).float()
            dir_vec = (idx_tensor - prev1.float()) / self.vocab_size
            
            logits = base_l - (g_ior * trace) + (g_mom * mom * dir_vec)
            logits_out.append(logits.unsqueeze(0))
            
            # 5. Update the Biological State
            if teacher_forcing:
                actual_token = seq[t]
            else:
                # Sample from distribution during dreaming
                probs = F.softmax(logits, dim=-1)
                actual_token = torch.multinomial(probs, 1)[0]
                seq[t] = actual_token # Overwrite buffer if generating
                
            one_hot = F.one_hot(actual_token, num_classes=self.vocab_size).float()
            
            # Trace decays by the LEARNED, context-specific amount
            trace = trace * decay + one_hot
            
            diff = actual_token.float() - prev1.float()
            mom = 0.9 * mom + 0.1 * torch.sign(diff)
            
        return torch.cat(logits_out, dim=0), gate_history

# ---- 3. Training Loop ----
model = LearnedMycelialCortex()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("=" * 66)
print("TRAINING DATA-DEPENDENT MYCELIAL GATES")
print("=" * 66)

epochs = 300
for epoch in range(epochs):
    total_loss = 0
    for s in seqs:
        optimizer.zero_grad()
        logits, _ = model(s, teacher_forcing=True)
        # Target is the sequence shifted by 2 (since we use order-2 context)
        targets = s[2:]
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {total_loss/len(seqs):.4f}")

# ---- 4. Dreaming (Generation) and Measurement ----
print("\n" + "=" * 66)
print("DREAMING & MEASURING METABOLIC GATES")
print("=" * 66)

# We seed it and let it run unclamped
gen_len = 25
dream_seq = torch.zeros(gen_len, dtype=torch.long)
dream_seq[0] = 5 # A4
dream_seq[1] = 4 # G4

model.eval()
with torch.no_grad():
    _, decay_history = model(dream_seq, teacher_forcing=False)

dreamed_notes = [NAMES[idx.item()] for idx in dream_seq]
print("Dreamed Sequence:", " ".join(dreamed_notes))

# ---- 5. Visualizing the Grammar (The Decisive Read) ----
fig, ax1 = plt.subplots(figsize=(12, 5))

x = np.arange(2, gen_len)
y_notes = dream_seq[2:].numpy()

ax1.plot(x, y_notes, 'o-', color='#1f77b4', label='Generated Pitch')
ax1.set_yticks(range(NS))
ax1.set_yticklabels(NAMES)
ax1.set_ylabel('Pitch State', color='#1f77b4')
ax1.tick_params(axis='y', labelcolor='#1f77b4')

ax2 = ax1.twinx()
ax2.plot(x, decay_history, 's--', color='#d62728', label='Learned Decay Gate')
ax2.set_ylabel('Decay Rate (1.0=Hold, 0.0=Flush)', color='#d62728')
ax2.set_ylim(0, 1)
ax2.tick_params(axis='y', labelcolor='#d62728')

plt.title('Data-Dependent Mycelial Drift\nDoes the decay gate fluctuate to parse musical grammar?')
fig.tight_layout()
fig.savefig("learned_tide_dynamics.png", dpi=120)
print("Figure saved -> learned_tide_dynamics.png")