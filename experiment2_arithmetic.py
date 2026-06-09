"""
experiment2_arithmetic.py

Experiment 2: Graph-Legal but Arithmetic-Obstructed Continuations

Goal: Show that graph legality alone is insufficient.
Two transitions can be equally graph-legal yet have different continuation admissibility.

Graph: 5-cycle with chord
  0 → 1 → 2 → 3 → 4 → 0  (cycle edges, weight=1)
  0 → 2                    (chord, weight=varies)

Key result:
  FSM: allows both 0→1 and 0→2
  Arithmetic gate: distinguishes based on continuation weight
"""

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, random
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.manual_seed(42); random.seed(42); np.random.seed(42)

# =============================================================================
# GRAPH DEFINITION
# =============================================================================

N_NODES = 5
# Cycle edges (all weight 1)
CYCLE_EDGES = [(i, (i+1)%N_NODES) for i in range(N_NODES)]
# Chord: 0→2, weight varies in experiment
CHORD = (0, 2)

# Tokens: nodes + special
TOKENS = ['PAD', 'START'] + [f'N{i}' for i in range(N_NODES)]
T2I    = {t:i for i,t in enumerate(TOKENS)}
I2T    = {i:t for t,i in T2I.items()}
VOCAB  = len(TOKENS)

# P4 adjacency (FSM): all edges allowed
fsm_adj = torch.zeros(VOCAB, VOCAB)
for s,t in CYCLE_EDGES + [CHORD]:
    fsm_adj[T2I[f'N{s}'], T2I[f'N{t}']] = 1.0

# Arithmetic continuation gate:
# Allow transition if weight >= threshold
# Cycle edges: weight = 1  → allowed when threshold ≤ 1
# Chord 0→2:   weight varies → control variable

def arithmetic_gate_logits(logits, last_tok, edge_weights, threshold=1):
    """
    Block any edge whose weight < threshold.
    edge_weights: dict (src_name, dst_name) → float
    """
    logits = logits.clone()
    last = I2T.get(last_tok, 'PAD')
    for ti in range(VOCAB):
        dst = I2T.get(ti, 'PAD')
        if dst.startswith('N'):
            w = edge_weights.get((last, dst), 0.0)
            if w < threshold:
                logits[ti] = float('-inf')
    return logits

# =============================================================================
# MODEL
# =============================================================================

class Block(nn.Module):
    def __init__(self, d=32, h=2):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ff   = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
    def forward(self, x):
        T = x.shape[1]
        mask = torch.triu(torch.ones(T,T,dtype=torch.bool), diagonal=1)
        a, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        x = x + a; return x + self.ff(self.ln2(x))

class GPT(nn.Module):
    def __init__(self, V=VOCAB, d=32, h=2, nl=2, ml=32):
        super().__init__()
        self.te = nn.Embedding(V, d); self.pe = nn.Embedding(ml, d)
        self.blocks = nn.Sequential(*[Block(d,h) for _ in range(nl)])
        self.ln = nn.LayerNorm(d); self.head = nn.Linear(d, V)
    def forward(self, idx):
        B,T = idx.shape
        return self.head(self.ln(self.blocks(
            self.te(idx) + self.pe(torch.arange(T).unsqueeze(0)))))

# =============================================================================
# DATA: train on cycle-only paths (model learns cycle traversal)
# Chord is visible in some training paths so model assigns it nonzero probability
# =============================================================================

def rand_path(include_chord=False, max_len=8):
    cur = random.randint(0, N_NODES-1)
    path = [T2I['START'], T2I[f'N{cur}']]
    for _ in range(random.randint(3, max_len)):
        # Sometimes take chord if at node 0 and chord included
        if include_chord and cur == 0 and random.random() < 0.3:
            nxt = 2  # chord to node 2
        else:
            nxt = (cur + 1) % N_NODES  # cycle
        path.append(T2I[f'N{nxt}']); cur = nxt
    return path

def pad_b(paths):
    ml = max(len(p) for p in paths)
    return torch.tensor([p + [0]*(ml-len(p)) for p in paths], dtype=torch.long)

# Train on paths that include the chord (so model learns it exists)
train_data = [rand_path(include_chord=True, max_len=8) for _ in range(3000)]
print("Training model (sees both cycle edges and chord)...")
m = GPT()
opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 30)
for ep in range(30):
    random.shuffle(train_data)
    for i in range(0, len(train_data), 64):
        b = train_data[i:i+64]
        if not b: continue
        x = pad_b(b); inp, tgt = x[:,:-1], x[:,1:]
        logits = m(inp); B,T,V_ = logits.shape
        loss = F.cross_entropy(logits.reshape(B*T,V_), tgt.reshape(B*T), ignore_index=0)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    sch.step()
print("  Done.\n")

# =============================================================================
# EXPERIMENT: compare FSM vs arithmetic gate on chord usage
# Weight assignments:
#   Condition A: chord weight = 0.5 (below threshold 1.0) → gate blocks
#   Condition B: chord weight = 2.0 (above threshold 1.0) → gate allows
#   FSM: always allows chord (only checks adjacency)
# =============================================================================

def measure_chord_rate(n=500, mode='baseline', chord_weight=1.0, threshold=1.0):
    """
    Generate from node 0. Measure fraction of times N2 is chosen as first next token.
    (N2 can be reached via chord 0→2 or via cycle 0→1→2.)
    Specifically: measure P(immediate next = N2 | at N0).
    """
    chord_count = 0   # next token is N2 directly (chord used)
    cycle_count = 0   # next token is N1 (cycle used)
    invalid_count = 0

    # Edge weights for arithmetic gate
    cycle_w = {(f'N{i}', f'N{(i+1)%N_NODES}'): 1.0 for i in range(N_NODES)}
    edge_w = {**cycle_w, ('N0', 'N2'): chord_weight}

    for _ in range(n):
        prefix = [T2I['START'], T2I['N0']]
        idx = torch.tensor([prefix]); out = list(prefix)

        logits = m(idx)[:, -1, :].squeeze(0)

        if mode == 'fsm':
            # Only block non-adjacent tokens
            last = out[-1]
            node_t = torch.tensor([1.0 if I2T.get(i,'').startswith('N') else 0.0
                                    for i in range(VOCAB)])
            m4 = fsm_adj[last]
            logits[node_t * (1-m4) > 0] = float('-inf')

        elif mode == 'arithmetic':
            last = out[-1]
            logits = arithmetic_gate_logits(logits, last, edge_w, threshold)

        probs = F.softmax(logits, dim=-1)
        tok = torch.multinomial(probs, 1).item()
        name = I2T.get(tok, 'PAD')

        if name == 'N2': chord_count += 1
        elif name == 'N1': cycle_count += 1
        else: invalid_count += 1

    return chord_count/n, cycle_count/n, invalid_count/n

print("="*65)
print("EXPERIMENT 2: Graph-Legal but Arithmetic-Obstructed Continuations")
print("="*65)
print("\nSetup: at node N0, both N1 (cycle) and N2 (chord) are graph-legal.")
print("FSM allows both. Arithmetic gate distinguishes by edge weight.\n")

print(f"{'Condition':<30} {'FSM':>10} {'Arith gate':>12}")
print(f"{'Model/Mode':<30} {'N2 rate':>10} {'N2 rate':>12}")
print("-"*55)

results = {}
for chord_w, label in [(0.5, 'chord_weight=0.5 (blocked)'),
                        (1.0, 'chord_weight=1.0 (threshold)'),
                        (2.0, 'chord_weight=2.0 (allowed)')]:
    fsm_chord, fsm_cycle, fsm_inv   = measure_chord_rate(n=500, mode='fsm',
                                                           chord_weight=chord_w)
    arith_chord, arith_cycle, arith_inv = measure_chord_rate(n=500,
                                                               mode='arithmetic',
                                                               chord_weight=chord_w)
    results[label] = (fsm_chord, arith_chord)
    print(f"  {label:<28}  {fsm_chord:.3f}      {arith_chord:.3f}")

print(f"""
Key result:
  When chord_weight < threshold: arithmetic gate blocks N2 (arith≈0)
  while FSM still allows it (fsm > 0).
  This is the anti-FSM result: same graph topology, different admissibility.
""")

# =============================================================================
# FIGURE
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Experiment 2: Graph-Legal but Arithmetic-Obstructed Continuations\n'
             'FSM allows both 0→1 and 0→2; Arithmetic gate distinguishes by weight',
             fontsize=11, fontweight='bold')

# Panel 1: Chord usage rate by condition
labels = ['weight=0.5\n(blocked)', 'weight=1.0\n(threshold)', 'weight=2.0\n(allowed)']
fsm_rates   = [results[k][0] for k in results]
arith_rates = [results[k][1] for k in results]
x = np.arange(3); w = 0.35
axes[0].bar(x-w/2, fsm_rates,   w, color='#e74c3c', alpha=0.85, label='FSM (adjacency only)')
axes[0].bar(x+w/2, arith_rates, w, color='#27ae60', alpha=0.85, label='Arithmetic gate')
axes[0].set_title('Chord (N0→N2) Usage Rate\n(FSM: insensitive to weight)',
                   fontweight='bold')
axes[0].set_ylabel('P(next=N2 | at N0)')
axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
axes[0].legend(); axes[0].grid(axis='y', alpha=0.3)
axes[0].axhline(0, color='black', lw=0.5)

# Panel 2: Graph diagram
ax2 = axes[1]; ax2.set_aspect('equal'); ax2.axis('off')
ax2.set_title('5-Cycle + Chord Graph\n(same adjacency, different arithmetic weights)',
              fontweight='bold')
angles = [2*np.pi*i/5 - np.pi/2 for i in range(5)]
xs = [np.cos(a) for a in angles]; ys = [np.sin(a) for a in angles]
for i, (x_,y_) in enumerate(zip(xs,ys)):
    ax2.add_patch(plt.Circle((x_,y_), 0.12, color='#3498db', zorder=3))
    ax2.text(x_, y_, f'N{i}', ha='center', va='center',
              fontsize=10, fontweight='bold', color='white', zorder=4)
for i in range(5):
    s,t = i, (i+1)%5
    ax2.annotate('', xy=(xs[t]*0.88, ys[t]*0.88),
                  xytext=(xs[s]*0.88, ys[s]*0.88),
                  arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.5))
# Chord
ax2.annotate('', xy=(xs[2]*0.88, ys[2]*0.88),
              xytext=(xs[0]*0.88, ys[0]*0.88),
              arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2,
                               connectionstyle='arc3,rad=0.3'))
ax2.text((xs[0]+xs[2])/2+0.15, (ys[0]+ys[2])/2+0.1,
          'chord\n(weight=var)', fontsize=8, color='#e74c3c', ha='center')
ax2.set_xlim(-1.5, 1.5); ax2.set_ylim(-1.5, 1.5)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/experiment2.png', dpi=150, bbox_inches='tight')
print("Figure saved: experiment2.png")
