import sys
sys.path.insert(0, ".")
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from agent import QLearningAgent
from smp_engine import compute_smp
from environment import (
    ACTIONS, CE_MODES, TRAFFIC_BUCKETS, bucket_to_representative_interval,
    normalize_lifetime, normalize_delay, DELAY_WEIGHT_BY_BUCKET,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

N_STATES_DYNAMIC = len(CE_MODES) * len(TRAFFIC_BUCKETS)  # adapts automatically
agent = QLearningAgent(n_states=N_STATES_DYNAMIC, n_actions=42)
agent.load("results/q_table.npy")

bucket_names = [b[2] for b in TRAFFIC_BUCKETS]
ce_names = [f"CE-{c}" for c in ["A", "B", "C"]]
T3412_vals = sorted(set(t3412 for t3412, _t3324 in ACTIONS))
CORRECTNESS_GAP_THRESHOLD = 0.02

# Build the T3412 grid (color), the full action label (cell text), and
# audit each state against its true-optimal answer directly -- so the
# figure honestly shows its own reliability, not just the raw policy.
t3412_grid = np.zeros((len(CE_MODES), len(TRAFFIC_BUCKETS)))
label_grid = [["" for _ in TRAFFIC_BUCKETS] for _ in CE_MODES]
wrong_cells = []  # (ce, bucket) pairs that fail the audit
n_correct = 0

for ce in CE_MODES:
    for bucket in range(len(TRAFFIC_BUCKETS)):
        state_idx = ce * len(TRAFFIC_BUCKETS) + bucket
        action_idx = int(np.argmax(agent.q_table[state_idx]))
        T3412, T3324 = ACTIONS[action_idx]
        t3412_grid[ce, bucket] = T3412
        label_grid[ce][bucket] = f"{T3412}s\n/{T3324}s"

        # True-optimal check, same method used throughout this project
        interval = bucket_to_representative_interval(bucket)
        w_dl = 1.0 * DELAY_WEIGHT_BY_BUCKET[bucket]
        rewards = {}
        for t3412_candidate in T3412_vals:
            r = compute_smp(t3412_candidate, 20, interval, ce)
            nlt = normalize_lifetime(r.lifetime_yr)
            ndl = normalize_delay(r.delay_seconds)
            rewards[t3412_candidate] = 0.7 * nlt - w_dl * ndl
        true_best = max(rewards, key=rewards.get)
        gap = rewards[true_best] - rewards.get(T3412, rewards[true_best])
        if gap <= CORRECTNESS_GAP_THRESHOLD:
            n_correct += 1
        else:
            wrong_cells.append((ce, bucket))

print(f"Audit: {n_correct}/30 states match their true-optimal answer")
if wrong_cells:
    print(f"Wrong states: {[(ce_names[ce], bucket_names[b]) for ce, b in wrong_cells]}")

# Larger figure and fonts throughout for legibility
fig, ax = plt.subplots(figsize=(16, 5.5))
im = ax.imshow(np.log10(t3412_grid), aspect="auto", cmap="viridis")

# Gridlines between cells -- makes each state visually distinct
ax.set_xticks(np.arange(-0.5, len(TRAFFIC_BUCKETS), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(CE_MODES), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=1.5)
ax.tick_params(which="minor", size=0)

ax.set_xticks(range(len(TRAFFIC_BUCKETS)))
ax.set_xticklabels(bucket_names, rotation=35, ha="right", fontsize=11)
ax.set_yticks(range(len(CE_MODES)))
ax.set_yticklabels(ce_names, fontsize=13)
ax.set_xlabel("Traffic bucket", fontsize=13, labelpad=10)
ax.set_ylabel("Coverage mode", fontsize=13, labelpad=10)
ax.set_title("Learned Policy: T3412/T3324 chosen per state", fontsize=15, pad=15)

# Annotate each cell with the actual chosen action, and outline any
# cell that failed the audit -- the figure now honestly represents its
# own reliability instead of looking uniformly confident everywhere.
vmin, vmax = np.log10(t3412_grid.min()), np.log10(t3412_grid.max())
for ce in CE_MODES:
    for bucket in range(len(TRAFFIC_BUCKETS)):
        val = t3412_grid[ce, bucket]
        norm = (np.log10(val) - vmin) / (vmax - vmin) if vmax > vmin else 0.5
        # FIX: this previously always returned "black" (both branches were
        # identical) -- now genuinely adapts: white text on dark cells,
        # black text on bright (yellow) cells, so nothing goes illegible.
        color = "white" if norm < 0.6 else "black"
        text = label_grid[ce][bucket]
        if (ce, bucket) in wrong_cells:
            text += "\n(unverified)"
        ax.text(bucket, ce, text, ha="center", va="center", fontsize=10.5,
                 color=color, fontweight="bold")
        if (ce, bucket) in wrong_cells:
            rect = plt.Rectangle((bucket - 0.5, ce - 0.5), 1, 1, fill=False,
                                  edgecolor="red", linewidth=3)
            ax.add_patch(rect)

cbar = fig.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label("log10(T3412, seconds)", fontsize=12)
cbar.ax.tick_params(labelsize=10)

fig.tight_layout()
path = os.path.join(FIG_DIR, "policy_heatmap.png")
fig.savefig(path, bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"[SAVED] {path}")