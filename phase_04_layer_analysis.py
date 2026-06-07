"""Phase 4 — Layer-by-Layer Analysis.

Captures what each Transformer layer learns by hooking into the output
of Layer 1 and comparing it against the input and the final output.

Flow captured:
    tokens [64, 62, 256]          ← Phase 3 output
        ↓ + positional embedding
    transformer_input [64,62,256] ← what enters the transformer
        ↓ Transformer Layer 1
    layer1_out [64, 62, 256]      ← what Layer 1 learned
        ↓ Transformer Layer 2
    embedding [64, 62, 256]       ← what Layer 2 refined

Uses REAL EEG data from the DataLoader — not dummy data.
"""

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from phase_02_DL       import create_dataloader
from phase_02_mse_tok  import EEGEncoder
from phase_04_attention import EEGTransformerEncoder

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Models ────────────────────────────────────────────────────────────────────
encoder   = EEGEncoder().to(device).eval()
attention = EEGTransformerEncoder().to(device).eval()

# ── Real Data ─────────────────────────────────────────────────────────────────
loader = create_dataloader(data_dir="data/EEGdatanpy", batch_size=64, split="all")
eeg, labels, subjects = next(iter(loader))
eeg = eeg.to(device)

# ── Phase 3: get real tokens ──────────────────────────────────────────────────
with torch.no_grad():
    tokens = encoder(eeg)   # [64, 62, 256]

# ── Hook: capture Layer 1 output ──────────────────────────────────────────────
layer1_out = []

def _hook(module, input, output):
    layer1_out.append(output.detach().cpu())

hook = attention.transformer.layers[0].register_forward_hook(_hook)

with torch.no_grad():
    embedding = attention(tokens)   # [64, 62, 256]

hook.remove()  # always clean up hooks

# ── Extract captured Layer 1 output ───────────────────────────────────────────
layer1 = layer1_out[0].to(device)   # [64, 62, 256]

# ── Compute transformer input (tokens + positional embedding) ─────────────────
# In eval mode, dropout = 0 so this is exact
trans_input = (tokens + attention.pos_embedding[:, :tokens.size(1), :]).detach()

# Move everything to CPU for analysis
trans_input_cpu = trans_input.cpu()
layer1_cpu      = layer1.cpu()
embedding_cpu   = embedding.detach().cpu()

print("\n" + "=" * 60)
print("  PHASE 4 — LAYER-BY-LAYER ANALYSIS")
print("=" * 60)

# ── 1. How much did each layer change the data? ───────────────────────────────
diff_layer1 = (layer1_cpu - trans_input_cpu).abs().mean().item()
diff_layer2 = (embedding_cpu - layer1_cpu).abs().mean().item()
diff_total  = (embedding_cpu - trans_input_cpu).abs().mean().item()

print(f"\n[A] Mean Absolute Change per Layer:")
print(f"    Transformer input → Layer 1 output : {diff_layer1:.6f}")
print(f"    Layer 1 output   → Layer 2 output  : {diff_layer2:.6f}")
print(f"    Total change (input → final)        : {diff_total:.6f}")
print(f"    Layer 1 contributed : {diff_layer1/diff_total*100:.1f}%")
print(f"    Layer 2 contributed : {diff_layer2/diff_total*100:.1f}%")

# ── 2. Variance at each stage ─────────────────────────────────────────────────
var_input  = trans_input_cpu.var(dim=2).mean().item()
var_layer1 = layer1_cpu.var(dim=2).mean().item()
var_embed  = embedding_cpu.var(dim=2).mean().item()

print(f"\n[B] Variance across embedding dimension (per token):")
print(f"    Transformer input : {var_input:.6f}")
print(f"    After Layer 1     : {var_layer1:.6f}  (Δ {var_layer1-var_input:+.6f})")
print(f"    After Layer 2     : {var_embed:.6f}  (Δ {var_embed-var_layer1:+.6f})")

# ── 3. Per-token position change ──────────────────────────────────────────────
# Mean absolute change at each of the 62 token positions
change_l1 = (layer1_cpu - trans_input_cpu).abs().mean(dim=(0, 2)).detach().numpy()  # [62]
change_l2 = (embedding_cpu - layer1_cpu).abs().mean(dim=(0, 2)).detach().numpy()    # [62]

print(f"\n[C] Per-position change (averaged across batch and dim):")
print(f"    Layer 1 — max change at token: {change_l1.argmax()}  (value: {change_l1.max():.4f})")
print(f"    Layer 2 — max change at token: {change_l2.argmax()}  (value: {change_l2.max():.4f})")

# ── Visualizations ────────────────────────────────────────────────────────────
out_dir = Path("outputs/phase04_layer_analysis")
out_dir.mkdir(parents=True, exist_ok=True)

# --- Plot A: Bar chart — which layer contributes more?
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(
    ["Layer 1\n(first pass)", "Layer 2\n(refinement)"],
    [diff_layer1, diff_layer2],
    color=["#4C72B0", "#DD8452"],
    width=0.4
)
ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=11)
ax.set_ylabel("Mean Absolute Change")
ax.set_title("How Much Each Transformer Layer Changed the Tokens\n(Real EEG Data)")
ax.set_ylim(0, max(diff_layer1, diff_layer2) * 1.3)
fig.tight_layout()
fig.savefig(out_dir / "ph4_layer_contribution.png", dpi=150)
plt.close(fig)
print("\nSaved: ph4_layer_contribution.png")

# --- Plot B: Line plot — per-token change across 62 positions
fig, ax = plt.subplots(figsize=(12, 4))
positions = np.arange(62)
ax.plot(positions, change_l1, color="#4C72B0", linewidth=1.5, label="Layer 1 change")
ax.plot(positions, change_l2, color="#DD8452", linewidth=1.5, label="Layer 2 change")
ax.fill_between(positions, change_l1, alpha=0.15, color="#4C72B0")
ax.fill_between(positions, change_l2, alpha=0.15, color="#DD8452")
ax.set_xlabel("Token Position (0 = earliest in trial, 61 = latest)")
ax.set_ylabel("Mean Absolute Change")
ax.set_title("Per-Token Change at Each Transformer Layer (Real EEG Data)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(out_dir / "ph4_per_token_change.png", dpi=150)
plt.close(fig)
print("Saved: ph4_per_token_change.png")

# --- Plot C: Variance progression
fig, ax = plt.subplots(figsize=(7, 4))
stages = ["Transformer\nInput", "After\nLayer 1", "After\nLayer 2"]
variances = [var_input, var_layer1, var_embed]
ax.plot(stages, variances, "o-", color="#2ca02c", linewidth=2, markersize=8)
for i, (s, v) in enumerate(zip(stages, variances)):
    ax.annotate(f"{v:.5f}", (s, v), textcoords="offset points",
                xytext=(0, 10), ha="center", fontsize=10)
ax.set_ylabel("Mean Token Variance (across embed dim)")
ax.set_title("Variance Evolution Through Transformer Layers (Real EEG Data)")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(out_dir / "ph4_variance_progression.png", dpi=150)
plt.close(fig)
print("Saved: ph4_variance_progression.png")

print(f"\n✅ All plots saved to: {out_dir.resolve()}")
print("=" * 60 + "\n")
