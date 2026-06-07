"""Phase 3.5 — Token Reality Check.

5 key checks to verify tokens are healthy, structured, diverse,
temporally dynamic, and sensitive to EEG frequency content.
Runs on real EEG data using the updated DataLoader and EEGEncoder.
"""

import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from phase_02_mse_tok import EEGEncoder
from phase_02_DL import create_dataloader

# ── Setup ─────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = EEGEncoder().to(device)
model.eval()

loader = create_dataloader(data_dir="data/EEGdatanpy", batch_size=64, split="all")

# Collect 4 batches for more robust checks (256 trials total)
all_tokens = []
all_eeg    = []

with torch.no_grad():
    for i, (eeg, labels, subjects) in enumerate(loader):
        eeg = eeg.to(device)
        tokens = model(eeg)                  # [64, 62, 256]
        all_tokens.append(tokens.cpu())
        all_eeg.append(eeg.cpu())
        if i >= 3:
            break

tokens_all = torch.cat(all_tokens, dim=0)   # [256, 62, 256]
eeg_all    = torch.cat(all_eeg,    dim=0)   # [256, 250, 64]

print(f"\nTokens collected: {tokens_all.shape}  "
      f"({tokens_all.shape[0]} trials × {tokens_all.shape[1]} tokens × {tokens_all.shape[2]} dim)\n")

# ══════════════════════════════════════════════════════════════════════════
# CHECK 1 — BASIC STATS (Health Check)
# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  CHECK 1 — BASIC STATS")
print("=" * 60)

t = tokens_all
print(f"  Min   : {t.min().item():.4f}")
print(f"  Max   : {t.max().item():.4f}")
print(f"  Mean  : {t.mean().item():.4f}   (ideal: ≈ 0)")
print(f"  Std   : {t.std().item():.4f}    (ideal: 0.05 – 0.5)")
print(f"  NaN   : {t.isnan().any().item()}")
print(f"  Inf   : {t.isinf().any().item()}")

mean_ok = abs(t.mean().item()) < 0.1
std_ok  = 0.05 < t.std().item() < 1.0
nan_ok  = not t.isnan().any().item()
print(f"\n  Verdict: {'✅ PASS' if mean_ok and std_ok and nan_ok else '❌ FAIL'}")

# ══════════════════════════════════════════════════════════════════════════
# CHECK 2 — PCA STRUCTURE (Are tokens structured or a random ball?)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  CHECK 2 — PCA STRUCTURE")
print("=" * 60)

# flatten to [N_tokens, 256]
tokens_flat = tokens_all.numpy().reshape(-1, tokens_all.shape[-1])  # [256*62, 256]

pca = PCA(n_components=10)
pca.fit(tokens_flat[:5000])   # limit for speed

explained = pca.explained_variance_ratio_
print(f"  Top 10 PCA components explain:")
for i, v in enumerate(explained):
    print(f"    PC{i+1}: {v*100:.1f}%")

top1 = explained[0]
top3 = explained[:3].sum()
collapsed = top1 > 0.80   # if 1 component explains 80%+ → collapsed

print(f"\n  PC1 explains : {top1*100:.1f}%  (collapsed if > 80%)")
print(f"  PC1–3 explain: {top3*100:.1f}%  (ideal: < 70%)")
print(f"\n  Verdict: {'❌ COLLAPSED — tokens all look the same' if collapsed else '✅ PASS — tokens are spread across multiple dimensions'}")

# ══════════════════════════════════════════════════════════════════════════
# CHECK 3 — COSINE SIMILARITY (Are tokens diverse or all pointing same way?)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  CHECK 3 — COSINE SIMILARITY")
print("=" * 60)

sample = tokens_flat[:1000]
sim_matrix = cosine_similarity(sample)
np.fill_diagonal(sim_matrix, 0)   # remove self-similarity

avg_sim = sim_matrix.mean()
max_sim = sim_matrix.max()

print(f"  Average similarity: {avg_sim:.4f}  (ideal: 0.1 – 0.4)")
print(f"  Max similarity    : {max_sim:.4f}  (ideal: < 0.95)")

diverse = avg_sim < 0.6
print(f"\n  Verdict: {'✅ PASS — tokens are diverse' if diverse else '❌ FAIL — tokens are too similar (collapsed)'}")

# ══════════════════════════════════════════════════════════════════════════
# CHECK 4 — TEMPORAL DYNAMICS (Do tokens change across the 62 time steps?)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  CHECK 4 — TEMPORAL DYNAMICS")
print("=" * 60)

# For each trial, compute mean absolute difference between consecutive tokens
# tokens_all: [256, 62, 256]
diffs = (tokens_all[:, 1:, :] - tokens_all[:, :-1, :]).abs().mean().item()
variance_over_time = tokens_all.var(dim=1).mean().item()

print(f"  Mean abs diff between consecutive tokens: {diffs:.6f}")
print(f"  Token variance across time               : {variance_over_time:.6f}")

dynamic = diffs > 0.001
print(f"\n  Verdict: {'✅ PASS — tokens change across time (dynamic)' if dynamic else '❌ FAIL — tokens are frozen (no temporal change)'}")

# ══════════════════════════════════════════════════════════════════════════
# CHECK 5 — FREQUENCY SENSITIVITY (Do tokens react to EEG frequency bands?)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  CHECK 5 — FREQUENCY SENSITIVITY")
print("=" * 60)

def fft_band_filter(x, low, high, fs=250):
    """Filter EEG to a specific frequency band using FFT. x: [B, T, C]"""
    Xf    = torch.fft.rfft(x, dim=1)
    freqs = torch.fft.rfftfreq(x.shape[1], d=1.0/fs)
    mask  = ((freqs >= low) & (freqs <= high)).float()
    Xf_filtered = Xf * mask[None, :, None].to(x.device)
    return torch.fft.irfft(Xf_filtered, n=x.shape[1], dim=1)

eeg_sample = eeg_all[:32].to(device)   # use 32 trials for speed

with torch.no_grad():
    t_orig  = model(eeg_sample)

    x_delta = fft_band_filter(eeg_sample, low=0.5, high=4.0)    # Delta band
    x_gamma = fft_band_filter(eeg_sample, low=30.0, high=45.0)  # Gamma band

    t_delta = model(x_delta)
    t_gamma = model(x_gamma)

diff_delta = (t_orig - t_delta).abs().mean().item()
diff_gamma = (t_orig - t_gamma).abs().mean().item()

print(f"  Token diff (original vs Delta 0.5–4 Hz) : {diff_delta:.6f}")
print(f"  Token diff (original vs Gamma 30–45 Hz) : {diff_gamma:.6f}")

sensitive = diff_delta > 0.001 or diff_gamma > 0.001
print(f"\n  Verdict: {'✅ PASS — tokens respond to frequency content' if sensitive else '❌ FAIL — tokens are frequency-blind'}")

# ══════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  PHASE 3.5 — FINAL VERDICT")
print("=" * 60)
results = {
    "Basic Stats"         : mean_ok and std_ok and nan_ok,
    "PCA Structure"       : not collapsed,
    "Cosine Similarity"   : diverse,
    "Temporal Dynamics"   : dynamic,
    "Frequency Sensitivity": sensitive,
}
for check, passed in results.items():
    print(f"  {'✅' if passed else '❌'}  {check}")

all_passed = all(results.values())
print(f"\n  {'✅ All checks passed — tokens are ready for Phase 4!' if all_passed else '⚠️  Some checks failed — review before proceeding.'}")
print("=" * 60 + "\n")

# ══════════════════════════════════════════════════════════════════════════
# VISUALIZATIONS — one plot per check
# ══════════════════════════════════════════════════════════════════════════
import matplotlib
matplotlib.use("Agg")   # no display needed — saves to files
import matplotlib.pyplot as plt
from pathlib import Path

out_dir = Path("outputs/token_viz")
out_dir.mkdir(parents=True, exist_ok=True)

# ── VIZ 1: Token Value Distribution (Check 1) ─────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(tokens_flat.flatten(), bins=100, color="#4C72B0", edgecolor="none", alpha=0.85)
ax.axvline(0, color="red", linestyle="--", linewidth=1, label="mean=0")
ax.set_title("Token Value Distribution  (Check 1 — Basic Stats)")
ax.set_xlabel("Token Value")
ax.set_ylabel("Frequency")
ax.legend()
fig.tight_layout()
fig.savefig(out_dir / "check1_token_distribution.png", dpi=150)
plt.close(fig)
print("Saved: check1_token_distribution.png")

# ── VIZ 2: PCA 2D Scatter (Check 2) ───────────────────────────────────────
pca2 = PCA(n_components=2)
tokens_2d = pca2.fit_transform(tokens_flat[:5000])

fig, ax = plt.subplots(figsize=(7, 6))
sc = ax.scatter(tokens_2d[:, 0], tokens_2d[:, 1], s=3, alpha=0.4,
                c=np.arange(len(tokens_2d)), cmap="plasma")
plt.colorbar(sc, ax=ax, label="Token index")
ax.set_title("PCA 2D Scatter of Tokens  (Check 2 — Structure)")
ax.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)")
fig.tight_layout()
fig.savefig(out_dir / "check2_pca_scatter.png", dpi=150)
plt.close(fig)
print("Saved: check2_pca_scatter.png")

# ── VIZ 3: Cosine Similarity Heatmap (Check 3) ────────────────────────────
sample_200 = tokens_flat[:200]
sim_200    = cosine_similarity(sample_200)

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(sim_200, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(im, ax=ax, label="Cosine Similarity")
ax.set_title("Cosine Similarity Heatmap (200 tokens)  (Check 3)")
ax.set_xlabel("Token index")
ax.set_ylabel("Token index")
fig.tight_layout()
fig.savefig(out_dir / "check3_cosine_heatmap.png", dpi=150)
plt.close(fig)
print("Saved: check3_cosine_heatmap.png")

# ── VIZ 4: Temporal Token Dynamics (Check 4) ──────────────────────────────
# Show mean token value across the 62 time steps for 5 random trials
fig, ax = plt.subplots(figsize=(9, 4))
for i in range(5):
    trial_mean = tokens_all[i].numpy().mean(axis=1)  # [62] mean over 256 dim
    ax.plot(trial_mean, alpha=0.8, label=f"Trial {i+1}")
ax.set_title("Mean Token Value Across 62 Time Steps  (Check 4 — Temporal Dynamics)")
ax.set_xlabel("Token Position (time)")
ax.set_ylabel("Mean Token Value")
ax.legend()
fig.tight_layout()
fig.savefig(out_dir / "check4_temporal_dynamics.png", dpi=150)
plt.close(fig)
print("Saved: check4_temporal_dynamics.png")

# ── VIZ 5: Frequency Sensitivity Bar Chart (Check 5) ──────────────────────
labels_freq  = ["Original vs Delta\n(0.5–4 Hz)", "Original vs Gamma\n(30–45 Hz)"]
values_freq  = [diff_delta, diff_gamma]
colors_freq  = ["#2196F3", "#FF5722"]

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(labels_freq, values_freq, color=colors_freq, alpha=0.85, width=0.5)
ax.axhline(0.001, color="gray", linestyle="--", linewidth=1, label="min threshold (0.001)")
for bar, val in zip(bars, values_freq):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.001,
            f"{val:.4f}", ha="center", va="bottom", fontsize=10)
ax.set_title("Frequency Sensitivity  (Check 5)")
ax.set_ylabel("Mean Absolute Token Difference")
ax.legend()
fig.tight_layout()
fig.savefig(out_dir / "check5_frequency_sensitivity.png", dpi=150)
plt.close(fig)
print("Saved: check5_frequency_sensitivity.png")

print(f"\n✅ All visualizations saved to: {out_dir.resolve()}\n")