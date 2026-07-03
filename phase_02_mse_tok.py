"""Phase 3: Multi-Scale EEG Encoder.

Converts raw EEG signals [B, 250, 64] into a sequence of context tokens [B, 63, embed_dim].

Four components in sequence:
  1. ChannelAttention  : SE block learns per-electrode weights (0→1)
                         Suppresses noisy frontal channels (Fp1, Fp2, F7, F8 etc.)
                         Amplifies clean occipital/visual channels (O1, O2, Oz, P3, P4)
  2. small_scale conv  : kernel=25, window=100ms — captures C1 (early visual cortex, ~50-80ms)
  3. medium_scale conv : kernel=51, window=204ms — captures N170 (object recognition, ~150-200ms)
  4. large_scale conv  : kernel=75, window=300ms — captures P300 (cognitive categorization, ~300ms)

At 250Hz sampling rate:
  1 sample = 4ms
  kernel=25 → 25 × 4ms = 100ms (C1 / early gamma)
  kernel=51 → 51 × 4ms = 204ms (N170 — the primary object-recognition ERP)
  kernel=75 → 75 × 4ms = 300ms (P300 — category-specific cognitive response)

All kernels are odd → padding = kernel//2 → output length = input length (250 samples).
Tokenizer then downsamples 250 → 63 tokens (stride=4, padding=1 to preserve last samples).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention over EEG electrodes.

    Learns a scalar weight for each of the 64 EEG channels, enabling the model
    to suppress noisy electrodes (e.g. Fp1, Fp2, F7, F8 — frontal muscle artefacts)
    and amplify informative electrodes (e.g. O1, Oz, O2, P3, Pz, P4 — visual cortex).

    Architecture:
        Input  : [B, C, T]  where C=64 electrodes
        Squeeze: global average pool across time → [B, C]   (one scalar per electrode)
        FC down: C → C//r  with GELU  (r=4 → 64→16, learns which electrodes co-activate)
        FC up  : C//r → C  with Sigmoid (gate: 0 = mute channel, 1 = pass channel fully)
        Scale  : multiply gate [B, C, 1] × input [B, C, T] → [B, C, T]

    Parameter count: Linear(C→r) + Linear(r→C) including biases
      = (C×r + r) + (r×C + C) = 2×C×r + r + C
      With C=64, r=16: 2×64×16 + 16 + 64 = 2,048 + 80 = 2,128 trainable parameters.
    """
    def __init__(self, num_channels: int = 64, reduction: int = 4):
        super().__init__()
        reduced = max(num_channels // reduction, 4)   # never go below 4
        self.fc = nn.Sequential(
            nn.Linear(num_channels, reduced, bias=True),
            nn.GELU(),
            nn.Linear(reduced, num_channels, bias=True),
            nn.Sigmoid(),   # output in [0, 1] — a per-channel gate
        )
        # Initialise the final linear's BIAS to 0 so sigmoid(small_random) ≈ 0.5 at start.
        # We do NOT zero the weights — that would make the gate permanently 0.5
        # regardless of input (sigmoid(0×anything + 0) = 0.5 always = dead gate).
        # Default kaiming weight init + bias=0 → gates vary slightly around 0.5 at init.
        nn.init.zeros_(self.fc[-2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        # Squeeze: average signal power across the full 1-second window
        gap = x.mean(dim=-1)               # [B, C]  — global average pool over time

        # Excitation: 2-layer FC produces a gate for every channel
        gate = self.fc(gap)                # [B, C]  values in [0, 1]
        gate = gate.unsqueeze(-1)          # [B, C, 1]  broadcast-ready

        # Scale: apply gate to original signal
        return x * gate                    # [B, C, T]


class ConvBlock(nn.Module):
    """Single temporal convolution branch: Conv1d → GroupNorm → GELU.

    Uses 'same' padding (padding = kernel_size // 2) so output length equals input.
    All kernel sizes must be ODD to guarantee symmetric padding and exact length preservation.

    WHY GroupNorm instead of BatchNorm:
      BatchNorm1d computes normalization statistics over the BATCH during training
      (e.g. mean/std of 256 samples), but uses stored RUNNING AVERAGES during eval.
      This creates a train/eval distribution shift — single-sample inference (batch=1)
      gets normalized by statistics estimated from 256-sample batches, silently
      corrupting every convolutional feature map at generation time.

      GroupNorm normalizes within each individual sample, completely independent of
      batch size. Training and inference are mathematically identical. No running
      statistics, no train/eval mismatch, no amplitude shift.

    GroupNorm config (with out_channels=128, num_groups=8):
      8 groups × 16 channels/group — each group of 16 feature channels is
      normalized together, analogous to how temporal frequency bands co-activate.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups: int = 8):
        super().__init__()
        assert kernel_size % 2 == 1, (
            f"kernel_size must be odd for symmetric 'same' padding, got {kernel_size}"
        )
        assert out_channels % num_groups == 0, (
            f"out_channels ({out_channels}) must be divisible by num_groups ({num_groups})"
        )
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,   # 'same' padding: output length = input length
        )
        # GroupNorm: no running statistics, identical at training and inference
        # Works with any batch size including batch_size=1 (no train/eval shift)
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.act  = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)    # normalized per-sample, per-group — no batch dependency
        x = self.act(x)
        return x


class MultiScaleEEGEncoder(nn.Module):
    """Parallel multi-scale temporal feature extractor for EEG.

    Pipeline:
      ChannelAttention → [small_scale | medium_scale | large_scale] → 1×1 fusion

    ChannelAttention (SE block):
      Learns per-electrode importance weights before any temporal convolution.
      Frontal artefact channels (Fp1, F7, F8 etc.) are gated toward 0.
      Occipital visual channels (O1, Oz, O2, P3, Pz, P4) are kept near 1.
      Result: the three ConvBlocks see mostly clean visual-cortex signal.

    Three ConvBlock branches detect VEPs at different latencies:
      small_scale  : kernel=25, window=100ms — C1 component (early visual cortex)
      medium_scale : kernel=51, window=204ms — N170 component (object recognition)
      large_scale  : kernel=75, window=300ms — P300 component (cognitive categorization)
    """
    def __init__(self, in_channels=64, hidden_dim=128):
        super().__init__()

        # Step 1 — Channel Attention: learn per-electrode weights
        self.channel_attn = ChannelAttention(num_channels=in_channels, reduction=4)

        # Step 2 — Three temporal branches, each seeing the attention-gated signal
        # Output of each: [B, hidden_dim, T]  where T = 250 (length preserved)
        self.small_scale  = ConvBlock(in_channels, hidden_dim, kernel_size=25)  # 100ms — C1
        self.medium_scale = ConvBlock(in_channels, hidden_dim, kernel_size=51)  # 204ms — N170
        self.large_scale  = ConvBlock(in_channels, hidden_dim, kernel_size=75)  # 300ms — P300

        # Step 3 — 1×1 conv fuses the 3 branches: [B, hidden_dim*3, T] → [B, hidden_dim, T]
        self.fusion = nn.Conv1d(hidden_dim * 3, hidden_dim, kernel_size=1)

    def forward(self, x):
        # x: [B, 64, 250]  (channels, time)

        # Step 1: learn which electrodes carry useful visual signal
        x = self.channel_attn(x)         # [B, 64, 250] — frontal noise gated down

        # Step 2: detect VEP components at three timescales
        s = self.small_scale(x)          # [B, 128, 250] — C1
        m = self.medium_scale(x)         # [B, 128, 250] — N170
        l = self.large_scale(x)          # [B, 128, 250] — P300

        # Step 3: fuse all three timescales into one feature map
        combined = torch.cat([s, m, l], dim=1)  # [B, 384, 250]
        out = self.fusion(combined)              # [B, 128, 250]
        return out


class Tokenizer(nn.Module):
    """Downsamples the 250-sample feature map to a sequence of 63 tokens.

    Uses a strided Conv1d (stride=4) to compress 4 time steps into 1 token.
    padding=1 ensures the last 2 samples are not discarded (previously they were):
      Without padding: floor((250-4)/4)+1 = 62 tokens (last 2 samples lost)
      With padding=1:  floor((250+2-4)/4)+1 = 63 tokens (all samples covered)
    """
    def __init__(self, in_dim=128, embed_dim=256, stride=4):
        super().__init__()
        self.token_conv = nn.Conv1d(
            in_channels=in_dim,
            out_channels=embed_dim,
            kernel_size=stride,
            stride=stride,
            padding=1,   # prevents last 2 samples from being silently discarded
        )

    def forward(self, x):
        # x: [B, 128, 250]
        tokens = self.token_conv(x)         # [B, embed_dim, 63]
        tokens = tokens.permute(0, 2, 1)    # [B, 63, embed_dim]
        return tokens


class EEGEncoder(nn.Module):
    """Full Phase 3 encoder: raw EEG [B, 250, 64] → token sequence [B, 63, embed_dim]."""
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.multi_scale = MultiScaleEEGEncoder(in_channels=64, hidden_dim=128)
        self.tokenizer   = Tokenizer(in_dim=128, embed_dim=embed_dim, stride=4)

    def forward(self, x):
        # x: [B, 250, 64]  (Time × Channels format from DataLoader)
        x = x.permute(0, 2, 1)          # [B, 64, 250]  (conv format: Channels × Time)
        features = self.multi_scale(x)  # [B, 128, 250]
        tokens   = self.tokenizer(features)  # [B, 63, embed_dim]
        return tokens


if __name__ == "__main__":
    import sys

    print("\n" + "=" * 60)
    print("  PHASE 3 — MULTI-SCALE EEG ENCODER VERIFICATION")
    print("=" * 60)

    model = EEGEncoder(embed_dim=224)  # use actual training embed_dim
    model.eval()

    # Real trial shape from DataLoader: [Batch, Time=250, Channels=64]
    dummy = torch.randn(4, 250, 64)

    with torch.no_grad():
        out = model(dummy)

    # Expected: [4, 63, 224]  (63 tokens after padding fix, embed_dim=224)
    shape_ok = out.shape == torch.Size([4, 63, 224])
    print(f"\n[1] Input  shape  : {tuple(dummy.shape)}")
    print(f"    Output shape  : {tuple(out.shape)}  {'✅' if shape_ok else '❌ (expected [4, 63, 224])'} ")

    no_nan = not out.isnan().any().item()
    no_inf = not out.isinf().any().item()
    print(f"[2] NaN in output : {'None ✅' if no_nan else 'FOUND ❌'}")
    print(f"    Inf in output : {'None ✅' if no_inf else 'FOUND ❌'}")

    print(f"\n  Kernel temporal coverage at 250Hz:")
    print(f"    small_scale  : kernel=25 → {25*4}ms  (C1 early visual response)")
    print(f"    medium_scale : kernel=51 → {51*4}ms  (N170 object recognition)")
    print(f"    large_scale  : kernel=75 → {75*4}ms  (P300 cognitive categorization)")
    print(f"\n  Token count: 63  (all 250 samples covered, last 2 no longer discarded)")

    all_ok = shape_ok and no_nan and no_inf
    print(f"\n{'✅ Phase 3 verified — ready for Phase 4!' if all_ok else '❌ Issues found — review above'}")
    print("=" * 60 + "\n")