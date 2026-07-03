"""Phase 4: Lightweight Attention Mechanism.

Takes the token sequence [B, 63, token_dim] from Phase 3 (MultiScaleEEGEncoder)
and applies 2-layer Multi-Head Self-Attention to produce a refined
shared embedding where every token is context-aware of all other tokens.

This shared embedding is then passed to Phase 5 for 6-branch feature separation.

Architecture (with actual training config token_dim=192, num_heads=3):
    Input [B, 63, 192]
      + Learnable Positional Embeddings [1, 63, 192]
      ↓ Transformer Layer 1: MHSA (3 heads × 64 dim) + FFN (192→384→192) + PreLayerNorm
      ↓ Transformer Layer 2: MHSA (3 heads × 64 dim) + FFN (192→384→192) + PreLayerNorm
      ↓ Final LayerNorm
    Output [B, 63, 192]  ← same shape, every token is now context-aware

Note: token count is 63 (not 62) after Phase 3 Tokenizer padding fix.
Note: 192/3 = 64-dim per head (standard). Smaller than 256 to reduce overfit risk.
"""

import torch
import torch.nn as nn


class EEGTransformerEncoder(nn.Module):
    """Lightweight 2-layer Transformer Encoder for EEG token sequences.

    Args:
        token_dim  : dimension of each input token        (actual training default: 192)
        num_heads  : number of attention heads             (actual training default: 3  → 3×64-dim heads)
        ff_dim     : feed-forward hidden dimension         (actual training default: 384 → 2×token_dim)
        num_layers : number of stacked transformer layers  (actual training default: 2)
        dropout    : dropout probability                   (actual training default: 0.32)
        max_tokens : number of learnable position slots    (default: 63 — after Bug #18 tokenizer padding fix;
                                                            was 62 before padding=1 was added to Tokenizer)
    """

    def __init__(
        self,
        token_dim: int = 256,
        num_heads: int = 4,
        ff_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_tokens: int = 63,   # 63 tokens from Phase 3 Tokenizer (padding=1 fix)
    ):
        super().__init__()

        # ── Learnable Positional Embeddings ───────────────────────────────
        # Shape [1, 63, token_dim] — one embedding per token position.
        # Broadcast across batches automatically.
        # Tells the model which token came first, second, ... 63rd in time.
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_tokens, token_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)  # small random init (ViT-style)

        # ── Transformer Encoder Layers ────────────────────────────────────
        # Each layer contains (with actual training config token_dim=192, num_heads=3):
        #   1. Multi-Head Self-Attention (3 heads × 64 dim = 192 total)  ← standard 64-dim heads
        #   2. Feed-Forward MLP         (192 → ff_dim → 192)
        #   3. LayerNorm + Residual     (applied before each sub-layer — Pre-LN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",    # consistent with ConvBlock activation in Phase 3
            batch_first=True,     # our format is [B, T, D] — not PyTorch default [T, B, D]
            norm_first=True,      # Pre-LayerNorm: more stable training on small datasets
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(token_dim),   # final LayerNorm after all layers complete
        )

        # NOTE: No external dropout here.
        # TransformerEncoderLayer already applies dropout internally:
        #   - After attention softmax weights
        #   - After attention projection output
        #   - After FFN first linear layer
        # That is 3× per layer, 6× total for 2 layers.
        # Adding an external Dropout on top would compound to 7 passes at p=0.32,
        # leaving only ~7% of token values active during training but 100% at inference
        # (14× information density shift → model generates noise at generation time).

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x   : [B, T, D]  token sequence from Phase 3
                   B = batch, T = 63 tokens (after Phase 3 padding fix), D = token_dim

        Returns:
            out : [B, T, D]  same shape — every token is now context-aware
        """
        # Step 1: inject positional information into each token
        # Without this, the Transformer treats tokens as an unordered set
        x = x + self.pos_embedding[:, :x.size(1), :]

        # Step 2: pass through 2 Transformer layers
        # Each layer applies dropout internally (after attention + after FFN)
        # 3× per layer × 2 layers = 6 total dropout passes at p=0.32
        out = self.transformer(x)

        return out


# ── Verification ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  PHASE 4 — ATTENTION MECHANISM VERIFICATION")
    print("=" * 60)

    model = EEGTransformerEncoder(token_dim=192, num_heads=3, ff_dim=384, num_layers=2, dropout=0.32)
    model.eval()

    # Real input from Phase 3: [Batch=64, Tokens=63, Dim=192]  (token_dim=192, 3 heads×64-dim)
    dummy = torch.randn(64, 63, 192)

    with torch.no_grad():
        out = model(dummy)

    # ── CHECK 1: Shape preserved ───────────────────────────────────────────
    shape_ok = out.shape == dummy.shape
    print(f"\n[1] Input  shape : {tuple(dummy.shape)}")
    print(f"    Output shape : {tuple(out.shape)}  {'✅' if shape_ok else '❌'}")

    # ── CHECK 2: dtype ─────────────────────────────────────────────────────
    dtype_ok = out.dtype == torch.float32
    print(f"[2] dtype        : {out.dtype}   {'✅' if dtype_ok else '❌'}")

    # ── CHECK 3: No NaN / Inf ──────────────────────────────────────────────
    nan_ok = not out.isnan().any().item()
    inf_ok = not out.isinf().any().item()
    print(f"[3] NaN in output: {'None ✅' if nan_ok else 'FOUND ❌'}")
    print(f"    Inf in output: {'None ✅' if inf_ok else 'FOUND ❌'}")

    # ── CHECK 4: Output is actually different from input ───────────────────
    # If the model is doing nothing, output == input
    mean_diff = (out - dummy).abs().mean().item()
    changed = mean_diff > 1e-4
    print(f"[4] Mean abs diff (input vs output): {mean_diff:.6f}  {'✅ model is transforming data' if changed else '❌ output unchanged'}")

    # ── CHECK 5: Positional embeddings shape ───────────────────────────────
    pos_shape_ok = model.pos_embedding.shape == (1, 63, 192)
    print(f"[5] Positional embedding shape: {tuple(model.pos_embedding.shape)}  {'✅' if pos_shape_ok else '❌'}")

    # ── Parameter count ────────────────────────────────────────────────────
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n    Trainable params: {total_params:,}")

    all_ok = shape_ok and dtype_ok and nan_ok and inf_ok and changed and pos_shape_ok
    print(f"\n{'✅ Phase 4 verified — ready for Phase 5!' if all_ok else '❌ Issues found — review above'}")
    print("=" * 60 + "\n")
