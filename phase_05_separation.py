"""Phase 5: Feature Separation — 6-Branch Disentanglement.

Takes the shared embedding [B, 63, token_dim] from Phase 4 and extracts
6 specialized latent representations, each capturing a different aspect
of the brain-image relationship.

KEY DESIGN: LearnablePool (per-branch soft attention pooling)
─────────────────────────────────────────────────────────────
Each branch has its own small attention layer that learns which of the
63 tokens to focus on. This replaces hardcoded time-window slicing.

WHY LearnablePool instead of hardcoded windows:
  - We don't know the exact stimulus-onset position inside the 1-second
    epoch without checking the raw data header (it could be -200ms to +800ms,
    or 0ms to 1000ms — these give completely different token indices for N170).
  - Individual subjects have slightly different ERP latencies (±30ms).
  - Hardcoded boundaries are guesses. Learnable boundaries are learned from data.
  - The model will naturally attend to P300-range tokens for the object branch
    (because category loss forces it to) — no manual specification needed.

After training, call model.get_attention_weights(x) to visualize which
tokens each branch learned to focus on (should correlate with known ERPs).

Branch output dimensions:
  object     : token_dim → object_dim     (default 128)
  temporal   : token_dim → temporal_dim   (default 128)
  spatial    : token_dim → spatial_dim    (default 64)
  subject    : token_dim → subject_dim    (default 64)
  view       : token_dim → view_dim       (default 64)
  appearance : token_dim → appearance_dim (default 768, matches CLIP ViT-L/14)

To switch to DINOv2: change appearance_dim = 768 → 1024.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Helper 1: Learnable soft-attention pooling ─────────────────────────────────
class LearnablePool(nn.Module):
    """Per-branch soft attention pooling over the 63 token sequence.

    Learns which EEG time tokens matter for this specific branch's task.
    Replaces hardcoded time-window slicing (which required knowing the exact
    stimulus-onset position inside the epoch — an assumption we cannot make).

    Architecture:
        Input  : [B, T, D]   (63 tokens, token_dim dimensions)
        Score  : Linear(D→1) → [B, T, 1]  (one score per token)
        Weight : Softmax over T → [B, T, 1]  (attention distribution, sums to 1)
        Output : weighted sum over T → [B, D]

    Initialization:
        Weights and bias start at zero → uniform attention at init (all tokens
        equally weighted). This is the correct neutral starting point — the model
        learns to focus on specific tokens as training progresses.

    After training, the attention weights reveal which EEG time windows
    each branch cares about. object_branch should focus on late tokens
    (P300 window), spatial_branch on early tokens (C1 window), etc.
    """
    def __init__(self, token_dim: int):
        super().__init__()
        self.attn = nn.Linear(token_dim, 1, bias=True)
        # Start with uniform attention: zero weights → all tokens get equal score
        # → softmax gives 1/T to each token → equal weighted average at init
        nn.init.zeros_(self.attn.weight)
        nn.init.zeros_(self.attn.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        scores  = self.attn(x)                    # [B, T, 1]  — one score per token
        weights = torch.softmax(scores, dim=1)    # [B, T, 1]  — sums to 1 over T
        pooled  = (x * weights).sum(dim=1)        # [B, D]     — learned weighted avg
        return pooled


# ── Helper 2: One branch MLP ───────────────────────────────────────────────────
class BranchMLP(nn.Module):
    """Small 2-layer MLP for one feature branch.

    Input : [B, token_dim]   pooled embedding from LearnablePool
    Output: [B, out_dim]     L2-normalized latent on the unit sphere

    Architecture: Linear(in → 2×in) → GELU → Dropout → Linear(2×in → out_dim)
    """
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        hidden = in_dim * 2

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        # Bug #7 fix: mean subtraction removed here (see separate fix in Bug #7)
        return F.normalize(out, dim=-1)   # L2 normalize → unit sphere


# ── Main: Feature Separation Module ───────────────────────────────────────────
class EEGFeatureSeparation(nn.Module):
    """6-branch feature separation for EEG → image generation.

    Each branch = LearnablePool (temporal selection) + BranchMLP (projection).

    LearnablePool: learns WHICH tokens (time points) to attend to from data.
    BranchMLP:     learns HOW to project the pooled signal to the latent space.

    Per-branch dimensions:
      - Object/Temporal : 128  (complex tasks, many categories)
      - Spatial/Subject/View : 64  (simpler, fewer classes)
      - Appearance : 512  (must match visual encoder — CLIP=512, DINOv2=1024)
    """

    def __init__(
        self,
        token_dim:      int   = 256,
        object_dim:     int   = 128,   # 72 categories — contrastive sweet spot
        temporal_dim:   int   = 128,   # continuous dynamics
        spatial_dim:    int   = 64,    # few spatial zones
        subject_dim:    int   = 64,    # 12 subjects
        view_dim:       int   = 64,    # few viewpoints
        appearance_dim: int   = 768,   # ViT-L/14=768 | DINOv2=1024 (NOT ViT-B/32=512)
        dropout:        float = 0.1,
    ):
        super().__init__()

        # ── Per-branch learnable pooling (one per branch) ──────────────────────
        # Each pool learns independently which tokens to attend to.
        # object_pool  → will learn to focus on late (P300) tokens for categorization
        # spatial_pool → will learn to focus on early (C1) tokens for spatial info
        # All start with uniform attention and specialise during training.
        self.object_pool      = LearnablePool(token_dim)
        self.temporal_pool    = LearnablePool(token_dim)
        self.spatial_pool     = LearnablePool(token_dim)
        self.subject_pool     = LearnablePool(token_dim)
        self.view_pool        = LearnablePool(token_dim)
        self.appearance_pool  = LearnablePool(token_dim)

        # ── Per-branch projection MLPs ─────────────────────────────────────────
        self.object_branch     = BranchMLP(token_dim, object_dim,     dropout)
        self.temporal_branch   = BranchMLP(token_dim, temporal_dim,   dropout)
        self.spatial_branch    = BranchMLP(token_dim, spatial_dim,    dropout)
        self.subject_branch    = BranchMLP(token_dim, subject_dim,    dropout)
        self.view_branch       = BranchMLP(token_dim, view_dim,       dropout)
        self.appearance_branch = BranchMLP(token_dim, appearance_dim, dropout)

        # Store dims for downstream access
        self.dims = {
            "object"    : object_dim,
            "temporal"  : temporal_dim,
            "spatial"   : spatial_dim,
            "subject"   : subject_dim,
            "view"      : view_dim,
            "appearance": appearance_dim,
        }

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x : [B, T, D]  shared embedding from Phase 4
                            B = batch, T = 63 tokens, D = token_dim

        Returns:
            dict of 6 L2-normalized latent tensors, each on the unit sphere
        """
        # Each branch independently attends to the tokens most relevant for it.
        # The attention weights are learned — not hardcoded — so the model
        # discovers the right EEG time windows from data.
        return {
            "object"    : self.object_branch    (self.object_pool(x)),      # [B, object_dim]
            "temporal"  : self.temporal_branch  (self.temporal_pool(x)),    # [B, temporal_dim]
            "spatial"   : self.spatial_branch   (self.spatial_pool(x)),     # [B, spatial_dim]
            "subject"   : self.subject_branch   (self.subject_pool(x)),     # [B, subject_dim]
            "view"      : self.view_branch       (self.view_pool(x)),        # [B, view_dim]
            "appearance": self.appearance_branch (self.appearance_pool(x)),  # [B, appearance_dim]
        }

    def get_attention_weights(self, x: torch.Tensor) -> dict:
        """Post-training diagnostic: which tokens does each branch attend to?

        Call this after training to verify the branches learned meaningful
        temporal specialisation. Expected (for THINGS-EEG data):
          object     → high attention on late tokens (P300 window, ~300ms post-onset)
          spatial    → high attention on early tokens (C1 window, ~80ms post-onset)
          temporal   → spread across all tokens (no strong preference)
          subject    → spread across all tokens (subject style throughout)

        Args:
            x : [B, T, D]  a batch of real EEG token sequences

        Returns:
            dict of attention weight tensors, shape [T] each (averaged over batch)
        """
        weights = {}
        with torch.no_grad():
            for name, pool in [
                ("object",     self.object_pool),
                ("temporal",   self.temporal_pool),
                ("spatial",    self.spatial_pool),
                ("subject",    self.subject_pool),
                ("view",       self.view_pool),
                ("appearance", self.appearance_pool),
            ]:
                scores = pool.attn(x)                    # [B, T, 1]
                w      = torch.softmax(scores, dim=1)    # [B, T, 1]
                weights[name] = w.mean(dim=0).squeeze()  # [T]  — mean over batch
        return weights


# ── Verification ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  PHASE 5 — FEATURE SEPARATION VERIFICATION")
    print("=" * 60)

    TOKEN_DIM = 192   # matches new training config (3 heads × 64-dim)

    model = EEGFeatureSeparation(
        token_dim      = TOKEN_DIM,
        object_dim     = 128,
        temporal_dim   = 128,
        spatial_dim    = 64,
        subject_dim    = 64,
        view_dim       = 64,
        appearance_dim = 768,   # ViT-L/14 — must match Phase 6 CLIP model
    )
    model.eval()

    # Real input shape from Phase 4: [Batch=64, Tokens=63, Dim=192]
    dummy = torch.randn(64, 63, TOKEN_DIM)

    with torch.no_grad():
        latents = model(dummy)

    # ── CHECK 1: Output shapes ─────────────────────────────────────────────────
    expected = {
        "object"    : (64, 128),
        "temporal"  : (64, 128),
        "spatial"   : (64, 64),
        "subject"   : (64, 64),
        "view"      : (64, 64),
        "appearance": (64, 768),  # ViT-L/14 = 768 (Bug #8 fix: was 512 for ViT-B/32)
    }\

    print(f"\n[1] Output shapes (6 branches):")
    all_shapes_ok = True
    for name, tensor in latents.items():
        ok = tuple(tensor.shape) == expected[name]
        print(f"    {name:<12}: {str(tuple(tensor.shape)):<12}  expected {expected[name]}  {'✅' if ok else '❌'}")
        all_shapes_ok = all_shapes_ok and ok

    # ── CHECK 2: L2 norm = 1.0 (unit sphere) ──────────────────────────────────
    print(f"\n[2] L2 norm check (all should be 1.0 — unit sphere):")
    all_norms_ok = True
    for name, tensor in latents.items():
        mean_norm = tensor.norm(dim=-1).mean().item()
        ok = abs(mean_norm - 1.0) < 1e-4
        print(f"    {name:<12}: mean norm = {mean_norm:.6f}  {'✅' if ok else '❌'}")
        all_norms_ok = all_norms_ok and ok

    # ── CHECK 3: No NaN / Inf ──────────────────────────────────────────────────
    print(f"\n[3] NaN / Inf check:")
    all_clean = True
    for name, tensor in latents.items():
        ok = not tensor.isnan().any() and not tensor.isinf().any()
        print(f"    {name:<12}: {'✅ Clean' if ok else '❌ NaN/Inf found'}")
        all_clean = all_clean and ok

    # ── CHECK 4: Branch diversity (branches must differ from each other) ───────
    print(f"\n[4] Branch diversity (same-dim pairs must NOT be identical):")
    all_diverse = True
    keys = list(latents.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = latents[keys[i]], latents[keys[j]]
            if a.shape == b.shape:
                diff = (a - b).abs().mean().item()
                ok = diff > 1e-4
                print(f"    {keys[i]} vs {keys[j]}: diff = {diff:.6f}  {'✅' if ok else '❌ identical!'}")
                all_diverse = all_diverse and ok

    # ── CHECK 5: Attention weights start uniform (correct init check) ──────────
    print(f"\n[5] Attention weight uniformity at init (should all be 1/63 = 0.0159):")
    attn_ok = True
    with torch.no_grad():
        for name, pool in [
            ("object", model.object_pool), ("spatial", model.spatial_pool)
        ]:
            scores  = pool.attn(dummy)
            weights = torch.softmax(scores, dim=1).squeeze(-1)   # [B, T]
            mean_w  = weights.mean().item()
            std_w   = weights.std().item()
            ok = abs(mean_w - 1/63) < 0.005
            print(f"    {name:<12}: mean={mean_w:.4f} (target {1/63:.4f}), std={std_w:.6f}  {'✅' if ok else '❌'}")
            attn_ok = attn_ok and ok

    # ── CHECK 6: Parameter count ───────────────────────────────────────────────
    print(f"\n[6] Parameter count per branch (pool + MLP):")
    total = 0
    for name in ["object", "temporal", "spatial", "subject", "view", "appearance"]:
        pool  = getattr(model, f"{name}_pool")
        mlp   = getattr(model, f"{name}_branch")
        count = sum(p.numel() for p in pool.parameters()) + \
                sum(p.numel() for p in mlp.parameters())
        total += count
        pool_p = sum(p.numel() for p in pool.parameters())
        mlp_p  = sum(p.numel() for p in mlp.parameters())
        print(f"    {name:<12}: pool={pool_p:>5,}  mlp={mlp_p:>7,}  total={count:>8,}")
    print(f"    {'TOTAL':<12}: {total:>8,} params")

    all_ok = all_shapes_ok and all_norms_ok and all_clean and all_diverse and attn_ok
    print(f"\n{'✅ Phase 5 verified — ready for Phase 6 training!' if all_ok else '❌ Issues found — review above'}")
    print("=" * 60 + "\n")
