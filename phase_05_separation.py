"""Phase 5: Feature Separation — 6-Branch Disentanglement.

Takes the shared embedding [B, 62, 256] from Phase 4 and splits it into
6 specialized latent representations, each capturing a different aspect
of the brain-image relationship.

Design decisions:
  - Mean pooling: collapse 62 tokens → [B, 256] (safe, stable, tokens already context-aware)
  - 6 separate 2-layer MLPs: one per feature type
  - L2 normalization on all outputs: unit sphere latents for stable Phase 6 losses
  - Per-branch dimensions tuned to the complexity of each branch's task:
      object     : 128  — 80 categories, contrastive proven sweet spot
      temporal   : 128  — continuous dynamics, bump to 192 if underfitting
      spatial    : 64   — few distinct spatial zones
      subject    : 64   — 12 subjects, generous for individual differences
      view       : 64   — few distinct viewpoints
      appearance : 512  — matches CLIP (change to 1024 for DINOv2)

Flow:
    Shared Embedding [B, 62, 256]
        ↓ mean pool across 62 tokens
    [B, 256]
        ↓ 6 separate MLPs (256 → 512 → out_dim)
    {
        object     : [B, 128]  — what object the subject saw
        temporal   : [B, 128]  — how brain response evolved over 1 second
        spatial    : [B, 64]   — where in the visual field the object was
        subject    : [B, 64]   — identity of the subject (sponge for brain bias)
        view       : [B, 64]   — angle / perspective of the object
        appearance : [B, 512]  — visual features aligned to CLIP / DINOv2
    }

To switch from CLIP to DINOv2: change appearance_dim = 512 → 1024. Nothing else changes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Helper: One branch MLP ─────────────────────────────────────────────────────
class BranchMLP(nn.Module):
    """Small 2-layer MLP for one feature branch.

    Args:
        in_dim  : input dimension (= token_dim from Phase 4, default 256)
        out_dim : latent dimension for this branch
        dropout : dropout probability
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        hidden = in_dim * 2   # 2× expansion: 256 → 512

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),   # 256 → 512
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),  # 512 → out_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x   : [B, in_dim]   pooled embedding
        Returns:
            out : [B, out_dim]  L2-normalized latent on unit sphere
        """
        out = self.net(x)
        # Prevent initialization bias by zero-centering the features across the batch
        if out.shape[0] > 1:
            out = out - out.mean(dim=0, keepdim=True)
        return F.normalize(out, dim=-1)   # L2 normalize → unit sphere



# ── Main: Feature Separation Module ───────────────────────────────────────────
class EEGFeatureSeparation(nn.Module):
    """6-branch feature separation for EEG → image generation.

    Per-branch dimensions are tuned to the complexity of each task:
      - Object/Temporal : 128  (complex, continuous, or many categories)
      - Spatial/Subject/View : 64  (simple discrete zones / few classes)
      - Appearance : 512  (must match visual encoder — CLIP=512, DINOv2=1024)

    Args:
        token_dim      : embedding dimension from Phase 4 (default: 256)
        object_dim     : latent dim for Object branch (default: 128)
        temporal_dim   : latent dim for Temporal branch (default: 128)
        spatial_dim    : latent dim for Spatial branch (default: 64)
        subject_dim    : latent dim for Subject branch (default: 64)
        view_dim       : latent dim for View branch (default: 64)
        appearance_dim : latent dim for Appearance branch (default: 512 = CLIP)
        dropout        : dropout probability (default: 0.1)
    """

    def __init__(
        self,
        token_dim:      int   = 256,
        object_dim:     int   = 128,   # 80 categories — contrastive sweet spot
        temporal_dim:   int   = 128,   # continuous dynamics — bump to 192 if needed
        spatial_dim:    int   = 64,    # few spatial zones — 128 would be redundant
        subject_dim:    int   = 64,    # 12 subjects — generous, not wasteful
        view_dim:       int   = 64,    # few viewpoints — 128 would be redundant
        appearance_dim: int   = 512,   # CLIP=512 | DINOv2=1024
        dropout:        float = 0.1,
    ):
        super().__init__()

        # ── 6 branches with tuned dimensions ──────────────────────────────────
        self.object_branch     = BranchMLP(token_dim, object_dim,     dropout)
        self.temporal_branch   = BranchMLP(token_dim, temporal_dim,   dropout)
        self.spatial_branch    = BranchMLP(token_dim, spatial_dim,    dropout)
        self.subject_branch    = BranchMLP(token_dim, subject_dim,    dropout)
        self.view_branch       = BranchMLP(token_dim, view_dim,       dropout)
        self.appearance_branch = BranchMLP(token_dim, appearance_dim, dropout)

        # Store dims for verification and downstream access
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
                            B = batch, T = 62 tokens, D = 256

        Returns:
            dict of 6 L2-normalized latent tensors, each on the unit sphere
        """
        # Step 1: Mean pool across 62 tokens → collapse temporal dimension
        # Tokens are already context-aware (Phase 4 attention) so mean pooling
        # does not lose information — every token already knows about all others
        pooled = x.mean(dim=1)          # [B, 62, 256] → [B, 256]

        # Step 2: Each branch extracts its specialized feature
        return {
            "object"    : self.object_branch(pooled),     # [B, 128]
            "temporal"  : self.temporal_branch(pooled),   # [B, 128]
            "spatial"   : self.spatial_branch(pooled),    # [B, 64]
            "subject"   : self.subject_branch(pooled),    # [B, 64]
            "view"      : self.view_branch(pooled),       # [B, 64]
            "appearance": self.appearance_branch(pooled), # [B, 512]
        }


# ── Verification ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  PHASE 5 — FEATURE SEPARATION VERIFICATION")
    print("=" * 60)

    model = EEGFeatureSeparation(
        token_dim      = 256,
        object_dim     = 128,
        temporal_dim   = 128,
        spatial_dim    = 64,
        subject_dim    = 64,
        view_dim       = 64,
        appearance_dim = 512,   # CLIP — change to 1024 for DINOv2
    )
    model.eval()

    # Real input shape from Phase 4: [Batch=64, Tokens=62, Dim=256]
    dummy = torch.randn(64, 62, 256)

    with torch.no_grad():
        latents = model(dummy)

    # ── CHECK 1: Output shapes ─────────────────────────────────────────────────
    expected = {
        "object"    : (64, 128),
        "temporal"  : (64, 128),
        "spatial"   : (64, 64),
        "subject"   : (64, 64),
        "view"      : (64, 64),
        "appearance": (64, 512),
    }

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

    # ── CHECK 4: Branch diversity (same-dim branches must differ) ─────────────
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

    # ── CHECK 5: Parameter count per branch ────────────────────────────────────
    print(f"\n[5] Parameter count per branch:")
    branch_map = {
        "object"    : model.object_branch,
        "temporal"  : model.temporal_branch,
        "spatial"   : model.spatial_branch,
        "subject"   : model.subject_branch,
        "view"      : model.view_branch,
        "appearance": model.appearance_branch,
    }
    total = 0
    for name, branch in branch_map.items():
        count = sum(p.numel() for p in branch.parameters())
        total += count
        print(f"    {name:<12}: {count:>8,} params")
    print(f"    {'TOTAL':<12}: {total:>8,} params")

    all_ok = all_shapes_ok and all_norms_ok and all_clean and all_diverse
    print(f"\n{'✅ Phase 5 verified — ready for Phase 6 training!' if all_ok else '❌ Issues found — review above'}")
    print("=" * 60 + "\n")
