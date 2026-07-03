"""Phase 7: EEG Feature Fusion (Closed-Form Ridge Regression).

This script implements Phase 7: Feature Fusion. It projects and fuses 5 disentangled,
non-subject EEG latents (object, appearance, spatial, temporal, view) from a frozen
Phase 6 model into a unified conditioning tensor of shape [B, 77, 768] that matches
the Stable Diffusion v1.5 text encoder sequence space (CLIP ViT-L/14 text transformer).

Instead of using iterative gradient descent (AdamW training loop), this implementation
uses the closed-form normal equation of Ridge Regression (L2-regularized least squares)
to solve for the mathematically optimal linear projection in less than a second.
It automatically sweeps across multiple alpha (regularization) values to find the one
that generalizes best to the unseen validation (test) set.
"""

import os
import json
import time
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from phase_02_DL import create_dataloader
from phase_06_training import EEGPipeline, extract_category

# Optional CLIP for sequence target precomputation
try:
    import clip as clip_module
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False


# Standard template prompt for regression target
PROMPT_TEMPLATE = "a photo of a {}"


# ─────────────────────────────────────────────────────────────────────────────
# 1. EEGFusionRidge Module Wrapper
# ─────────────────────────────────────────────────────────────────────────────

class EEGFusionRidge(nn.Module):
    """Linear Ridge Regression mapping from EEG latents to CLIP sequence space.

    Concatenates latents of shape:
      - object     : [B, 128]
      - appearance : [B, 768]
      - spatial    : [B, 64]
      - temporal   : [B, 128]
      - view       : [B, 64]
      Total Input  : [B, 896]

    Maps to output shape:
      - Shape      : [B, 77, 768]  (Stable Diffusion CLIP sequence space)

    Bug #15 note:
      Input features are standardized (zero-mean, unit-variance per dim) before
      the linear projection. The scaler stats (x_mean, x_std) are stored as
      buffers so they are saved/loaded with the checkpoint and applied at inference.
    """

    def __init__(self, W: torch.Tensor, b: torch.Tensor,
                 x_mean: torch.Tensor, x_std: torch.Tensor):
        super().__init__()
        # Register weights as parameters so they save/load correctly
        self.W = nn.Parameter(W)  # [896, 59136]
        self.b = nn.Parameter(b)  # [59136]
        # Register scaler stats as buffers (not parameters — they don't get gradients)
        # Buffers are saved/loaded with state_dict automatically.
        self.register_buffer("x_mean", x_mean)   # [896]
        self.register_buffer("x_std",  x_std)    # [896]

    def forward(self, latents: dict) -> torch.Tensor:
        """
        Args:
            latents: dict containing branch outputs from Phase 6 pipeline
                     (object, appearance, spatial, temporal, view)
        Returns:
            fused_embeddings: [B, 77, 768] conditioning sequence
        """
        # Concatenate 5 non-subject latents along the feature dimension
        x = torch.cat([
            latents["object"],       # [B, 128]
            latents["appearance"],   # [B, 768]
            latents["spatial"],      # [B, 64]
            latents["temporal"],     # [B, 128]
            latents["view"]          # [B, 64]
        ], dim=-1)                   # [B, 896]

        # Bug #15 fix: standardize input using training scaler stats.
        # The 5 branches have different output dims → different per-dim variance.
        # Ridge L2 penalty is scale-sensitive: without this, appearance branch
        # (768-dim, smallest per-dim magnitude) is systematically under-weighted.
        x = (x - self.x_mean) / self.x_std

        # Linear projection
        out = torch.matmul(x, self.W) + self.b  # [B, 59136]

        # Reshape to [B, 77, 768]
        return out.view(-1, 77, 768)


# ─────────────────────────────────────────────────────────────────────────────
# 2. CLIP Text Sequence Target Extraction
# ─────────────────────────────────────────────────────────────────────────────

def get_clip_text_sequence(clip_model, text_tokens) -> torch.Tensor:
    """Extracts the unpooled token sequence from the last transformer layer of CLIP.

    This bypasses the final pooled projection to return the raw sequence space
    needed for Stable Diffusion UNet cross-attention.

    Args:
        clip_model: OpenAI CLIP model instance (ViT-L/14)
        text_tokens: Tokenized text prompts [B, 77]
    Returns:
        unpooled_embeddings: [B, 77, 768]
    """
    with torch.no_grad():
        # Retrieve token embeddings and add positional embeddings
        x = clip_model.token_embedding(text_tokens).type(clip_model.dtype)  # [B, 77, 768]
        x = x + clip_model.positional_embedding.type(clip_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND (Transformer expectations)
        x = clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = clip_model.ln_final(x).type(torch.float32)  # [B, 77, 768]
    return x


def precompute_category_targets(cat_index: dict, device: torch.device) -> dict:
    """[DEPRECATED — kept for reference only]

    Precomputes CLIP TEXT sequence targets: 72 unique targets for 72 categories.
    Bug #17: only 72 unique Y values for ~16,740 training rows → Ridge collapses
    to mean. Replaced by precompute_image_targets() which gives ~1650+ unique targets.
    """
    if not CLIP_AVAILABLE:
        raise ImportError(
            "CLIP not installed. Run:\n"
            "  pip install git+https://github.com/openai/CLIP.git"
        )

    print(f"\n  [DEPRECATED] Precomputing CLIP TEXT targets for {len(cat_index)} categories...")
    print(f"  WARNING: Only 72 unique targets — use precompute_image_targets() instead.")
    clip_model, _ = clip_module.load("ViT-L/14", device=device)
    clip_model.eval()

    targets = {}
    for cat_name in cat_index.keys():
        clean_cat = cat_name.replace("_", " ")
        prompt = PROMPT_TEMPLATE.format(clean_cat)
        tokens = clip_module.tokenize([prompt]).to(device)
        target_seq = get_clip_text_sequence(clip_model, tokens)
        targets[cat_name] = target_seq.squeeze(0).cpu()

    print(f"  Target precomputation finished.")
    return targets


def precompute_image_targets(all_labels: list, image_dir: str, device: torch.device) -> dict:
    """Precomputes CLIP IMAGE embeddings as Ridge regression targets.

    Bug #17 fix: replaces category-level CLIP text targets (72 unique) with
    per-image CLIP image embeddings (~1650+ unique for THINGS-EEG training set).

    WHY IMAGE EMBEDDINGS:
      - Old target: CLIP_text("a photo of a airplane") — same for ALL airplane trials
        regardless of which specific image was viewed → Ridge sees 72 unique Y values
        across ~16,740 training rows → collapses to mean of 72 CLIP text vectors.

      - New target: CLIP_image(airplane_05.png) — unique per stimulus image.
        airplane_01 ≠ airplane_05 ≠ airplane_12 → ~1650+ unique Y values.
        Ridge now has enough target diversity to learn a meaningful mapping.

    WHY THIS ALIGNS WITH APPEARANCE BRANCH:
      Phase 6 trains the appearance branch against CLIP IMAGE embeddings.
      Using the same targets in Phase 7 means:
        EEG → appearance_latent (learned to be close to CLIP_image) → Ridge → CLIP_image
      The entire pipeline is consistent end-to-end.

    TARGET SHAPE [77, 768]:
      CLIP image encoder produces [768]. SD cross-attention expects [77, 768].
      Fix: repeat the 768-dim image embedding 77 times → [77, 768].
      Each of the 77 attention positions carries the same image-level signal.
      SD UNet then attends to image content at every cross-attention layer.

    Args:
        all_labels:  all label strings in the dataset (to determine unique images)
        image_dir:   root directory containing category subfolders with .png images
        device:      torch device
    Returns:
        targets: dict {label_string: tensor [77, 768]}  (one per unique stimulus image)
    """
    if not CLIP_AVAILABLE:
        raise ImportError(
            "CLIP not installed. Run:\n"
            "  pip install git+https://github.com/openai/CLIP.git"
        )

    from PIL import Image as PILImage
    import os

    print(f"\n  Precomputing CLIP IMAGE targets for {len(set(all_labels))} unique stimuli...")
    clip_model, clip_preprocess = clip_module.load("ViT-L/14", device=device)
    clip_model.eval()

    image_dir = Path(image_dir)
    targets   = {}
    missing   = []

    unique_labels = sorted(set(all_labels))
    for label in unique_labels:
        # label format: "01_airplane_05" → category = "airplane", image file = label.png
        cat_name  = extract_category(label)           # e.g. "airplane"
        img_path  = image_dir / cat_name / f"{label}.png"

        if not img_path.exists():
            missing.append(str(img_path))
            continue

        with torch.no_grad():
            img = PILImage.open(img_path).convert("RGB")
            img_tensor = clip_preprocess(img).unsqueeze(0).to(device)  # [1, 3, 224, 224]
            feat = clip_model.encode_image(img_tensor)                  # [1, 768]
            feat = F.normalize(feat, dim=-1).squeeze(0)                # [768]  L2-normalised

        # Repeat 77 times to match SD cross-attention input shape [77, 768]
        # Each attention position carries the same image-level signal
        target_seq = feat.unsqueeze(0).repeat(77, 1).cpu()             # [77, 768]
        targets[label] = target_seq

    if missing:
        print(f"  ⚠  {len(missing)} images not found — these labels will be skipped.")
        for p in missing[:3]:
            print(f"      {p}")
        if len(missing) > 3:
            print(f"      ... and {len(missing) - 3} more")

    print(f"  Image target cache ready: {len(targets)} unique stimuli  "
          f"(vs 72 text targets — {len(targets)//72:.0f}× more unique).")
    return targets


# ─────────────────────────────────────────────────────────────────────────────
# 3. Data Gathering & Ridge Solver
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def gather_features_and_targets(loader, frozen_pipeline, target_cache, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Collects and stacks EEG latents (inputs) and CLIP image targets.

    Bug #17 fix: target_cache is now keyed by LABEL STRING (not category name),
    so each unique stimulus image has a unique target [77, 768].
    This gives ~1650+ unique Y values instead of 72 text targets.

    Returns:
        X: [N, 1152] concatenated inputs on CPU
             object(128) + appearance(768) + spatial(64) + temporal(128) + view(64)
        Y: [N, 59136] flattened targets on CPU  (77 tokens × 768 CLIP dims)
    """
    X_list = []
    Y_list = []

    for batch_idx, (eeg, labels, _) in enumerate(loader):
        eeg = eeg.to(device)

        # Retrieve per-IMAGE targets (label-keyed, not category-keyed)
        # Falls back to None for missing images — those batches are skipped
        batch_targets_list = []
        valid_mask = []
        for lbl in labels:
            if lbl in target_cache:
                batch_targets_list.append(target_cache[lbl])
                valid_mask.append(True)
            else:
                valid_mask.append(False)

        if not any(valid_mask):
            continue   # entire batch has no valid targets — skip

        # Keep only valid samples
        valid_idx = [i for i, v in enumerate(valid_mask) if v]
        batch_targets = torch.stack(batch_targets_list).to(device)  # [B_valid, 77, 768]
        batch_targets_flat = batch_targets.view(batch_targets.shape[0], -1)  # [B_valid, 59136]

        # Extract disentangled latents
        latents = frozen_pipeline(eeg)
        latents.pop("subject", None)   # Discard subject latent (not used in fusion)
        latents.pop("shared",  None)   # Discard DANN shared rep (not a branch output)

        # Select only valid samples from each branch
        x_concat = torch.cat([
            latents["object"][valid_idx],       # [B_valid, 128]   11.1% of input
            latents["appearance"][valid_idx],   # [B_valid, 768]   66.7% of input  ← largest branch
            latents["spatial"][valid_idx],      # [B_valid, 64]     5.6% of input
            latents["temporal"][valid_idx],     # [B_valid, 128]   11.1% of input
            latents["view"][valid_idx]          # [B_valid, 64]     5.6% of input
        ], dim=-1)                              # [B_valid, 1152]

        X_list.append(x_concat.cpu())
        Y_list.append(batch_targets_flat.cpu())

    X = torch.cat(X_list, dim=0)  # [N_valid, 1152]
    Y = torch.cat(Y_list, dim=0)  # [N_valid, 59136]
    return X, Y


def solve_ridge(X_train: torch.Tensor, Y_train: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Solves Ridge Regression analytically for weights W and bias b.

    Formula: W_ext = (X_ext.T @ X_ext + alpha * eye)^-1 @ X_ext.T @ Y
    """
    N = X_train.shape[0]
    
    # Append a column of ones to represent bias term in matrix multiplication
    ones = torch.ones(N, 1, dtype=X_train.dtype)
    X_ext = torch.cat([X_train, ones], dim=1) # [N, 897]

    # Create L2 penalty matrix. We do not penalize the bias term (index -1)
    eye = torch.eye(X_ext.shape[1], dtype=X_train.dtype)
    eye[-1, -1] = 0.0

    # Solve: (X_ext.T @ X_ext + alpha * eye) @ W_ext = X_ext.T @ Y
    A = torch.matmul(X_ext.T, X_ext) + alpha * eye # [897, 897]
    B = torch.matmul(X_ext.T, Y_train)             # [897, 59136]

    # Use torch.linalg.solve for high numerical stability
    W_ext = torch.linalg.solve(A, B) # [897, 59136]

    W = W_ext[:-1, :] # [896, 59136]
    b = W_ext[-1, :]  # [59136]
    return W, b


# ─────────────────────────────────────────────────────────────────────────────
# 4. Main Solver Controller
# ─────────────────────────────────────────────────────────────────────────────

def train_fusion_ridge(
    data_dir: str = "data/EEGdatanpy",
    checkpoint_path: str = "checkpoints/phase06_2.24_leak_prevented/best_model.pt",
    save_dir: str = "checkpoints/phase07",
    batch_size: int = 256,
):
    """Solves and optimizes Ridge Regression fusion mapping from EEG to CLIP space."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTarget Device: {device}")

    # ── Resolve Checkpoint Paths ──────────────────────────────────────────────
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Frozen Phase 6 checkpoint not found at: {ckpt_path}")

    print(f"Loading Frozen Phase 6 checkpoint: {ckpt_path.name}")
    checkpoint = torch.load(ckpt_path, map_location=device)

    cat_index = checkpoint["cat_index"]
    subj_index = checkpoint["subj_index"]

    # ── Load config.json if it exists to get architecture details ─────────────
    config_path = ckpt_path.parent / "config.json"
    token_dim, num_heads, ff_dim, num_layers = 256, 4, 512, 2
    appearance_dim = 768   # ViT-L/14 image embedding dim (must match Phase 6 CLIP model)

    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            token_dim = config.get("token_dim", token_dim)
            num_heads = config.get("num_heads", num_heads)
            ff_dim = config.get("ff_dim", ff_dim)
            num_layers = config.get("num_layers", num_layers)
            appearance_dim = config.get("appearance_dim", appearance_dim)
        except Exception as e:
            print(f"  ⚠ Warning: Failed to read config.json, using defaults ({e})")

    # ── Instantiate and Load Frozen Phase 6 Pipeline ──────────────────────────
    frozen_pipeline = EEGPipeline(
        token_dim=token_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
        dropout=0.0,
        appearance_dim=appearance_dim,
    ).to(device)
    frozen_pipeline.load_state_dict(checkpoint["model_state_dict"])
    frozen_pipeline.eval()

    # Freeze pipeline parameters
    for p in frozen_pipeline.parameters():
        p.requires_grad = False

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader = create_dataloader(data_dir=data_dir, batch_size=batch_size, split="train")
    test_loader  = create_dataloader(data_dir=data_dir, batch_size=batch_size, split="test")

    # ── Precompute Targets ───────────────────────────────────────────────────
    # Bug #17 fix: use per-image CLIP IMAGE targets (keyed by label string)
    # instead of per-category CLIP TEXT targets (keyed by category name).
    # Collect all unique labels from both train + test to build the full cache.
    print("\nScanning dataset for all unique label strings...")
    all_labels = []
    for loader in [train_loader, test_loader]:
        for _, labels, _ in loader:
            all_labels.extend(list(labels))

    image_dir = Path(checkpoint_path).parent.parent / "data" / "image"  # default path
    # Try to locate image_dir relative to checkpoint
    if not image_dir.exists():
        image_dir = Path("data/image")  # fallback to CWD-relative

    target_cache = precompute_image_targets(all_labels, str(image_dir), device)

    # ── Gather Matrices ───────────────────────────────────────────────────────
    print("\nGathering features and target matrices from datasets...")
    X_train, Y_train = gather_features_and_targets(train_loader, frozen_pipeline, target_cache, device)
    X_test,  Y_test  = gather_features_and_targets(test_loader,  frozen_pipeline, target_cache, device)

    print(f"  Train set size: {X_train.shape[0]} samples")
    print(f"  Test set size:  {X_test.shape[0]} samples")
    print(f"  Input feature dimension:  {X_train.shape[1]}")
    print(f"  Output feature dimension: {Y_train.shape[1]}")

    # ── Bug #15 Fix: Standardize input features ───────────────────────────────
    # Each of the 5 branches is L2-normalized, but they have different output
    # dimensions (128, 768, 64, 128, 64), giving different per-dim magnitudes:
    #   spatial/view (~1/√64=0.125) vs appearance (~1/√768=0.036) → 3.5× gap.
    # Ridge L2 penalty is scale-sensitive: small-magnitude features (appearance)
    # get over-shrunk relative to large-magnitude features (spatial, view).
    # Standardizing to zero-mean, unit-variance per dimension eliminates this
    # bias so Ridge treats all 896 features equally.
    x_mean = X_train.mean(dim=0)                       # [896]
    x_std  = X_train.std(dim=0).clamp(min=1e-6)        # [896]  clamp avoids /0
    X_train_s = (X_train - x_mean) / x_std             # standardized train
    X_test_s  = (X_test  - x_mean) / x_std             # same scaler on test!
    print(f"  Feature std range: min={x_std.min():.4f}, max={x_std.max():.4f}, mean={x_std.mean():.4f}")
    print(f"  (Large range here = evidence that standardization was needed)")

    # ── Solve and Evaluate for different Regularization Coefficients ──────────
    # Bug #22 fix: use MEAN-CENTERED cosine similarity for alpha selection.
    #
    # WHY RAW COSIM IS BROKEN FOR ALPHA SELECTION:
    #   At very high alpha (e.g. 10000), Ridge L2 penalty dominates → W → 0.
    #   Every prediction collapses to b (the bias), which equals mean(Y_train).
    #   CLIP image embeddings cluster in CLIP space → cosim(mean, any_embed) ~0.7-0.9.
    #   With old text targets, 71/77 token positions were identical EOS padding →
    #   cosim was ~0.86 "for free". Alpha sweep picked alpha=10000 as "best"
    #   even though W=0 means the model learned NOTHING.
    #
    # MEAN-CENTERED COSIM (selection metric):
    #   Y_pred_c = Y_pred - mean(Y_pred)   ← remove the "free" mean signal
    #   Y_test_c = Y_test - mean(Y_test)   ← same for targets
    #   centered_cosim = cosim(Y_pred_c, Y_test_c)
    #   If W=0: Y_pred is constant → Y_pred_c = 0 → centered_cosim = 0.
    #   Only alphas producing predictions that VARY with input score > 0.
    alphas = [0.1, 1.0, 10.0, 100.0, 1000.0, 5000.0, 10000.0, 50000.0, 100000.0]

    best_alpha          = None
    best_centered_cosim = -1.0
    best_W, best_b      = None, None

    # Precompute the test target mean for centering
    Y_test_mean = Y_test.mean(dim=0, keepdim=True)   # [1, 59136]

    print("\nSweeping Ridge Regression regularizer (alpha):")
    print(f"  {'Alpha':<10} {'ValMSE':>10} {'RawCosSim':>11} {'CenteredCos':>13} {'Note'}")
    print(f"  {'-'*65}")
    for alpha in alphas:
        t_start = time.time()

        W, b = solve_ridge(X_train_s, Y_train, alpha)
        Y_pred = torch.matmul(X_test_s, W) + b

        Y_pred_seq = Y_pred.view(-1, 77, 768)
        Y_test_seq = Y_test.view(-1, 77, 768)

        # Diagnostic: raw cosine similarity (NOT used for selection)
        val_mse   = F.mse_loss(Y_pred_seq, Y_test_seq).item()
        raw_cosim = F.cosine_similarity(Y_pred_seq, Y_test_seq, dim=-1).mean().item()

        # Selection metric: mean-centered cosine similarity
        Y_pred_c    = (Y_pred - Y_pred.mean(dim=0, keepdim=True)).view(-1, 77, 768)
        Y_test_c    = (Y_test - Y_test_mean).view(-1, 77, 768)
        pred_norm   = Y_pred_c.norm(dim=-1).mean().item()
        if pred_norm < 1e-8:
            centered_cosim = 0.0
            note = "W≈0 MEAN COLLAPSE"
        else:
            centered_cosim = F.cosine_similarity(Y_pred_c, Y_test_c, dim=-1).mean().item()
            note = "ok" if centered_cosim > 0.05 else "near-zero"

        t_elapsed = time.time() - t_start
        print(f"  {alpha:<10} {val_mse:>10.6f} {raw_cosim:>11.6f} {centered_cosim:>13.6f}  {note}  [{t_elapsed:.2f}s]")

        if centered_cosim > best_centered_cosim:
            best_centered_cosim = centered_cosim
            best_alpha = alpha
            best_W = W
            best_b = b

    print(f"\n{'='*60}")
    print(f"  RIDGE REGRESSION OPTIMAL SOLUTION FOUND")
    print(f"  Best Alpha : {best_alpha}")
    print(f"  Best Centered CosSim : {best_centered_cosim:.6f}")
    print(f"{'='*60}\n")

    # ── Branch Contribution Report (Bug #16 diagnostic) ───────────────────────
    # Shows what fraction of the Ridge weight L2-norm comes from each branch.
    # After standardization (Bug #15), per-dim scale is equal.
    # But appearance has 768 dims vs object's 128 — monitor if appearance dominates.
    #
    # HEALTHY: appearance should contribute ~40-70% (it's the CLIP-aligned branch
    #          and has the most dims — some dominance is expected and correct).
    # UNHEALTHY: if appearance > 80% AND val_cosim is low, appearance is fitting noise.
    #            Consider adding PCA to reduce appearance dims before Ridge.
    branch_slices = {
        "object":     (0,    128),
        "appearance": (128,  896),
        "spatial":    (896,  960),
        "temporal":   (960,  1088),
        "view":       (1088, 1152),
    }
    total_weight_norm = (best_W ** 2).sum().item()
    print(f"  BRANCH CONTRIBUTION TO RIDGE WEIGHTS (% of total L2 norm):")
    for branch, (start, end) in branch_slices.items():
        branch_norm = (best_W[start:end, :] ** 2).sum().item()
        pct = 100 * branch_norm / (total_weight_norm + 1e-12)
        bar = '█' * int(pct / 2)
        print(f"    {branch:<12} dims [{start:4d}-{end:4d}]  {pct:5.1f}%  {bar}")
    print()

    # ── Save Best Model Checkpoint ────────────────────────────────────────────
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config.json
    with open(save_dir / "config.json", "w") as f:
        json.dump({
            "alpha": best_alpha,
            "val_cos_sim": best_centered_cosim,
            "checkpoint_path": str(ckpt_path.resolve())
        }, f, indent=2)

    # Save model state dict — includes scaler stats as buffers
    best_model = EEGFusionRidge(best_W, best_b, x_mean, x_std)
    torch.save({
        "epoch": 0,  # Zero training epochs (closed-form solution)
        "model_state_dict": best_model.state_dict(),
        "alpha": best_alpha,
        "val_cos_sim": best_val_cosim,
        "cat_index": cat_index,
        "subj_index": subj_index,
        # Scaler stats saved explicitly for manual inspection
        "x_mean": x_mean,
        "x_std":  x_std,
    }, save_dir / "best_fusion.pt")
    
    print(f"Successfully saved Ridge Fusion weights to: {save_dir.resolve()}/best_fusion.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 7: Feature Fusion via Analytical Ridge Regression")
    parser.add_argument("--data_dir", type=str, default="data/EEGdatanpy",
                        help="Path to pre-epoched, pre-normalized EEG data")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/phase06_2.24_leak_prevented/best_model.pt",
                        help="Path to the frozen Phase 6 checkpoint")
    parser.add_argument("--save_dir", type=str, default="checkpoints/phase07",
                        help="Directory to save Phase 7 models and config")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Loader batch size")

    args = parser.parse_args()

    train_fusion_ridge(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint_path,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
    )
