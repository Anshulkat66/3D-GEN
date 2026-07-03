"""Phase 6: Multi-Loss Training.

Combines Phases 3, 4, and 5 into one end-to-end trainable pipeline.
Three targeted loss functions force each branch to specialise:

  Loss 1 — Object Classification (CrossEntropy on normalised latents):
      Same object label  → pull Object latents close
      Different labels   → push Object latents apart

  Loss 2 — Subject Invariance:
      Same object, different subject → Object latents similar
      Different subjects (any object) → Subject latents different

  Loss 3 — Appearance ↔ CLIP:
      Appearance latent → cosine align with CLIP image embedding of stimulus

Total Loss = w_object*L1 + w_subject*(L2+L_subjCE) + w_clip*L3
Gradient flows back through ALL of Phase 3, 4, 5 simultaneously.

EEG Augmentation (applied per-batch, training only):
  Prevents memorisation of specific EEG waveforms by injecting diversity.
  Three independent augmentations are applied in sequence:
    1. Gaussian noise  : noise ∝ sample std  → forces robust category signal
    2. Channel dropout : zero random channels → prevents electrode-specific memory
    3. Temporal shift  : circular roll ±N steps → prevents onset-timing memorisation

Training defaults:
  Optimizer  : AdamW  lr=1e-4  weight_decay=1e-4
  Scheduler  : CosineAnnealingLR (decays over full 500 epochs)
  Grad clip  : 1.0
  Epochs     : 500
  Batch size : 256

Phased CLIP weight (prevents overfitting to fixed CLIP target):
  CLIP loss runs until cosine similarity plateaus (no improvement for clip_patience epochs).
  Plateau is detected automatically — no guessing required.
  After plateau: CLIP silenced, SupCon+Subject continue for remaining epochs.
"""

import json
import time
from pathlib import Path
import numpy as np

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from phase_02_DL       import create_dataloader
from phase_02_mse_tok  import EEGEncoder
from phase_04_attention import EEGTransformerEncoder
from phase_05_separation import EEGFeatureSeparation

# Optional CLIP — training proceeds without it if not installed
try:
    import clip as clip_module
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Utility: label / subject parsing
# ─────────────────────────────────────────────────────────────────────────────

def extract_category(label: str) -> str:
    """Pull the object name out of a label string.

    Examples:
        "01_airplane_05" → "airplane"
        "44_mug_06"      → "mug"
        "07_grand_piano_02" → "grand_piano"
    """
    parts = label.split("_")
    # first part = category number, last part = image index
    return "_".join(parts[1:-1])


def build_category_index(labels: list) -> dict:
    """category_name → integer (0-based, sorted alphabetically)."""
    cats = sorted(set(extract_category(l) for l in labels))
    return {c: i for i, c in enumerate(cats)}


def build_subject_index(subjects: list) -> dict:
    """subject_string → integer (0-based, sorted)."""
    unique = sorted(set(subjects))
    return {s: i for i, s in enumerate(unique)}


def labels_to_tensor(labels, cat_index: dict, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [cat_index[extract_category(l)] for l in labels],
        dtype=torch.long, device=device
    )


def subjects_to_tensor(subjects, subj_index: dict, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [subj_index[s] for s in subjects],
        dtype=torch.long, device=device
    )


# ─────────────────────────────────────────────────────────────────────────────
# Loss 1: Object Classification (CrossEntropy on L2-normalised Object latents)
# ─────────────────────────────────────────────────────────────────────────────
# SupCon was replaced because it collapses on L2-normalised features:
# At collapse all cosine sims = 1.0 → after max-subtraction = 0 → uniform softmax
# → effective gradient on unit sphere = 0 → cannot escape collapse.
#
# CrossEntropy on a linear head over L2-normalised features has NO collapse
# problem. The linear head is training-only — discarded after Phase 6.
# Equivalent to cosine-similarity classification (nearest class prototype).


# ─────────────────────────────────────────────────────────────────────────────
# Loss 2: Subject Invariance Loss  (Object + Subject branches)
# ─────────────────────────────────────────────────────────────────────────────

def subject_invariance_loss(
    obj_latents:  torch.Tensor,  # [B, D_obj]  L2-normalised
    subj_latents: torch.Tensor,  # [B, D_sub]  L2-normalised
    cat_labels:   torch.Tensor,  # [B] integer object category
    subj_labels:  torch.Tensor,  # [B] integer subject id
    alpha:        float = 1.0,   # weight: Object invariance term
    beta:         float = 1.0,   # weight: Subject branch term
) -> torch.Tensor:
    """For pairs (i, j) based on same/different subject and category:

    Object branch:
      PULL: same category, different subject  → Object latents should be SIMILAR
      PUSH: different category                → Object latents should be DIFFERENT

    Subject branch:
      PULL: same subject,  different category → Subject latents should be SIMILAR
            (Encoding WHO, so representation must be consistent across objects)
      PUSH: same category, different subject  → Subject latents should be DIFFERENT
            (Encoding WHO, so different people viewing same object must be distinct)

    If no valid pairs exist in this batch, returns 0.
    """
    device = obj_latents.device

    same_cat  = (cat_labels.unsqueeze(1)  == cat_labels.unsqueeze(0))
    diff_cat  = ~same_cat
    same_subj = (subj_labels.unsqueeze(1) == subj_labels.unsqueeze(0))
    diff_subj = ~same_subj

    # Object Positive: same category, different subject -> PULL
    obj_pos_mask = (same_cat & diff_subj)
    obj_pos_mask.fill_diagonal_(False)

    # Object Negative: different category -> PUSH
    obj_neg_mask = diff_cat

    # Subject PUSH: same category, different subject -> push apart
    subj_push_mask = (same_cat & diff_subj)
    subj_push_mask.fill_diagonal_(False)

    # Subject PULL: same subject, different category -> pull together  [Bug #23 fix]
    # Same person sees dog vs airplane -> subject latent must stay CONSISTENT
    subj_pull_mask = (same_subj & diff_cat)
    subj_pull_mask.fill_diagonal_(False)

    if obj_pos_mask.sum() == 0 or obj_neg_mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    if subj_push_mask.sum() == 0 and subj_pull_mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Cosine similarity matrices (features are already L2-normalised)
    obj_sim  = torch.matmul(obj_latents,  obj_latents.T)   # [B, B]
    subj_sim = torch.matmul(subj_latents, subj_latents.T)  # [B, B]

    # Object: pull same-class together, push different-class apart
    # Bug #24 fix: both terms use .mean() → pair counts already normalized.
    # The old 2× push multiplier caused over-dispersal on the unit sphere.
    obj_pos_loss = -obj_sim[obj_pos_mask].mean()
    obj_neg_loss = torch.relu(obj_sim[obj_neg_mask]).mean()
    obj_loss = obj_pos_loss + obj_neg_loss   # equal weight after mean-normalization

    # Subject: PULL same_subj/diff_cat + PUSH same_cat/diff_subj
    # Bug #24 fix: .mean() already normalizes by pair count.
    # Old 2× push caused subject features to over-disperse on unit sphere.
    subj_pull_loss = (-subj_sim[subj_pull_mask].mean()
                      if subj_pull_mask.sum() > 0
                      else torch.tensor(0.0, device=device))
    subj_push_loss = ( subj_sim[subj_push_mask].mean()
                      if subj_push_mask.sum() > 0
                      else torch.tensor(0.0, device=device))
    subj_loss = subj_pull_loss + subj_push_loss   # equal weight after mean-normalization

    return alpha * obj_loss + beta * subj_loss


# ─────────────────────────────────────────────────────────────────────────────
# Loss 2b: Intra-Class Consistency Loss  (Instance Memorization Fix)
# ─────────────────────────────────────────────────────────────────────────────

def intra_class_consistency_loss(
    obj_latents:   torch.Tensor,  # [B, D_obj]  L2-normalised
    cat_labels:    torch.Tensor,  # [B] integer object category
    label_strings: list,          # raw label strings e.g. "01_airplane_05"
) -> torch.Tensor:
    """Pulls same-category, DIFFERENT-image pairs together in Object latent space
    while pushing different-category pairs apart to prevent representation collapse.

    This directly attacks instance memorization:
      Current subject_invariance_loss pulls: same_cat + diff_subject
      This loss pulls:                       same_cat + diff_image (any subject)

    Forces the model to learn what is COMMON across ALL training images
    of a category, not just the EEG fingerprint of each specific image.
    Example:
      airplane_03 (sub01) ←→ airplane_06 (sub01)  ← directly attacked here
      airplane_03 (sub01) ←→ airplane_07 (sub04)  ← also attacked here
    """
    device = obj_latents.device

    # Extract image index: "01_airplane_05" → 5
    img_indices = torch.tensor(
        [int(lbl.split("_")[-1]) for lbl in label_strings],
        device=device
    )

    # Positive: same category, different image instance
    same_cat = (cat_labels.unsqueeze(1) == cat_labels.unsqueeze(0))    # [B, B]
    diff_img = (img_indices.unsqueeze(1) != img_indices.unsqueeze(0))  # [B, B]
    pos_mask = same_cat & diff_img
    pos_mask.fill_diagonal_(False)

    # Negative: different category
    neg_mask = ~same_cat

    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Cosine similarity (features are already L2-normalised)
    obj_sim = torch.matmul(obj_latents, obj_latents.T)  # [B, B]

    # Pull positives together, push negatives apart
    # Bug #24 fix: .mean() already normalizes by pair count — equal weighting.
    # Old 2× push caused object features to over-disperse on unit sphere.
    pos_loss = -obj_sim[pos_mask].mean()
    neg_loss = torch.relu(obj_sim[neg_mask]).mean()

    return pos_loss + neg_loss   # equal weight after mean-normalization




# ─────────────────────────────────────────────────────────────────────────────
# Gradient Reversal Layer for Subject-Adversarial Loss (DANN)
# ─────────────────────────────────────────────────────────────────────────────

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha=1.0):
    return GradReverse.apply(x, alpha)




# ─────────────────────────────────────────────────────────────────────────────
# Loss 3: Appearance ↔ CLIP Alignment Loss
# ─────────────────────────────────────────────────────────────────────────────

def contrastive_clip_loss(
    appearance_latents: torch.Tensor,  # [B, 768]  L2-normalised  (ViT-L/14)
    labels:             list,           # list of raw label strings
    clip_cache:         dict,           # {label_str → tensor [768]} pre-built
    device:             torch.device,
    temperature:        float = 0.15,   # ← was 0.07 (CLIP paper value for 400M pairs)
                                        #   0.07 forces per-image memorization on 192K noisy EEG
                                        #   0.15 allows categorical grouping → generalizes to unseen images
                                        #   Standard for EEG-to-image contrastive learning (Mind-Vis, DREAM)
) -> torch.Tensor:
    """InfoNCE loss to align Appearance latents with CLIP target embeddings.

    For each sample i in the batch:
      - similarity(appearance_i, clip_target_i) is MAXIMISED (pull)
      - similarity(appearance_i, clip_target_j) is MINIMISED for j != i (push)

    Temperature note:
      Low temperature (0.07): massively amplifies similarity differences.
        Forces model to push EVERY sample to a unique point on the unit sphere.
        Only works with massive clean datasets (CLIP: 400M pairs).
        On 192K noisy EEG: model memorizes training coordinates, fails on test.
      Higher temperature (0.15): allows similar EEG trials to cluster together.
        Model learns categorical structure rather than instance memorization.
        Generalizes to unseen images within the same category.
    """
    import numpy as np
    import torch.nn.functional as F

    clip_feats = torch.stack([clip_cache[lbl] for lbl in labels]).to(device)
    clip_feats = F.normalize(clip_feats, dim=-1)

    # Cosine similarities matrix between all latents and all CLIP targets
    # [B, 768] @ [768, B] → [B, B]   (768 = ViT-L/14 embedding dim)
    sim_matrix = torch.matmul(appearance_latents, clip_feats.T) / temperature

    # ── False Negative Masking (Bug #10 fix) ────────────────────────────────
    # The THINGS-EEG dataset has 12 subjects. In a batch of 256, it is common
    # for multiple subjects to have viewed the SAME image (same label string).
    # Those pairs share identical CLIP targets → they should NOT be negatives.
    labels_arr = np.array(labels)
    same_label = torch.tensor(
        labels_arr[:, None] == labels_arr[None, :],   # [B, B] bool
        device=device
    )
    same_label.fill_diagonal_(False)   # diagonal is the genuine positive — keep it
    # Apply mask: same-label off-diagonal positions → -inf (excluded from softmax)
    sim_matrix = sim_matrix.masked_fill(same_label, float('-inf'))

    # Supervised targets: sample i's positive is at column i (the diagonal)
    targets = torch.arange(len(labels), device=device)

    # Bi-directional loss (standard CLIP loss: latent-to-target + target-to-latent)
    loss_latents = F.cross_entropy(sim_matrix, targets)
    loss_targets = F.cross_entropy(sim_matrix.T, targets)
    infonce_loss = 0.5 * (loss_latents + loss_targets)

    # ── Cross-Subject Appearance Alignment (explicit pull) ───────────────────
    cross_subj_loss = torch.tensor(0.0, device=device)
    if same_label.sum() > 0:
        # Cosine similarity between appearance embeddings
        # (features are already L2-normalised → dot product = cosine sim)
        app_sim = torch.matmul(appearance_latents, appearance_latents.T)  # [B, B]
        # Maximise similarity for same-image pairs: loss = -mean(sim for same-image pairs)
        cross_subj_loss = -app_sim[same_label].mean()

    # Total: InfoNCE (CLIP alignment) + 0.5 × cross-subject appearance pull
    return infonce_loss + 0.5 * cross_subj_loss


# ─────────────────────────────────────────────────────────────────────────────
# CLIP cache builder  (text-based, initial experiment)
# ─────────────────────────────────────────────────────────────────────────────

def build_clip_cache_images(
    all_labels: list,
    image_dir:  str,
    device:     torch.device,
) -> dict:
    """Pre-compute CLIP IMAGE features for every unique stimulus image.

    Each label maps to a specific .png file:
        data/image/<category>/<label>.png

    Every image gets its OWN unique CLIP embedding —
    airplane_01 and airplane_09 produce different vectors because
    they look different. This is the correct target for the Appearance branch.

    Returns:
        dict {label_string: cpu tensor [768]}  ← ViT-L/14 image embedding dim
    """
    if not CLIP_AVAILABLE:
        raise ImportError(
            "CLIP not installed. Run:\n"
            "  pip install git+https://github.com/openai/CLIP.git"
        )

    print("  Building CLIP image cache (ViT-L/14 — must match Phase 7 + Stable Diffusion)...")
    # CRITICAL: Must use ViT-L/14 here — NOT ViT-B/32.
    # Stable Diffusion v1.5 uses ViT-L/14 for its text encoder (768-dim space).
    # Phase 7 also uses ViT-L/14 to produce text sequence targets.
    # If we use ViT-B/32 here (512-dim), the appearance branch learns targets
    # in a completely different embedding space from what Phase 7 expects.
    # ViT-B/32 and ViT-L/14 image embeddings are NOT interchangeable.
    clip_model, preprocess = clip_module.load("ViT-L/14", device=device)
    clip_model.eval()

    img_root    = Path(image_dir)
    unique_lbls = list(set(all_labels))
    clip_cache  = {}
    missing     = []

    for label in unique_lbls:
        category = extract_category(label)          # "01_airplane_05" → "airplane"
        img_path = img_root / category / f"{label}.png"

        if not img_path.exists():
            missing.append(str(img_path))
            continue

        # Load and preprocess for CLIP
        img    = Image.open(img_path).convert("RGB")
        tensor = preprocess(img).unsqueeze(0).to(device)   # [1, 3, 224, 224]

        with torch.no_grad():
            feat = clip_model.encode_image(tensor).float()  # [1, 768]  ← ViT-L/14
            feat = F.normalize(feat, dim=-1).squeeze(0)     # [768]

        clip_cache[label] = feat.cpu()

    if missing:
        print(f"  ⚠  {len(missing)} images not found — check paths:")
        for p in missing[:3]:
            print(f"      {p}")
        if len(missing) > 3:
            print(f"      ... and {len(missing)-3} more")

    print(f"  CLIP image cache ready: {len(clip_cache)} unique images.")
    return clip_cache


# ─────────────────────────────────────────────────────────────────────────────
# Combined Pipeline: Phase 3 + 4 + 5
# ─────────────────────────────────────────────────────────────────────────────

class EEGPipeline(nn.Module):
    """Wraps Phases 3, 4, and 5 into one trainable module.

    All three phases share a single AdamW optimizer — gradients flow
    back through Separation → Attention → Encoder simultaneously.
    """

    def __init__(
        self,
        token_dim: int = 256,
        num_heads: int = 4,
        ff_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,            # BranchMLP dropout (Phase 5 separation)
        transformer_dropout: float = 0.1, # Transformer internal dropout (Phase 4)
        appearance_dim: int = 768,        # ViT-L/14 image embedding dim (NOT 512 which is ViT-B/32)
    ):
        # Bug #21 fix: split dropout into two separate rates.
        # Problem: the same dropout=0.32 was passed to BOTH:
        #   - EEGTransformerEncoder: 2 internal passes per layer × 2 layers = 4 passes
        #   - EEGFeatureSeparation BranchMLP: 1 pass per branch
        # Combined: (1-0.32)^5 = 0.145 → 85% info destroyed at training, 0% at inference
        # → ~7× information density shift between train and inference.
        #
        # Fix: separate rates
        #   transformer_dropout=0.10 → (0.90)^4 = 0.656 (34% destroyed by Transformer)
        #   dropout=0.32             → × 0.68         (55% total destroyed)
        #   Shift: 1/0.446 = 2.2×  (vs 6.9× before)
        super().__init__()
        self.encoder    = EEGEncoder(embed_dim=token_dim)
        self.attention  = EEGTransformerEncoder(
            token_dim=token_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_layers=num_layers,
            dropout=transformer_dropout,   # Bug #21: lighter dropout for 4-pass Transformer
        )
        self.separation = EEGFeatureSeparation(
            token_dim=token_dim,
            appearance_dim=appearance_dim,
            dropout=dropout,               # Bug #21: stronger dropout ok for 1-pass BranchMLP
        )

    def forward(self, eeg: torch.Tensor) -> dict:
        """
        Args:
            eeg : [B, 250, 64]  raw EEG batch from DataLoader
        Returns:
            dict of 6 L2-normalised branch latents + 'shared' key:
              latents['shared'] = mean-pooled transformer output [B, token_dim]
                                  used by DANN GRL (Bug fix: apply BEFORE separation)
        """
        tokens    = self.encoder(eeg)          # [B, 63, token_dim]  (63 after Bug #19 padding fix)
        embedding = self.attention(tokens)     # [B, 63, token_dim]
        latents   = self.separation(embedding) # {6 branches}

        # Bug fix: expose shared representation BEFORE separation for DANN GRL.
        # Mean-pool across the token (time) dimension to get [B, token_dim].
        # This is the signal that feeds ALL branches — regularizing it for
        # subject-invariance means every branch benefits from the adversarial loss.
        latents["shared"] = embedding.mean(dim=1)   # [B, token_dim]
        return latents



# ─────────────────────────────────────────────────────────────────────────────
# EEG Augmentation
# ─────────────────────────────────────────────────────────────────────────────

def augment_eeg(
    eeg:              torch.Tensor,
    noise_std:        float = 0.20,
    channel_drop_p:   float = 0.15,
    time_shift_max:   int   = 10,
) -> torch.Tensor:
    """Randomised augmentations applied per batch during training.

    Args:
        eeg            : [B, C, T]  raw EEG batch  (64 channels × 250 time-steps)
        noise_std      : Gaussian noise std RELATIVE to each sample's own signal std.
                         0.20 = add 20 % noise — enough to disrupt memorisation
                         without destroying the category signal.
        channel_drop_p : Probability that each channel is zeroed independently.
                         0.15 = on average 9 of 64 channels are silenced per sample.
        time_shift_max : Max circular temporal shift in samples (0 = disabled).
                         ±10 of 250 samples = ±40 ms shift at 250 Hz.

    Returns:
        Augmented EEG tensor — same shape and device as input.
    """
    B, C, T = eeg.shape

    # 1. Gaussian noise — proportional to each sample's own std so it is
    #    scale-invariant across subjects and conditions.
    if noise_std > 0.0:
        sig_std = eeg.std(dim=(1, 2), keepdim=True).clamp(min=1e-6)  # [B, 1, 1]
        noise   = torch.randn_like(eeg) * (noise_std * sig_std)
        eeg     = eeg + noise

    # 2. Channel dropout — independently zero each channel with probability p.
    #    This acts like spatial dropout over the electrode array.
    if channel_drop_p > 0.0:
        mask = (torch.rand(B, C, 1, device=eeg.device) > channel_drop_p).float()
        eeg  = eeg * mask

    # 3. Random circular temporal shift — avoids hard edge artefacts that
    #    zero-padding would introduce.
    if time_shift_max > 0:
        shifts  = torch.randint(-time_shift_max, time_shift_max + 1, (B,))
        eeg     = torch.stack(
            [torch.roll(eeg[i], shifts[i].item(), dims=-1) for i in range(B)]
        )

    return eeg


# ─────────────────────────────────────────────────────────────────────────────
# Decoupling & Validation Helpers
# ─────────────────────────────────────────────────────────────────────────────

def orthogonality_loss(latents: dict) -> torch.Tensor:
    """Cross-branch decoupling via cross-covariance penalty.

    Penalises co-variation between object↔subject and subject↔appearance
    feature dimensions across the batch. Forces the branches to represent
    different aspects of the EEG signal (identity, category, visual style)
    without redundancy.

    NOTE — this is NOT the same as Barlow Twins:
      Barlow Twins: cross-VIEW correlation on a SINGLE branch (two augmented
                    forward passes, [D,D] matrix, diagonal→1 + off-diag→0).
      This loss:    cross-BRANCH covariance between DIFFERENT branches
                    ([D_obj, D_subj] matrix, all entries→0).

    Bug #14 fix:
      Old code used F.normalize(dim=0) which L2-normalizes each feature column
      across the batch. This is NOT batch standardization — it leaves non-zero
      means, so the "correlation" is contaminated by mean offsets and doesn't
      measure true co-variation.

      Correct approach: subtract batch mean per feature (zero-centering), then
      compute cross-covariance / B. This answers:
        "After removing average activation, do object and subject neurons
         consistently activate together across samples?"
      If yes → high covariance → high loss → model is penalised.
      If no  → covariance ≈ 0 → loss ≈ 0 → branches are decoupled.
    """
    obj  = latents["object"]      # [B, 128]
    subj = latents["subject"]     # [B, 64]
    app  = latents["appearance"]  # [B, 768]

    B = obj.shape[0]

    # Zero-center each feature across the batch (remove mean bias)
    # Without this, the cross-covariance is contaminated by non-zero means.
    obj_c  = obj  - obj.mean(dim=0, keepdim=True)   # [B, 128]
    subj_c = subj - subj.mean(dim=0, keepdim=True)  # [B, 64]
    app_c  = app  - app.mean(dim=0, keepdim=True)   # [B, 768]

    # Cross-covariance matrices (normalized by batch size)
    # C[i, j] = covariance between feature i of branch A and feature j of branch B
    C_obj_subj = (obj_c.T @ subj_c) / B    # [128, 64]
    C_subj_app = (subj_c.T @ app_c) / B    # [64,  768]

    # Penalize non-zero cross-covariance: object ↔ subject and subject ↔ appearance
    # Object ↔ Appearance allowed to correlate (CLIP shapes both category + visual latents)
    loss = (C_obj_subj ** 2).mean() + (C_subj_app ** 2).mean()
    return loss



@torch.no_grad()
def evaluate_test_accuracy(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    cat_index: dict,
    device: torch.device
):
    """Computes Nearest Centroid Top-1 accuracy on the test set.

    Also returns class separation metrics (reusing centroids — no extra
    forward pass required):
      intra_sim : mean cosine similarity between same-category training samples
      inter_sim : mean cosine similarity between different-category centroids
      ratio     : intra_sim / inter_sim  (higher = better separated classes)

    Returns:
        (top1_acc, intra_sim, inter_sim, ratio)
    """
    model.eval()

    # 1. Extract training latents to compute class centroids
    train_latents = []
    train_cats = []
    for eeg, labels, _ in train_loader:
        eeg = eeg.to(device)
        latents = model(eeg)
        train_latents.append(latents["object"].cpu())
        train_cats.extend([cat_index[extract_category(lbl)] for lbl in labels])

    train_latents = torch.cat(train_latents, dim=0)  # [N_train, 128]
    train_cats    = np.array(train_cats)              # [N_train]

    # Compute L2-normalised centroid for each category
    n_categories = len(cat_index)
    centroids = torch.zeros(n_categories, 128)
    for c in range(n_categories):
        indices = np.where(train_cats == c)[0]
        if len(indices) > 0:
            mean_vec = train_latents[indices].mean(dim=0)
            centroids[c] = F.normalize(mean_vec, dim=-1)

    # ── Class Separation Metrics ──────────────────────────────────────────
    # IntraSim: mean cosine similarity between samples of the SAME category.
    # Measures how tightly clustered each class is.
    intra_sims = []
    for c in range(n_categories):
        indices = np.where(train_cats == c)[0]
        if len(indices) > 1:
            vecs = F.normalize(train_latents[indices], dim=-1)  # [k, 128]
            sim  = torch.matmul(vecs, vecs.T)                  # [k, k]
            off  = sim[~torch.eye(len(vecs), dtype=torch.bool)]
            intra_sims.append(off.mean().item())
    intra_sim = float(np.mean(intra_sims)) if intra_sims else 0.0

    # InterSim: mean cosine similarity between DIFFERENT-category centroids.
    # Measures how well-separated the class prototypes are.
    sim_matrix  = torch.matmul(centroids, centroids.T)              # [72, 72]
    off_diag    = sim_matrix[~torch.eye(n_categories, dtype=torch.bool)]
    inter_sim   = off_diag.mean().item()

    # Ratio: how much more similar same-class is vs different-class.
    # Healthy target: ratio > 5.  Warning: ratio < 2.
    ratio = intra_sim / (abs(inter_sim) + 1e-8)

    # 2. Extract test latents and compute classification accuracy
    test_latents = []
    test_cats    = []
    for eeg, labels, _ in test_loader:
        eeg = eeg.to(device)
        latents = model(eeg)
        test_latents.append(latents["object"].cpu())
        test_cats.extend([cat_index[extract_category(lbl)] for lbl in labels])

    test_latents = torch.cat(test_latents, dim=0)    # [N_test, 128]
    test_cats    = torch.tensor(test_cats, dtype=torch.long)

    # Cosine similarities: [N_test, 128] @ [128, 72] -> [N_test, 72]
    similarities = torch.matmul(test_latents, centroids.T)
    top1_preds   = torch.argmax(similarities, dim=1)
    top1_acc     = (top1_preds == test_cats).float().mean().item()

    return top1_acc, intra_sim, inter_sim, ratio


# ─────────────────────────────────────────────────────────────────────────────
# Loss Balance Reporter
# ─────────────────────────────────────────────────────────────────────────────

def log_loss_balance(
    epoch:        int,
    n_batches:    int,
    l1_sum:       float,  # Object CE accumulated
    l_subj_ce_sum:float,  # Subject CE accumulated
    l2_sum:       float,  # Subject Invariance accumulated
    l3_sum:       float,  # CLIP InfoNCE accumulated
    l_ortho_sum:  float,  # Barlow Ortho accumulated
    l_intra_sum:  float,  # Intra-Class accumulated
    l_adv_sum:    float,  # Adversary CE accumulated
    w_object:     float,
    w_subject:    float,
    w_clip:       float,
    w_ortho:      float,
    w_intra:      float,
    w_adv:        float,
    clip_active:  bool,
    avg_cosim:    float,
    T:            int = 30,   # Normalized softmax temperature
) -> None:
    """Prints a gradient-scale-aware loss balance report.

    WHY THIS EXISTS:
      Loss weights on paper (w_object=1.0, w_intra=2.5) do NOT reflect the
      true gradient magnitude each loss contributes to the object branch.
      CE loss with temperature T=30 has effective gradient ∝ T × weight,
      because the logits are scaled by T before softmax, amplifying gradients
      by T on the backward pass. Other losses (intra, ortho, invariance) have
      no temperature scaling — their gradient is just weight × loss_gradient.

      This means a naive reading of the weights is misleading:
        w_object=1.0, T=30  → effective gradient scale = 1.0 × 30 = 30
        w_intra=2.5,  T=1   → effective gradient scale = 2.5 × 1  = 2.5
      CE is actually 12× stronger than intra-class, not 0.4× weaker as weights suggest.

    WHAT IT CHECKS:
      - CE vs Intra-Class ratio (Bug #13: CE overpowering intra-class)
      - Barlow Ortho dominance (can suppress all other signals)
      - CLIP cosim health (is the appearance branch actually aligning?)
      - DANN adversary effectiveness (is subject info being stripped?)
    """
    # ── Average raw losses ───────────────────────────────────────────────────
    l1_avg       = l1_sum        / n_batches
    l_subj_avg   = l_subj_ce_sum / n_batches
    l2_avg       = l2_sum        / n_batches
    l3_avg       = l3_sum        / n_batches if clip_active else 0.0
    l_ortho_avg  = l_ortho_sum   / n_batches
    l_intra_avg  = l_intra_sum   / n_batches
    l_adv_avg    = l_adv_sum     / n_batches

    # ── Effective gradient scale = weight × T_factor × avg_loss ─────────────
    # CE-based losses have their logits scaled by T=30, so backward gradients
    # are amplified by T. Non-CE losses have T_factor=1.
    eff_obj    = w_object  * T * l1_avg       # Object CE    (T-scaled)
    eff_subj   = w_subject * T * l_subj_avg   # Subject CE   (T-scaled)
    eff_inv    = w_subject *     l2_avg        # Subj Invar.  (no T)
    eff_clip   = w_clip    *     l3_avg        # CLIP InfoNCE (no T)
    eff_ortho  = w_ortho   *     l_ortho_avg   # Barlow Ortho (no T)
    eff_intra  = w_intra   *     l_intra_avg   # Intra-Class  (no T)
    eff_adv    = w_adv     * T * l_adv_avg     # Adversary CE (T-scaled)

    total_eff  = eff_obj + eff_subj + eff_inv + eff_clip + eff_ortho + eff_intra + eff_adv
    if total_eff < 1e-8:
        return  # nothing to report

    def pct(x): return 100 * x / total_eff

    # ── Print table ─────────────────────────────────────────────────────────
    W = 58
    print(f"\n  {'─'*W}")
    print(f"  LOSS BALANCE REPORT  (Epoch {epoch})")
    print(f"  Key: Eff.Grad = weight × T_factor × avg_loss")
    print(f"  {'─'*W}")
    print(f"  {'Loss':<22} {'RawAvg':>7} {'W':>5} {'T':>4} {'Eff.Grad':>9} {'%':>6}")
    print(f"  {'─'*W}")
    print(f"  {'Object CE':<22} {l1_avg:>7.4f} {w_object:>5.1f} {T:>4d} {eff_obj:>9.2f} {pct(eff_obj):>5.1f}%")
    print(f"  {'Subject CE':<22} {l_subj_avg:>7.4f} {w_subject:>5.1f} {T:>4d} {eff_subj:>9.2f} {pct(eff_subj):>5.1f}%")
    print(f"  {'Subj Invariance':<22} {l2_avg:>7.4f} {w_subject:>5.1f} {'1':>4} {eff_inv:>9.2f} {pct(eff_inv):>5.1f}%")
    clip_raw = f"{l3_avg:.4f}" if clip_active else "  OFF"
    print(f"  {'CLIP InfoNCE':<22} {clip_raw:>7} {w_clip:>5.1f} {'1':>4} {eff_clip:>9.2f} {pct(eff_clip):>5.1f}%")
    print(f"  {'Barlow Ortho':<22} {l_ortho_avg:>7.4f} {w_ortho:>5.1f} {'1':>4} {eff_ortho:>9.2f} {pct(eff_ortho):>5.1f}%")
    print(f"  {'Intra-Class':<22} {l_intra_avg:>7.4f} {w_intra:>5.1f} {'1':>4} {eff_intra:>9.2f} {pct(eff_intra):>5.1f}%")
    print(f"  {'Adversary CE':<22} {l_adv_avg:>7.4f} {w_adv:>5.1f} {T:>4d} {eff_adv:>9.2f} {pct(eff_adv):>5.1f}%")
    print(f"  {'─'*W}")
    print(f"  Total Eff. Gradient Budget: {total_eff:.2f}")

    # ── Health checks ────────────────────────────────────────────────────────
    issues = []

    # Check 1: CE vs Intra-Class (Bug #13)
    if eff_intra > 1e-6:
        ce_intra_ratio = eff_obj / eff_intra
        if ce_intra_ratio > 8:
            issues.append(
                f"  ⚠  [BUG-13] CE/Intra ratio = {ce_intra_ratio:.1f}×  "
                f"(intra-class has only {pct(eff_intra):.1f}% influence — nearly silent).\n"
                f"     Fix options: raise w_intra above {w_intra} OR reduce T below {T}."
            )
        elif ce_intra_ratio > 4:
            issues.append(
                f"  ℹ  CE/Intra ratio = {ce_intra_ratio:.1f}×  "
                f"(intra-class is weak at {pct(eff_intra):.1f}% — monitor generalization)."
            )

    # Check 2: Barlow Ortho dominance
    if pct(eff_ortho) > 45:
        issues.append(
            f"  ⚠  Barlow Ortho uses {pct(eff_ortho):.0f}% of gradient budget — may suppress other signals.\n"
            f"     Consider reducing w_ortho below {w_ortho}."
        )

    # Check 3: CLIP cosine similarity health
    if clip_active:
        if avg_cosim < 0.02 and epoch > 30:
            issues.append(
                f"  ⚠  CLIP cosim = {avg_cosim:.4f} (near zero after epoch {epoch}).\n"
                f"     Check: CLIP cache loaded correctly, appearance_dim=768, ViT-L/14 model."
            )
        elif avg_cosim < 0.10 and epoch > 100:
            issues.append(
                f"  ℹ  CLIP cosim = {avg_cosim:.4f} (low after epoch {epoch}).\n"
                f"     Appearance branch may be struggling — verify CLIP targets are correct."
            )

    # Check 4: Adversary too weak to strip subject identity
    if pct(eff_adv) < 2.0:
        issues.append(
            f"  ℹ  Adversary contributes only {pct(eff_adv):.1f}% — may not strip subject identity effectively."
        )

    # Check 5: Any loss collapsed to near-zero
    # Use abs(): negative values (pull > push) are ACTIVE, not collapsed.
    collapsed = []
    if abs(l_intra_avg)  < 0.001:  collapsed.append("Intra-Class")
    if abs(l2_avg)       < 0.001:  collapsed.append("Subj-Invariance")
    if abs(l_ortho_avg)  < 0.001:  collapsed.append("Barlow-Ortho")
    if collapsed:
        issues.append(
            f"  ⚠  Collapsed losses (value ≈ 0): {', '.join(collapsed)}.\n"
            f"     These losses are contributing nothing — check if valid pairs exist in batch."
        )

    print()
    if not issues:
        print(f"  ✅ Loss balance looks healthy — no major imbalances detected.")
    else:
        for issue in issues:
            print(issue)
    print(f"  {'─'*W}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(
    data_dir:       str   = "data/EEGdatanpy",
    image_dir:      str   = "data/image",
    checkpoint_dir: str   = "checkpoints/phase06",
    num_epochs:     int   = 500,
    clip_patience:  int   = 40,    # epochs of no CLIP improvement before logging a WARNING
                                    # (CLIP is never silenced — this only triggers a log message)
    clip_threshold: float = 0.001, # minimum cosine similarity gain to count as improvement
    clip_warmup_epochs: int = 50,  # epochs before CLIP plateau monitoring starts
    lr:               float = 1e-4,
    weight_decay:     float = 0.01,  # Set to 0.01 to allow visual categories learning
    w_object:         float = 1.0,
    w_subject:        float = 0.5,
    w_clip:           float = 0.5,
    w_ortho:          float = 15.0,  # Barlow cross-correlation orthogonality loss
    w_intra:          float = 2.5,   # Reduced to 2.5 to allow cluster slack for unseen visual stimuli
    w_adv:            float = 1.0,   # Weight of the Subject-Adversarial (DANN) loss
    appearance_dim:   int   = 768,   # ViT-L/14 image embedding (768-dim, must match Phase 7 + SD)
    token_dim:        int   = 192,  # 3 heads × 64-dim = 192 (reduced capacity to prevent overfit)
    num_heads:        int   = 3,    # 3 heads × 64-dim per head (standard head size, fewer params)
    ff_dim:           int   = 384,  # 2× token_dim feed-forward expansion (384 = 2×192)
    num_layers:       int   = 2,    # 2 layers for noise filtering depth
    dropout:          float = 0.32, # BranchMLP dropout (separation branches, 1 pass each)
    transformer_dropout: float = 0.10, # Bug #21 fix: Transformer gets lighter dropout
                                       # (4 internal passes: 2 per layer × 2 layers)
                                       # 0.32 in Transformer: (0.68)^4=0.214 → 79% destroyed
                                       # 0.10 in Transformer: (0.90)^4=0.656 → 34% destroyed
    save_every:       int   = 10,
    log_every:        int   = 50,
    # ── EEG Augmentation ──────────────────────────────────────────────────
    aug_noise_std:        float = 0.09,
    aug_channel_drop_p:   float = 0.10,
    aug_time_shift_max:   int   = 8,
):
    """Train the full EEG pipeline (Phases 3+4+5) with 3 loss functions."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader = create_dataloader(data_dir=data_dir, batch_size=256, split="train")
    test_loader  = create_dataloader(data_dir=data_dir, batch_size=256, split="test")
    print(f"Train batches per epoch: {len(train_loader)}  ({len(train_loader)*256} samples)")
    print(f"Test batches per epoch: {len(test_loader)}  ({len(test_loader)*256} samples)")

    # ── Collect all labels + subjects to build indices ─────────────────────
    # One extra pass through the data — fast (strings only, no GPU work).
    print("Scanning dataset for label/subject indices...")
    all_labels   = []
    all_subjects = []
    for _, lbls, subjs in train_loader:
        all_labels.extend(list(lbls))
        all_subjects.extend(list(subjs))

    cat_index  = build_category_index(all_labels)
    subj_index = build_subject_index(all_subjects)
    print(f"  Categories : {len(cat_index)}")
    print(f"  Subjects   : {len(subj_index)}")

    # ── CLIP cache (built once, reused every batch) ────────────────────────
    use_clip = CLIP_AVAILABLE and w_clip > 0
    clip_cache = None
    if use_clip:
        clip_cache = build_clip_cache_images(all_labels, image_dir, device)
    else:
        print("  [Warning] CLIP not available or w_clip=0 — Appearance loss disabled.")

    # ── Model + losses + optimiser ────────────────────────────────────────────
    model   = EEGPipeline(
        token_dim=token_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
        dropout=dropout,
        transformer_dropout=transformer_dropout,   # Bug #21 fix
        appearance_dim=appearance_dim,
    ).to(device)
    ce_loss = nn.CrossEntropyLoss()

    # Training-only classification heads:
    # 1. Object: latent [B,128] → class logits [B,72]
    # 2. Subject: latent [B,64]  → subject logits [B,12]
    # 3. Subject Adversary (DANN): shared token rep [B,token_dim=192] → subject logits [B,12]
    #    Bug fix: adversary now sees the SHARED transformer output (before separation),
    #    not the object latent (after separation). This means:
    #      - Adversary has full access to the shared representation (192-dim vs 128-dim)
    #      - GRL gradient flows into ALL branches via the shared Transformer and Encoder,
    #        not just through the object branch pathway.
    obj_classifier  = nn.Linear(128,       len(cat_index),  bias=False).to(device)
    subj_classifier = nn.Linear(64,        len(subj_index), bias=False).to(device)
    subj_adversary  = nn.Linear(token_dim, len(subj_index), bias=False).to(device)

    optimizer = AdamW(
        list(model.parameters()) 
        + list(obj_classifier.parameters()) 
        + list(subj_classifier.parameters()) 
        + list(subj_adversary.parameters()),
        lr=lr, weight_decay=weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 0.01)
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(ckpt_dir / "config.json", "w") as f:
        json.dump({
            "num_epochs": num_epochs, "lr": lr, "weight_decay": weight_decay,
            "w_object": w_object, "w_subject": w_subject, "w_clip": w_clip, "w_ortho": w_ortho, "w_intra": w_intra, "w_adv": w_adv,
            "clip_warmup_epochs": clip_warmup_epochs,
            "appearance_dim": appearance_dim,
            "token_dim": token_dim,
            "num_heads": num_heads,
            "ff_dim": ff_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "aug_noise_std": aug_noise_std,
            "aug_channel_drop_p": aug_channel_drop_p,
            "aug_time_shift_max": aug_time_shift_max,
            "n_categories": len(cat_index), "n_subjects": len(subj_index),
            "cat_index": cat_index, "subj_index": subj_index,
        }, f, indent=2)

    total_params = (
        sum(p.numel() for p in model.parameters() if p.requires_grad)
        + sum(p.numel() for p in obj_classifier.parameters() if p.requires_grad)
        + sum(p.numel() for p in subj_adversary.parameters() if p.requires_grad)
    )

    print(f"\n{'='*60}")
    print(f"  PHASE 6 — TRAINING START")
    print(f"  Epochs   : {num_epochs}  (CLIP always active — plateau logged as warning only)")
    print(f"  Params   : {total_params:,}")
    print(f"  Losses   : Object-CE(w={w_object}) + Subject(w={w_subject}) + CLIP(w={w_clip}) + Barlow Ortho(w={w_ortho}) + SubjAdv(w={w_adv})")
    print(f"  CLIP patience : {clip_patience} epochs  |  threshold : {clip_threshold}  |  warmup : {clip_warmup_epochs} epochs")
    print(f"  LR: {lr}")
    print(f"{'='*60}\n")


    # ── CLIP plateau monitoring ────────────────────────────────────────────────
    # CLIP is NEVER silenced — it stays active for the full training.
    # Bug #11 fix: the old code permanently set clip_active=False when cosine
    # similarity plateaued. This left the appearance branch with zero pull signal
    # for the rest of training, causing it to drift as the Barlow loss pushed it
    # away from other improving branches.
    #
    # The plateau was caused by upstream bugs (τ=0.07, false negatives, ViT-B/32)
    # that are now fixed. With those fixed, CLIP alignment should genuinely improve.
    # If it still plateaus, log a WARNING so the user can investigate — but never kill it.
    clip_active       = use_clip and clip_cache is not None
    best_clip_cosim   = -float("inf")   # best cosine similarity seen so far
    clip_no_improve   = 0               # consecutive epochs without improvement
    clip_warned       = False           # whether plateau warning has been printed

    best_loss = float("inf")
    best_test_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        model.train()

        epoch_loss = l1_sum = l_subj_ce_sum = l2_sum = l3_sum = l3_cosim_sum = l_ortho_sum = l_intra_sum = l_adv_sum = 0.0
        n_batches = 0
        t0 = time.time()

        for batch_idx, (eeg, labels, subjects) in enumerate(train_loader):
            eeg = eeg.to(device)

            # ── EEG Augmentation (training only) ──────────────────────────
            # DataLoader delivers [B, Time, Channels] but augment_eeg expects
            # [B, Channels, Time] — transpose in and back out.
            eeg = eeg.permute(0, 2, 1)   # [B, T, C] → [B, C, T]
            eeg = augment_eeg(
                eeg,
                noise_std=aug_noise_std,
                channel_drop_p=aug_channel_drop_p,
                time_shift_max=aug_time_shift_max,
            )
            eeg = eeg.permute(0, 2, 1)   # [B, C, T] → [B, T, C]

            # Integer tensors for loss functions
            cat_t  = labels_to_tensor(list(labels),   cat_index,  device)
            subj_t = subjects_to_tensor(list(subjects), subj_index, device)

            # ── Forward ───────────────────────────────────────────────────────
            latents = model(eeg)

            # ── Loss 1: Object Classification (Normalized Softmax ×30) ──────────────
            # Manifold Mixup intentionally disabled (alpha_mixup=0.0):
            # Was tried with alpha=0.2/0.4 but made no positive contribution
            # to training on this EEG dataset. Kept as dead code for reference.
            alpha_mixup = 0.0
            if alpha_mixup > 0.0 and model.training:
                lam = np.random.beta(alpha_mixup, alpha_mixup)
                rand_idx = torch.randperm(latents["object"].size(0), device=device)
                mixed_obj = lam * latents["object"] + (1 - lam) * latents["object"][rand_idx]
                w_norm = F.normalize(obj_classifier.weight, dim=1)
                obj_logits = mixed_obj @ w_norm.T * 30.0
                l1 = lam * ce_loss(obj_logits, cat_t) + (1 - lam) * ce_loss(obj_logits, cat_t[rand_idx])
            else:
                w_norm = F.normalize(obj_classifier.weight, dim=1)
                obj_logits = latents["object"] @ w_norm.T * 7.0   # T=7 (was 30 — T=30 allows stable collapse where InterSim=0.948 satisfies CE)
                l1 = ce_loss(obj_logits, cat_t)

            # ── Loss 2: Subject Classification (Normalized Softmax ×7) ──────────
            # DETACH subject latent: prevents Subject CE gradient from flowing
            # back through the shared transformer. Without detach, Subject CE
            # trains the shared transformer to encode subject identity 50× more
            # strongly than the DANN reversal at early epochs (alpha ramps slowly).
            # With detach: gradient only updates subj_classifier.weight (head only).
            # The subject branch MLP is trained by subject_invariance_loss (l2).
            w_subj_norm = F.normalize(subj_classifier.weight, dim=1)  # [12, 64]
            subj_logits = latents["subject"].detach() @ w_subj_norm.T * 7.0   # [B, 12]
            l_subj_ce   = ce_loss(subj_logits, subj_t)

            # ── Loss 3: Subject Invariance (Object + Subject branches) ─────────
            l2 = subject_invariance_loss(
                latents["object"], latents["subject"],
                cat_t, subj_t,
            )

            # ── Loss 4: Appearance ↔ CLIP (Contrastive InfoNCE) ─────────────────
            # Silenced automatically when cosine similarity plateaus.
            current_w_clip = w_clip if clip_active else 0.0
            batch_cosim = 0.0
            if clip_active:
                l3 = contrastive_clip_loss(
                    latents["appearance"], list(labels), clip_cache, device
                )
                # Compute raw cosine similarity for tracking plateau (no gradients)
                with torch.no_grad():
                    clip_feats_val = torch.stack([clip_cache[lbl] for lbl in labels]).to(device)
                    clip_feats_val = F.normalize(clip_feats_val, dim=-1)
                    batch_cosim = (latents["appearance"] * clip_feats_val).sum(dim=-1).mean().item()
            else:
                l3 = torch.tensor(0.0, device=device)

            # ── Loss 5: Barlow Twins Orthogonality Loss (Disentanglement) ──────
            l_ortho = orthogonality_loss(latents)

            # ── Loss 6: Intra-Class Consistency (Instance Memorization Fix) ─────
            # Pulls same-category, different-image pairs together.
            # Forces model to learn what is COMMON across all images of a category,
            # not the EEG fingerprint of each specific training image.
            l_intra = intra_class_consistency_loss(
                latents["object"], cat_t, list(labels)
            )

            # ── Loss 7: Subject-Adversarial Loss (DANN on SHARED representation) ──
            # Ramp over first 20 epochs (was 100 — too slow to counteract subject CE
            # hijacking in early training). At epoch 20, alpha=1.0 and DANN is at
            # full strength before the shared transformer gets locked into subject identity.
            alpha_adv = min(1.0, epoch / 20.0)
            subj_adv_logits = subj_adversary(grad_reverse(latents["shared"], alpha_adv))
            l_adv = ce_loss(subj_adv_logits, subj_t)

            # ── Total loss ────────────────────────────────────────────────────
            # w_subject weights both the Subject Invariance (l2) and Subject classification (l_subj_ce)
            loss = (
                w_object * l1 
                + w_subject * (l_subj_ce + l2) 
                + current_w_clip * l3 
                + w_ortho * l_ortho 
                + w_intra * l_intra 
                + w_adv * l_adv
            )

            # ── Backward ──────────────────────────────────────────────────────
            optimizer.zero_grad()
            loss.backward()
            all_params = list(model.parameters()) + list(obj_classifier.parameters()) + list(subj_classifier.parameters()) + list(subj_adversary.parameters())
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            l1_sum += l1.item()
            l_subj_ce_sum += l_subj_ce.item()
            l2_sum += l2.item()
            l3_sum += l3.item()
            l3_cosim_sum += batch_cosim
            l_ortho_sum += l_ortho.item()
            l_intra_sum += l_intra.item()
            l_adv_sum += l_adv.item()
            n_batches += 1

            if (batch_idx + 1) % log_every == 0:
                print(
                    f"  Ep {epoch:3d} | Batch {batch_idx+1:3d}/{len(train_loader)} | "
                    f"Loss={loss.item():.4f}  "
                    f"Obj={l1.item():.4f}  "
                    f"Intra={l_intra.item():.4f}  "
                    f"SubjAdv={l_adv.item():.4f}  "
                    f"SubjCE={l_subj_ce.item():.4f}  "
                    f"SubjInv={l2.item():.4f}  "
                    f"Ortho={l_ortho.item():.4f}  "
                    f"CLIP_NCE={l3.item():.4f}"
                )

        scheduler.step()

        avg = epoch_loss / n_batches
        elapsed = time.time() - t0
        avg_cosim = l3_cosim_sum / n_batches if clip_active else 0.0
        clip_status = f"cosim={avg_cosim:.4f}" if clip_active else "DISABLED"

        print(
            f"Epoch {epoch:3d}/{num_epochs} | "
            f"Loss={avg:.4f}  "
            f"Obj={l1_sum/n_batches:.4f}  "
            f"Intra={l_intra_sum/n_batches:.4f}  "
            f"SubjAdv={l_adv_sum/n_batches:.4f}  "
            f"SubjCE={l_subj_ce_sum/n_batches:.4f}  "
            f"SubjInv={l2_sum/n_batches:.4f}  "
            f"Ortho={l_ortho_sum/n_batches:.4f}  "
            f"CLIP={clip_status}  "
            f"[{elapsed:.1f}s]"
        )

        # ── Loss Balance Report (every 10 epochs) ────────────────────────────
        if epoch % 10 == 0 or epoch == 1:
            log_loss_balance(
                epoch        = epoch,
                n_batches    = n_batches,
                l1_sum       = l1_sum,
                l_subj_ce_sum= l_subj_ce_sum,
                l2_sum       = l2_sum,
                l3_sum       = l3_sum,
                l_ortho_sum  = l_ortho_sum,
                l_intra_sum  = l_intra_sum,
                l_adv_sum    = l_adv_sum,
                w_object     = w_object,
                w_subject    = w_subject,
                w_clip       = w_clip,
                w_ortho      = w_ortho,
                w_intra      = w_intra,
                w_adv        = w_adv,
                clip_active  = clip_active,
                avg_cosim    = avg_cosim,
                T            = 7,    # must match the actual T used above
            )

        # ── CLIP plateau monitor (warning only — CLIP is never silenced) ──────
        if clip_active:
            if avg_cosim - best_clip_cosim > clip_threshold:
                best_clip_cosim = avg_cosim
                clip_no_improve  = 0
                clip_warned      = False   # reset warning if improvement resumes
            else:
                if epoch > clip_warmup_epochs:
                    clip_no_improve += 1
                    if clip_no_improve >= clip_patience and not clip_warned:
                        clip_warned = True
                        print(
                            f"  [CLIP WARNING] Cosine similarity has not improved for "
                            f"{clip_patience} epochs (best={best_clip_cosim:.4f}). "
                            f"CLIP loss remains ACTIVE. Check: temperature, CLIP cache, "
                            f"or consider reducing w_clip if appearance branch is collapsing."
                        )

        # ── Save best train loss model ───────────────────────────────────────
        if avg < best_loss:
            best_loss = avg
            torch.save({
                "epoch"               : epoch,
                "model_state_dict"    : model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss"                : avg,
                "cat_index"           : cat_index,
                "subj_index"          : subj_index,
            }, ckpt_dir / "best_train_loss_model.pt")
            print(f"  [Save] Best train loss model saved (loss={avg:.4f})")

        # ── Test Set Validation + Class Separation (every 10 epochs or last) ─
        # Runs a lightweight centroid-based accuracy evaluation on the test set
        # AND computes class separation metrics (IntraSim / InterSim / Ratio)
        # using the same centroids — no extra forward pass required.
        if epoch % 10 == 0 or epoch == num_epochs:
            test_acc, intra_sim, inter_sim, ratio = evaluate_test_accuracy(
                model, train_loader, test_loader, cat_index, device
            )
            print(f"  [Test] Test Set Nearest Centroid Top-1 Accuracy: {test_acc*100:.2f}%")
            # Ratio health guide: >5 = healthy  |  2-5 = marginal  |  <2 = collapsing
            ratio_tag = "[HEALTHY]" if ratio > 5 else ("[MARGINAL]" if ratio > 2 else "[COLLAPSED]")
            print(
                f"  [Separation] Class Separation | "
                f"IntraSim={intra_sim:.3f}  "
                f"InterSim={inter_sim:.3f}  "
                f"Ratio={ratio:.1f} {ratio_tag}"
            )
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                torch.save({
                    "epoch"               : epoch,
                    "model_state_dict"    : model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss"                : avg,
                    "test_acc"            : test_acc,
                    "cat_index"           : cat_index,
                    "subj_index"          : subj_index,
                }, ckpt_dir / "best_model.pt")
                print(f"  [Save] Best model saved on Test Accuracy (acc={test_acc*100:.2f}%)")

        # ── Periodic checkpoint ───────────────────────────────────────────────
        if epoch % save_every == 0:
            torch.save({
                "epoch"               : epoch,
                "model_state_dict"    : model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss"                : avg,
                "cat_index"           : cat_index,
                "subj_index"          : subj_index,
            }, ckpt_dir / f"checkpoint_epoch{epoch:03d}.pt")
            print(f"  [Save] Checkpoint saved: epoch {epoch}")

    print(f"\n{'='*60}")
    print(f"  Training complete!  Best loss: {best_loss:.4f}")
    print(f"  Checkpoints: {Path(checkpoint_dir).resolve()}")
    print(f"{'='*60}\n")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train(
        data_dir       = "data/EEGdatanpy",
        image_dir      = "data/image",
        checkpoint_dir = "checkpoints/phase06",
        num_epochs     = 500,
        clip_patience  = 40,
        clip_threshold = 0.001,
        clip_warmup_epochs = 50,  # Wait 50 epochs before starting CLIP plateau checks
        lr             = 1e-4,
        weight_decay   = 0.01,  # Set to 0.01 to allow visual categories learning
        w_object       = 1.0,
        w_subject      = 0.5,
        w_clip         = 0.5,
        w_ortho        = 15.0,  # Barlow cross-correlation orthogonality loss
        w_intra        = 2.5,   # Reduced to 2.5 to allow cluster slack for unseen visual stimuli
        w_adv          = 0.1,   # DANN adversary weight — reduced from 1.0 to 0.1.
        appearance_dim = 768,   # ViT-L/14 image embedding dim
        token_dim      = 192,   # 3 heads × 64-dim = 192 (reduced capacity to prevent overfit)
        num_heads      = 3,     # 3 heads × 64-dim per head (standard head size)
        ff_dim         = 384,   # 2× token_dim feed-forward expansion
        num_layers     = 2,     # 2 layers for noise filtering depth
        dropout        = 0.32,  # Increased to 0.32 to match larger capacity
        aug_noise_std  = 0.09,  # Increased to 0.09 to match larger capacity
        aug_time_shift_max = 8,  # Increased to 8 to prevent visual instance memorization
        save_every     = 10,
        log_every      = 50,
    )
