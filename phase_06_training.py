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
    beta:         float = 1.0,   # weight: Subject discrimination term
) -> torch.Tensor:
    """For pairs (i, j) where same object but different subject:

      Object branch  → MAXIMISE cosine similarity  (ignore who looked)
                       AND MINIMISE different-category similarity (prevents collapse)
      Subject branch → MINIMISE cosine similarity  (capture who looked)

    If no valid pairs exist in this batch, returns 0.
    """
    device = obj_latents.device

    same_cat  = (cat_labels.unsqueeze(1) == cat_labels.unsqueeze(0))   # [B, B]
    diff_subj = (subj_labels.unsqueeze(1) != subj_labels.unsqueeze(0)) # [B, B]
    
    # Object Positive: same category, different subject
    obj_pos_mask = (same_cat & diff_subj)
    obj_pos_mask.fill_diagonal_(False)

    # Object Negative: different category
    obj_neg_mask = ~same_cat

    # Subject: same category, different subject (should have different subject latents)
    subj_mask = (same_cat & diff_subj)
    subj_mask.fill_diagonal_(False)

    if obj_pos_mask.sum() == 0 or obj_neg_mask.sum() == 0 or subj_mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Cosine similarity matrices (features are already L2-normalised)
    obj_sim  = torch.matmul(obj_latents,  obj_latents.T)   # [B, B]
    subj_sim = torch.matmul(subj_latents, subj_latents.T)  # [B, B]

    # Object: pull same-class together (maximise similarity)
    # and push different-class apart (minimise similarity above 0.0)
    obj_pos_loss = -obj_sim[obj_pos_mask].mean()
    obj_neg_loss = torch.relu(obj_sim[obj_neg_mask]).mean()
    obj_loss = obj_pos_loss + 2.0 * obj_neg_loss

    # Subject: minimise similarity → maximise negative similarity
    subj_loss = subj_sim[subj_mask].mean()

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

    # Pull positives together, push negatives apart (margin 0.0)
    pos_loss = -obj_sim[pos_mask].mean()
    neg_loss = torch.relu(obj_sim[neg_mask]).mean()

    return pos_loss + 2.0 * neg_loss




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
    appearance_latents: torch.Tensor,  # [B, 512]  L2-normalised
    labels:             list,           # list of raw label strings
    clip_cache:         dict,           # {label_str → tensor [512]} pre-built
    device:             torch.device,
    temperature:        float = 0.07,   # Standard InfoNCE temperature parameter
) -> torch.Tensor:
    """InfoNCE loss to align Appearance latents with CLIP target embeddings.

    For each sample i in the batch:
      - similarity(appearance_i, clip_target_i) is MAXIMISED (pull)
      - similarity(appearance_i, clip_target_j) is MINIMISED for j != i (push)

    This contrastive force prevents collapse to a constant average vector.
    """
    clip_feats = torch.stack([clip_cache[lbl] for lbl in labels]).to(device)
    clip_feats = F.normalize(clip_feats, dim=-1)

    # Cosine similarities matrix between all latents and all CLIP targets
    # [B, 512] @ [512, B] → [B, B]
    sim_matrix = torch.matmul(appearance_latents, clip_feats.T) / temperature

    # Ground truth: sample i matches target i
    targets = torch.arange(len(labels), device=device)

    # Bi-directional loss (standard CLIP loss: image-to-text + text-to-image)
    loss_latents = F.cross_entropy(sim_matrix, targets)
    loss_targets = F.cross_entropy(sim_matrix.T, targets)

    return 0.5 * (loss_latents + loss_targets)


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
        dict {label_string: cpu tensor [512]}
    """
    if not CLIP_AVAILABLE:
        raise ImportError(
            "CLIP not installed. Run:\n"
            "  pip install git+https://github.com/openai/CLIP.git"
        )

    print("  Building CLIP image cache...")
    clip_model, preprocess = clip_module.load("ViT-B/32", device=device)
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
            feat = clip_model.encode_image(tensor).float()  # [1, 512]
            feat = F.normalize(feat, dim=-1).squeeze(0)     # [512]

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
        dropout: float = 0.1,
        appearance_dim: int = 512,
    ):
        super().__init__()
        self.encoder    = EEGEncoder(embed_dim=token_dim)
        self.attention  = EEGTransformerEncoder(
            token_dim=token_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.separation = EEGFeatureSeparation(
            token_dim=token_dim,
            appearance_dim=appearance_dim,
            dropout=dropout,
        )

    def forward(self, eeg: torch.Tensor) -> dict:
        """
        Args:
            eeg : [B, 250, 64]  raw EEG batch from DataLoader
        Returns:
            dict of 6 L2-normalised branch latents
        """
        tokens    = self.encoder(eeg)         # [B, 62, token_dim]
        embedding = self.attention(tokens)    # [B, 62, token_dim]
        latents   = self.separation(embedding) # {6 branches}
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
    """Enforces cross-correlation orthogonality between key latents using batch-wise L2 normalization.
    
    This prevents representation collapse: if a branch collapses to a constant, 
    its batch-wise L2-normalized vector becomes a constant vector of ones (scaled by 1/sqrt(B)),
    resulting in a cosine similarity of 1.0 (maximum penalty) with any other collapsed branch.
    Only orthogonal, diverse features can achieve the minimum loss of 0.0.
    """
    obj = latents["object"]          # [B, 128]
    subj = latents["subject"]        # [B, 64]
    app = latents["appearance"]      # [B, 512]

    # L2 normalize along the batch dimension (dim=0) to represent column cosine similarities
    eps = 1e-8
    obj_norm  = F.normalize(obj,  p=2, dim=0, eps=eps)
    subj_norm = F.normalize(subj, p=2, dim=0, eps=eps)
    app_norm  = F.normalize(app,  p=2, dim=0, eps=eps)

    # Cross-correlation matrices: elements represent cosine similarities between feature columns
    C_obj_subj = torch.matmul(obj_norm.T, subj_norm)          # [128, 64]
    C_subj_app = torch.matmul(subj_norm.T, app_norm)          # [64, 512]

    # Penalize non-zero entries: Decouple subject from object, and subject from appearance (blocks subject leakage).
    # Object and Appearance are allowed to correlate (so CLIP guide can shape category latents).
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
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(
    data_dir:       str   = "data/EEGdatanpy",
    image_dir:      str   = "data/image",
    checkpoint_dir: str   = "checkpoints/phase06",
    num_epochs:     int   = 500,
    clip_patience:  int   = 40,    # silence CLIP after this many epochs without improvement
    clip_threshold: float = 0.001, # minimum cosine similarity gain to count as improvement
    clip_warmup_epochs: int = 50,  # epochs before CLIP plateau checking starts
    lr:               float = 1e-4,
    weight_decay:     float = 0.01,  # Set to 0.01 to allow visual categories learning
    w_object:         float = 1.0,
    w_subject:        float = 0.5,
    w_clip:           float = 0.5,
    w_ortho:          float = 15.0,  # Barlow cross-correlation orthogonality loss
    w_intra:          float = 2.5,   # Reduced to 2.5 to allow cluster slack for unseen visual stimuli
    w_adv:            float = 1.0,   # Weight of the Subject-Adversarial (DANN) loss
    appearance_dim:   int   = 512,
    token_dim:        int   = 224, # Increased to 224 (80% params scaling)
    num_heads:        int   = 4,   # 4 heads (each head = 56-dim)
    ff_dim:           int   = 416, # Increased to 416 (80% params scaling)
    num_layers:       int   = 2,   # 2 layers for noise filtering depth
    dropout:          float = 0.32, # Increased to 0.32 to match larger capacity
    save_every:       int   = 10,
    log_every:        int   = 50,
    # ── EEG Augmentation ──────────────────────────────────────────────────
    aug_noise_std:        float = 0.09,  # Increased to 0.09 to match larger capacity
    aug_channel_drop_p:   float = 0.10,  # Restored to 0.10
    aug_time_shift_max:   int   = 8,     # Increased to 8 to prevent visual instance memorization
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
        appearance_dim=appearance_dim,
    ).to(device)
    ce_loss = nn.CrossEntropyLoss()

    # Training-only classification heads:
    # 1. Object: latent [B,128] → class logits [B,72]
    # 2. Subject: latent [B,64]  → subject logits [B,12]
    # 3. Subject Adversary (DANN): latent [B,128] → subject logits [B,12]
    obj_classifier  = nn.Linear(128, len(cat_index), bias=False).to(device)
    subj_classifier = nn.Linear(64, len(subj_index), bias=False).to(device)
    subj_adversary  = nn.Linear(128, len(subj_index), bias=False).to(device)

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
    print(f"  Epochs   : {num_epochs}  (CLIP auto-silenced on plateau)")
    print(f"  Params   : {total_params:,}")
    print(f"  Losses   : Object-CE(w={w_object}) + Subject(w={w_subject}) + CLIP(w={w_clip} until plateau) + Barlow Ortho(w={w_ortho}) + SubjAdv(w={w_adv})")
    print(f"  CLIP patience : {clip_patience} epochs  |  threshold : {clip_threshold}  |  warmup : {clip_warmup_epochs} epochs")
    print(f"  LR: {lr}")
    print(f"{'='*60}\n")


     # ── CLIP plateau tracking ──────────────────────────────────────────────────
    # Tracks cosine similarity (not loss) — higher is better.
    clip_active       = use_clip and clip_cache is not None
    best_clip_cosim   = -float("inf")   # best cosine similarity seen so far
    clip_no_improve   = 0               # consecutive epochs without improvement
    clip_stopped_ep   = None            # epoch at which CLIP was silenced

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

            # ── Loss 1: Object Classification (Normalized Softmax ×30) ──────────
            w_norm     = F.normalize(obj_classifier.weight, dim=1)   # [72, 128]
            obj_logits = latents["object"] @ w_norm.T * 30.0         # [B, 72]
            l1 = ce_loss(obj_logits, cat_t)

            # ── Loss 2: Subject Classification (Normalized Softmax ×30) ──────────
            # Force Subject branch to learn a stable representation of subject identity
            w_subj_norm = F.normalize(subj_classifier.weight, dim=1)  # [12, 64]
            subj_logits = latents["subject"] @ w_subj_norm.T * 30.0    # [B, 12]
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

            # ── Loss 7: Subject-Adversarial Loss (Strip subject from Object branch) ──
            # Alpha increases linearly over first 100 epochs to 1.0, then remains 1.0
            alpha_adv = min(1.0, epoch / 100.0)
            subj_adv_logits = subj_adversary(grad_reverse(latents["object"], alpha_adv))
            l_adv = ce_loss(subj_adv_logits, subj_t)

            # ── Total loss ────────────────────────────────────────────────────
            # w_subject weights both the Subject Invariance (l2) and Subject classification (l_subj_ce)
            loss = w_object * l1 + w_subject * (l_subj_ce + l2) + current_w_clip * l3 + w_ortho * l_ortho + w_intra * l_intra + w_adv * l_adv

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
        clip_status = f"cosim={avg_cosim:.4f}" if clip_active else f"OFF (ep{clip_stopped_ep})"

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

        # ── CLIP plateau check ─────────────────────────────────────────────
        if clip_active:
            if avg_cosim - best_clip_cosim > clip_threshold:
                best_clip_cosim = avg_cosim
                clip_no_improve  = 0
            else:
                if epoch > clip_warmup_epochs:
                    clip_no_improve += 1
                    if clip_no_improve >= clip_patience:
                        clip_active     = False
                        clip_stopped_ep = epoch
                        print(
                            f"  [CLIP Silence] CLIP loss plateaued at epoch {epoch} "
                            f"(cosim={best_clip_cosim:.4f}, no gain for {clip_patience} epochs). "
                            f"Silencing CLIP — SupCon+Subject continue."
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
        w_adv          = 1.0,   # Subject-Adversarial (DANN) weight
        appearance_dim = 512,
        token_dim      = 224,   # 80% parameter scaling
        num_heads      = 4,     # 4 heads (each head = 56-dim)
        ff_dim         = 416,   # 80% parameter scaling
        num_layers     = 2,     # 2 layers for noise filtering depth
        dropout        = 0.32,  # Increased to 0.32 to match larger capacity
        aug_noise_std  = 0.09,  # Increased to 0.09 to match larger capacity
        aug_time_shift_max = 8,  # Increased to 8 to prevent visual instance memorization
        save_every     = 10,
        log_every      = 50,
    )
