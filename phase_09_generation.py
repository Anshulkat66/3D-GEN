"""Phase 9: EEG-to-Image Reconstruction.

This script loads the frozen Phase 6 pipeline and the trained Phase 7 Ridge Fusion model,
extracts semantic conditioning sequences from test set EEG signals, and feeds them
directly to Stable Diffusion v1.5 to reconstruct stimulus images.

The generated images are saved side-by-side with the original stimulus images in the
outputs/generation/ directory for visual validation.
"""

import os
import json
import random
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline

from phase_02_DL import create_dataloader
from phase_06_training import EEGPipeline, extract_category
from phase_07_fusion import EEGFusionRidge


# ─────────────────────────────────────────────────────────────────────────────
# 1. Hugging Face CLIP Text Sequence Extractor
# ─────────────────────────────────────────────────────────────────────────────

def get_hf_clip_text_sequence(text_encoder, tokenizer, text: str, device: torch.device) -> torch.Tensor:
    """Extracts the unpooled token sequence from Hugging Face's CLIP text model.

    This uses the text encoder already loaded inside the Stable Diffusion pipeline,
    saving memory and ensuring perfect compatibility.

    Args:
        text_encoder: Stable Diffusion's text_encoder (CLIPTextModel)
        tokenizer: Stable Diffusion's tokenizer (CLIPTokenizer)
        text: Input string prompt
        device: Target hardware device
    Returns:
        unpooled_embeddings: [1, 77, 768] tensor
    """
    with torch.no_grad():
        inputs = tokenizer(
            text,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        input_ids = inputs.input_ids.to(device)  # [1, 77]
        outputs = text_encoder(input_ids)
        # last_hidden_state contains the unpooled token embeddings: [1, 77, 768]
        embeds = outputs.last_hidden_state.type(torch.float32)
    return embeds


# ─────────────────────────────────────────────────────────────────────────────
# 2. Main Reconstruction Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_reconstruction(
    data_dir: str = "data/EEGdatanpy",
    image_dir: str = "data/image",
    phase06_path: str = "checkpoints/phase06_2.24_leak_prevented/best_model.pt",
    phase07_path: str = "checkpoints/phase07/best_fusion.pt",
    output_dir: str = "outputs/generation",
    num_samples: int = 5,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 30,
    random_seed: int = 42,
):
    """Generates images from test set EEG signals using Stable Diffusion."""
    # Set seed for reproducible generation layouts
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ── Resolve Checkpoint Paths ──────────────────────────────────────────────
    p06_path = Path(phase06_path)
    p07_path = Path(phase07_path)
    if not p06_path.exists():
        raise FileNotFoundError(f"Phase 6 checkpoint not found at: {p06_path}")
    if not p07_path.exists():
        raise FileNotFoundError(f"Phase 7 checkpoint not found at: {p07_path}")

    print(f"Loading Phase 6 checkpoint: {p06_path.name}")
    checkpoint_p06 = torch.load(p06_path, map_location=device)
    
    print(f"Loading Phase 7 checkpoint: {p07_path.name}")
    checkpoint_p07 = torch.load(p07_path, map_location=device)

    cat_index = checkpoint_p06["cat_index"]

    # ── Resolve Configs & Parameters ──────────────────────────────────────────
    config_p06_path = p06_path.parent / "config.json"
    token_dim, num_heads, ff_dim, num_layers = 256, 4, 512, 2
    appearance_dim = 768   # ViT-L/14 image embedding dim (must match Phase 6 CLIP model)

    if config_p06_path.exists():
        try:
            with open(config_p06_path, "r") as f:
                config = json.load(f)
            token_dim = config.get("token_dim", token_dim)
            num_heads = config.get("num_heads", num_heads)
            ff_dim = config.get("ff_dim", ff_dim)
            num_layers = config.get("num_layers", num_layers)
            appearance_dim = config.get("appearance_dim", appearance_dim)
        except Exception as e:
            print(f"  ⚠ Warning: Failed to read config.json, using defaults ({e})")

    # ── Instantiate and Load Models ───────────────────────────────────────────
    print("Loading models...")
    
    # 1. Frozen Phase 6 Pipeline
    frozen_pipeline = EEGPipeline(
        token_dim=token_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
        dropout=0.0,                # BranchMLP dropout — disabled at inference
        transformer_dropout=0.0,    # Transformer dropout — disabled at inference (Bug #21 fix)
        appearance_dim=appearance_dim,
    ).to(device)
    frozen_pipeline.load_state_dict(checkpoint_p06["model_state_dict"])
    frozen_pipeline.eval()
    for p in frozen_pipeline.parameters():
        p.requires_grad = False

    # 2. Frozen Phase 7 Ridge Fusion Model
    # Load via load_state_dict() so ALL saved state is restored:
    #   - W, b  : Ridge regression weights and bias
    #   - x_mean, x_std : training-set scaler stats (Bug #15 / Bug #18 fix)
    # Old code extracted only W and b manually, leaving x_mean/x_std as
    # zeroed/uninitialized buffers — causing wrong standardization at inference.
    W      = checkpoint_p07["model_state_dict"]["W"]
    b      = checkpoint_p07["model_state_dict"]["b"]
    x_mean = checkpoint_p07["x_mean"]
    x_std  = checkpoint_p07["x_std"]
    fusion_model = EEGFusionRidge(W, b, x_mean, x_std).to(device)
    fusion_model.load_state_dict(checkpoint_p07["model_state_dict"])
    fusion_model.eval()
    for p in fusion_model.parameters():
        p.requires_grad = False

    print("EEG encoding and translation modules loaded successfully.")

    # ── Load Stable Diffusion Pipeline ────────────────────────────────────────
    print("\nLoading Stable Diffusion v1.5...")
    sd_dtype = torch.float16 if device.type == "cuda" else torch.float32
    
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=sd_dtype,
        safety_checker=None, # Disable checker to save memory and avoid false triggers
        requires_safety_checker=False
    ).to(device)
    
    if device.type == "cuda":
        print("  Enabling memory-saving attention slicing...")
        pipe.enable_attention_slicing()

    # ── Precompute Unconditioned Embedding (Negative Prompt) ──────────────────
    print("Precomputing negative (unconditioned) prompt embeddings...")
    # Extracts the unpooled CLIP sequence [1, 77, 768] for an empty string
    negative_embeds = get_hf_clip_text_sequence(pipe.text_encoder, pipe.tokenizer, "", device)
    negative_embeds = negative_embeds.to(dtype=sd_dtype)

    # ── Load Test Dataset ─────────────────────────────────────────────────────
    print("\nLoading test set dataset...")
    # Shuffling enables picking random visual categories to test on
    test_loader = create_dataloader(data_dir=data_dir, batch_size=1, split="test")
    test_samples = list(test_loader)
    print(f"Loaded {len(test_samples)} test set trials.")

    # Select random trials to generate
    selected_indices = random.sample(range(len(test_samples)), min(num_samples, len(test_samples)))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting Image Generation ({num_samples} samples)...")
    for i, idx in enumerate(selected_indices):
        eeg, labels, _ = test_samples[idx]
        eeg = eeg.to(device)
        label = labels[0]
        category = extract_category(label)
        
        print(f"\n[{i+1}/{num_samples}] Trial Index: {idx} | Category: '{category}' | Stimulus: '{label}'")

        # 1. Extract latents from frozen Phase 6
        with torch.no_grad():
            latents = frozen_pipeline(eeg)
            latents.pop("subject", None)  # Discard subject latent (not used in fusion)
            latents.pop("shared",  None)  # Discard DANN shared rep (added by GRL fix)

        # 2. Fuse latents to CLIP space using Phase 7 Ridge
        with torch.no_grad():
            prompt_embeds = fusion_model(latents) # [1, 77, 768]
            prompt_embeds = prompt_embeds.to(dtype=sd_dtype)

        # 3. Generate Image via Stable Diffusion
        print("  Generating image via Stable Diffusion...")
        with torch.autocast(device.type):
            generated_image = pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_embeds,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).images[0]

        # 4. Load Original Stimulus Image
        img_root = Path(image_dir)
        original_img_path = img_root / category / f"{label}.png"
        original_image = None
        
        if original_img_path.exists():
            original_image = Image.open(original_img_path)
        else:
            print(f"  ⚠ Original stimulus image not found at: {original_img_path}")

        # 5. Plot Side-by-Side Comparison
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Plot original stimulus
        if original_image is not None:
            axes[0].imshow(original_image)
            axes[0].set_title(f"Original Stimulus\n({label})", fontsize=12, fontweight="bold")
        else:
            axes[0].text(0.5, 0.5, "Original Stimulus\nNot Found", 
                         ha="center", va="center", fontsize=12, color="gray")
            axes[0].set_title("Original Stimulus", fontsize=12, fontweight="bold")
        axes[0].axis("off")

        # Plot reconstruction
        axes[1].imshow(generated_image)
        axes[1].set_title(f"EEG Reconstruction\n(Target: {category})", fontsize=12, fontweight="bold")
        axes[1].axis("off")

        # Save plot
        save_path = output_dir / f"reconstruction_{label}.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  ✅ Saved comparison plot to: {save_path.resolve()}")

    print(f"\n{'='*60}")
    print(f"  RECONSTRUCTION COMPLETE!")
    print(f"  All comparison plots saved in: {output_dir.resolve()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 9: EEG-to-Image Generation via Stable Diffusion")
    parser.add_argument("--data_dir", type=str, default="data/EEGdatanpy",
                        help="Path to pre-epoched, pre-normalized EEG data")
    parser.add_argument("--image_dir", type=str, default="data/image",
                        help="Path to original stimulus images")
    parser.add_argument("--phase06_path", type=str, default="checkpoints/phase06_2.24_leak_prevented/best_model.pt",
                        help="Path to the frozen Phase 6 checkpoint")
    parser.add_argument("--phase07_path", type=str, default="checkpoints/phase07/best_fusion.pt",
                        help="Path to the frozen Phase 7 Ridge Fusion checkpoint")
    parser.add_argument("--output_dir", type=str, default="outputs/generation",
                        help="Directory to save comparison plots")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of test samples to reconstruct")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="Classifier-Free Guidance (CFG) scale")
    parser.add_argument("--num_inference_steps", type=int, default=30,
                        help="Denoising steps for Stable Diffusion")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for sample layout reproducibility")

    args = parser.parse_args()

    run_reconstruction(
        data_dir=args.data_dir,
        image_dir=args.image_dir,
        phase06_path=args.phase06_path,
        phase07_path=args.phase07_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        random_seed=args.random_seed
    )
