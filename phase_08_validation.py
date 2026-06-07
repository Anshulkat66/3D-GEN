"""Phase 8: Latent Validation.

Loads the trained Phase 6 pipeline checkpoint, runs inference on the train
and test splits, and performs evaluation:
  1. Computes Object class centroids on the training set.
  2. Evaluates Top-1 and Top-5 classification accuracy on the test set using
     Nearest Centroid matching.
  3. Computes average cosine similarity of Appearance latents vs. CLIP image embeddings.
  4. Generates a t-SNE plot of test Object latents to check for semantic clustering.
"""

import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from phase_02_DL import create_dataloader
from phase_06_training import EEGPipeline, build_clip_cache_images, CLIP_AVAILABLE, extract_category


def run_validation(
    data_dir:        str = "data/EEGdatanpy",
    image_dir:       str = "data/image",
    checkpoint_dir:  str = "checkpoints/phase06",
    checkpoint_path: str = None,          # if None → auto-picks latest epoch
    output_dir:      str = "outputs/validation",
    perplexity:      int = 30,
    random_seed:     int = 42,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ── Resolve Checkpoint Path (auto-pick latest if not specified) ───────────
    if checkpoint_path is None:
        ckpt_dir = Path(checkpoint_dir)
        candidates = sorted(
            ckpt_dir.glob("checkpoint_epoch*.pt"),
            key=lambda p: int(p.stem.replace("checkpoint_epoch", ""))
        )
        if not candidates:
            raise FileNotFoundError(f"No checkpoints found in: {ckpt_dir}")
        ckpt_path = candidates[-1]   # highest epoch number
        print(f"Auto-selected latest checkpoint: {ckpt_path.name}")
    else:
        ckpt_path = Path(checkpoint_path)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)

    cat_index = checkpoint["cat_index"]
    subj_index = checkpoint["subj_index"]
    n_categories = len(cat_index)

    print(f"  Checkpoint Epoch : {checkpoint['epoch']}")
    print(f"  Training Loss    : {checkpoint['loss']:.4f}")
    print(f"  Categories       : {n_categories}")
    print(f"  Subjects         : {len(subj_index)}")

    # Reverse mapping for display/legends
    cat_names = {v: k for k, v in cat_index.items()}

    # ── Load config.json if it exists to get architecture details ─────────────
    import json
    config_path = ckpt_path.parent / "config.json"
    token_dim, num_heads, ff_dim, num_layers, dropout = 256, 4, 512, 2, 0.1
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            token_dim = config.get("token_dim", token_dim)
            num_heads = config.get("num_heads", num_heads)
            ff_dim = config.get("ff_dim", ff_dim)
            num_layers = config.get("num_layers", num_layers)
            dropout = config.get("dropout", dropout)
            print("Loaded architecture parameters from config.json:")
            print(f"  token_dim={token_dim}, num_heads={num_heads}, ff_dim={ff_dim}, num_layers={num_layers}")
        except Exception as e:
            print(f"  ⚠ Warning: Failed to read config.json, using defaults ({e})")
    else:
        print("  config.json not found, using legacy default architecture.")

    # ── Instantiate and Load Model ────────────────────────────────────────────
    model = EEGPipeline(
        token_dim=token_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
        dropout=dropout,
        appearance_dim=512,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # ── Loaders ───────────────────────────────────────────────────────────────
    # batch_size=256, shuffle=False (we want clean arrays for evaluation)
    train_loader = create_dataloader(data_dir=data_dir, batch_size=256, split="train")
    test_loader  = create_dataloader(data_dir=data_dir, batch_size=256, split="test")

    # Override shuffle for evaluation
    train_loader = torch.utils.data.DataLoader(
        train_loader.dataset, batch_size=256, shuffle=False, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_loader.dataset, batch_size=256, shuffle=False, num_workers=0
    )

    print(f"Train samples for centroids: {len(train_loader.dataset)}")
    print(f"Test samples for validation : {len(test_loader.dataset)}")

    # ──────────────────────────────────────────────────────────────────────────
    # Step 1: Compute Class Centroids from Training Set
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[1/4] Extracting training latents to compute class centroids...")
    train_latents = []
    train_cats = []

    with torch.no_grad():
        for eeg, labels, _ in train_loader:
            eeg = eeg.to(device)
            latents = model(eeg)
            train_latents.append(latents["object"].cpu())
            
            cat_ints = [cat_index[extract_category(lbl)] for lbl in labels]
            train_cats.extend(cat_ints)

    train_latents = torch.cat(train_latents, dim=0)  # [N_train, 128]
    train_cats    = np.array(train_cats)            # [N_train]

    # Compute mean vector for each of the 72 classes
    centroids = torch.zeros(n_categories, 128)
    for c in range(n_categories):
        indices = np.where(train_cats == c)[0]
        if len(indices) > 0:
            # Mean representation of the category
            mean_vec = train_latents[indices].mean(dim=0)
            # Re-normalize to unit sphere
            centroids[c] = F.normalize(mean_vec, dim=-1)
        else:
            print(f"  ⚠ Warning: Class {c} ('{cat_names[c]}') has no training samples!")

    # ──────────────────────────────────────────────────────────────────────────
    # Step 2: Evaluate Object and Appearance Latents on Test Set
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[2/4] Running inference on the test set (unseen stimuli)...")
    test_obj_latents = []
    test_app_latents = []
    test_cats = []
    test_labels = []

    with torch.no_grad():
        for eeg, labels, _ in test_loader:
            eeg = eeg.to(device)
            latents = model(eeg)
            test_obj_latents.append(latents["object"].cpu())
            test_app_latents.append(latents["appearance"].cpu())
            
            cat_ints = [cat_index[extract_category(lbl)] for lbl in labels]
            test_cats.extend(cat_ints)
            test_labels.extend(list(labels))

    test_obj_latents = torch.cat(test_obj_latents, dim=0)  # [N_test, 128]
    test_app_latents = torch.cat(test_app_latents, dim=0)  # [N_test, 512]
    test_cats        = torch.tensor(test_cats, dtype=torch.long)

    # ── Calculate Classification Accuracies (Nearest Centroid) ────────────────
    # Cosine similarities: dot product of L2-normalized vectors
    # [N_test, 128] @ [128, 72] → [N_test, 72] similarities
    similarities = torch.matmul(test_obj_latents, centroids.T)

    # Top-1 accuracy
    top1_preds = torch.argmax(similarities, dim=1)
    top1_correct = (top1_preds == test_cats).sum().item()
    top1_acc = top1_correct / len(test_cats)

    # Top-5 accuracy
    _, top5_preds = torch.topk(similarities, k=5, dim=1)  # [N_test, 5]
    top5_correct = 0
    for i in range(len(test_cats)):
        if test_cats[i] in top5_preds[i]:
            top5_correct += 1
    top5_acc = top5_correct / len(test_cats)

    # ── Calculate CLIP Cosine Similarity ──────────────────────────────────────
    avg_clip_sim = 0.0
    if CLIP_AVAILABLE:
        print("  Extracting test CLIP target embeddings...")
        test_clip_cache = build_clip_cache_images(test_labels, image_dir, device)
        
        clip_sims = []
        for i, lbl in enumerate(test_labels):
            if lbl in test_clip_cache:
                target_feat = F.normalize(test_clip_cache[lbl].to(device), dim=-1)
                test_feat   = test_app_latents[i].to(device)
                
                sim = torch.dot(test_feat, target_feat).item()
                clip_sims.append(sim)
        
        if clip_sims:
            avg_clip_sim = np.mean(clip_sims)
    else:
        print("  ⚠ CLIP not available; skipping CLIP cosine similarity evaluation.")

    # Print results
    print(f"\n" + "=" * 60)
    print("  TEST SET GENERALIZATION METRICS")
    print("=" * 60)
    print(f"  Top-1 Accuracy (Nearest Centroid): {top1_acc*100:6.2f}%")
    print(f"  Top-5 Accuracy (Nearest Centroid): {top5_acc*100:6.2f}%")
    if CLIP_AVAILABLE:
        print(f"  Avg CLIP Cosine Similarity      : {avg_clip_sim:8.4f}")
    print("=" * 60)

    # ──────────────────────────────────────────────────────────────────────────
    # ──────────────────────────────────────────────────────────────────────────
    # Step 3: generate t-SNE Plot of Test Set
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[3/4] Running t-SNE dimensionality reduction on test object latents...")
    
    # Run t-SNE on test
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_seed,
        n_iter=1000,
        init="pca",
        learning_rate="auto",
    )
    tsne_results = tsne.fit_transform(test_obj_latents.numpy())

    # Create test plot
    print("\n[4/4] Generating t-SNE scatter plot for test set...")
    plt.figure(figsize=(12, 10))
    
    scatter = plt.scatter(
        tsne_results[:, 0],
        tsne_results[:, 1],
        c=test_cats.numpy(),
        cmap="tab20",
        alpha=0.8,
        s=20,
        edgecolors="none",
    )
    
    cbar = plt.colorbar(scatter)
    cbar.set_label("Category Class ID", fontsize=11)
    
    plt.title(
        f"t-SNE of Test Set Object Latents (Unseen Stimuli)\n"
        f"Top-1 Acc: {top1_acc*100:.1f}% | Top-5 Acc: {top5_acc*100:.1f}%",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("t-SNE Axis 1", fontsize=11)
    plt.ylabel("t-SNE Axis 2", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.4)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_epoch = checkpoint.get("epoch", "unknown")
    img_path = out_dir / f"tsne_test_epoch{ckpt_epoch}.png"
    plt.savefig(img_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Test t-SNE plot saved to: {img_path.resolve()}")

    # ──────────────────────────────────────────────────────────────────────────
    # Step 4: generate t-SNE Plot of Training Set (Subsampled for speed)
    # ──────────────────────────────────────────────────────────────────────────
    print("\nRunning t-SNE dimensionality reduction on training set (subsampled for speed)...")
    n_train_total = len(train_latents)
    np.random.seed(random_seed)
    sub_indices = np.random.choice(n_train_total, min(2000, n_train_total), replace=False)
    
    train_latents_sub = train_latents[sub_indices]
    train_cats_sub    = train_cats[sub_indices]

    tsne_train = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_seed,
        n_iter=1000,
        init="pca",
        learning_rate="auto",
    )
    tsne_train_results = tsne_train.fit_transform(train_latents_sub.numpy())

    print("Generating t-SNE scatter plot for training set...")
    plt.figure(figsize=(12, 10))
    
    scatter_train = plt.scatter(
        tsne_train_results[:, 0],
        tsne_train_results[:, 1],
        c=train_cats_sub,
        cmap="tab20",
        alpha=0.8,
        s=20,
        edgecolors="none",
    )
    
    cbar_train = plt.colorbar(scatter_train)
    cbar_train.set_label("Category Class ID", fontsize=11)
    
    plt.title(
        f"t-SNE of Training Set Object Latents — Epoch {ckpt_epoch} (Subsampled 2000 trials)\n"
        "Training Reference — Compare with Test Plot to Assess Generalization",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("t-SNE Axis 1", fontsize=11)
    plt.ylabel("t-SNE Axis 2", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.4)

    img_path_train = out_dir / f"tsne_train_epoch{ckpt_epoch}.png"
    plt.savefig(img_path_train, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Training t-SNE plot saved to: {img_path_train.resolve()}")
    
    print("\nValidation and visualization run complete!")


if __name__ == "__main__":
    run_validation()

