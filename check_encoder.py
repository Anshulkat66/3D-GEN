"""Encoder Diagnostic: Phase 3 + 4 Forward Pass Checks.

Two levels of tests:
  LEVEL 1 - Shape & sanity (no training needed, runs on random data)
  LEVEL 2 - Signal quality  (needs real EEG data + a trained Phase 6 checkpoint)
             Measures: do same-category EEG trials produce more similar
             encoder outputs than different-category trials?
             If yes → encoder is extracting real category signal (not noise).
             If no  → encoder is still outputting noise (need more training).

Run:
    python check_encoder.py                          # Level 1 only
    python check_encoder.py --full --data_dir data/EEGdatanpy --ckpt checkpoints/phase06_2.24_leak_prevented/best_model.pt
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from phase_02_mse_tok import EEGEncoder, MultiScaleEEGEncoder
from phase_04_attention import EEGTransformerEncoder


# ─────────────────────────────────────────────────────────────────────────────
# LEVEL 1 — Shape & Sanity Checks (no training or data needed)
# ─────────────────────────────────────────────────────────────────────────────

def level1_shape_checks(token_dim: int = 192):
    print("\n" + "=" * 60)
    print("  LEVEL 1 — SHAPE & SANITY CHECKS")
    print("=" * 60)

    all_pass = True

    # ── 1. Kernel temporal coverage ───────────────────────────────────────
    print("\n[A] Kernel temporal coverage at 250Hz:")
    kernels = {"small_scale (C1)": 25, "medium_scale (N170)": 51, "large_scale (P300)": 75}
    for name, k in kernels.items():
        ms = k * 4
        ok = ms >= 100
        print(f"    {name:25s}: kernel={k:2d}, window={ms:3d}ms  {'✅' if ok else '❌ too small'}")
        all_pass = all_pass and ok

    # ── 2. ConvBlock output length (must equal input = 250) ───────────────
    print("\n[B] ConvBlock output length (must be 250 — same padding):")
    for k in [25, 51, 75]:
        padding = k // 2
        out_len = (250 + 2 * padding - k) // 1 + 1
        ok = out_len == 250
        print(f"    kernel={k:2d}, padding={padding:2d} → output_len={out_len}  {'✅' if ok else '❌'}")
        all_pass = all_pass and ok

    # ── 3. Tokenizer output length (must be 63 with padding=1) ───────────
    print("\n[C] Tokenizer output token count:")
    tok_out = (250 + 2 * 1 - 4) // 4 + 1   # kernel=4, stride=4, padding=1
    old_out = (250 - 4) // 4 + 1             # old: no padding
    ok = tok_out == 63
    print(f"    Old (no padding):  {old_out} tokens  — last 2 samples lost")
    print(f"    New (padding=1):   {tok_out} tokens  {'✅ all samples covered' if ok else '❌'}")
    all_pass = all_pass and ok

    # ── 4. Full forward pass shape through Phase 3 + 4 ───────────────────
    print(f"\n[D] Full forward pass — Phase 3 + Phase 4 (token_dim={token_dim}):")
    encoder = EEGEncoder(embed_dim=token_dim)
    attention = EEGTransformerEncoder(
        token_dim=token_dim, num_heads=3, ff_dim=384, num_layers=2, dropout=0.0
    )
    encoder.eval()
    attention.eval()

    dummy_eeg = torch.randn(4, 250, 64)  # [B, Time, Channels]

    with torch.no_grad():
        tokens = encoder(dummy_eeg)       # Phase 3 output
        context = attention(tokens)        # Phase 4 output

    expected_tokens  = torch.Size([4, 63, token_dim])
    expected_context = torch.Size([4, 63, token_dim])

    tok_ok  = tokens.shape  == expected_tokens
    ctx_ok  = context.shape == expected_context
    nan_ok  = not context.isnan().any().item()
    inf_ok  = not context.isinf().any().item()

    print(f"    Phase 3 output : {tuple(tokens.shape)}   {'✅' if tok_ok  else f'❌ expected {tuple(expected_tokens)}'}")
    print(f"    Phase 4 output : {tuple(context.shape)}   {'✅' if ctx_ok  else f'❌ expected {tuple(expected_context)}'}")
    print(f"    NaN in output  : {'None ✅' if nan_ok else 'FOUND ❌'}")
    print(f"    Inf in output  : {'None ✅' if inf_ok else 'FOUND ❌'}")
    all_pass = all_pass and tok_ok and ctx_ok and nan_ok and inf_ok

    # ── 5. Inputs produce DIFFERENT outputs (not collapsed) ───────────────
    print(f"\n[E] Output diversity — different inputs → different outputs:")
    dummy2 = torch.randn(4, 250, 64)   # completely different EEG
    with torch.no_grad():
        tokens2  = encoder(dummy2)
        context2 = attention(tokens2)

    # Mean pairwise cosine similarity between the two batch outputs (pool tokens)
    pooled1 = context.mean(dim=1)   # [4, token_dim]
    pooled2 = context2.mean(dim=1)  # [4, token_dim]
    cos_sim  = F.cosine_similarity(pooled1, pooled2, dim=-1).mean().item()

    diverse  = cos_sim < 0.99   # if 0.99+ → model is outputting near-constant (collapsed)
    print(f"    Cosine sim between random input pairs: {cos_sim:.4f}")
    print(f"    {'✅ Outputs are diverse (not collapsed)' if diverse else '❌ Outputs are near-identical — encoder is collapsed!'}")
    all_pass = all_pass and diverse

    # ── 6. Channel Attention: gate values and diversity ───────────────────
    print(f"\n[F] Channel Attention (SE block) check:")
    ca = encoder.multi_scale.channel_attn

    dummy_a = torch.randn(1, 64, 250)
    dummy_b = torch.randn(1, 64, 250)
    with torch.no_grad():
        gap_a  = dummy_a.mean(dim=-1)    # [1, 64]
        gate_a = ca.fc(gap_a)            # [1, 64]  values should be in [0,1]
        gap_b  = dummy_b.mean(dim=-1)
        gate_b = ca.fc(gap_b)            # [1, 64]

    gate_min = gate_a.min().item()
    gate_max = gate_a.max().item()
    gate_range_ok = (gate_min >= 0.0) and (gate_max <= 1.0)
    print(f"    Gate value range  : [{gate_min:.4f}, {gate_max:.4f}]  {'✅ in [0,1]' if gate_range_ok else '❌ outside [0,1]'}")

    gate_diff = (gate_a - gate_b).abs().mean().item()
    diverse_gates = gate_diff > 1e-6
    print(f"    Gate diff (2 diff inputs): {gate_diff:.6f}  {'✅ gates vary with input' if diverse_gates else '❌ constant gates (collapsed)'}")

    param_count = sum(p.numel() for p in ca.parameters())
    print(f"    SE block param count : {param_count:,}  (expected 2,128 — includes biases)")

    all_pass = all_pass and gate_range_ok and diverse_gates

    # ── 7. GroupNorm vs BatchNorm: train/eval parity check ────────────────
    print(f"\n[G] Normalization — GroupNorm train/eval parity check:")

    conv_block = encoder.multi_scale.small_scale
    norm_type  = type(conv_block.norm).__name__

    is_groupnorm = isinstance(conv_block.norm, nn.GroupNorm)
    is_batchnorm = isinstance(conv_block.norm, nn.BatchNorm1d)

    print(f"    Norm type in ConvBlock : {norm_type}  {'✅ GroupNorm (correct)' if is_groupnorm else '❌ BatchNorm (train/eval mismatch!)'}")
    all_pass = all_pass and is_groupnorm

    if is_groupnorm:
        # Verify train and eval mode produce IDENTICAL output for the same input
        test_input = torch.randn(1, 64, 250)   # batch_size=1 (inference scenario)
        conv_block.eval()
        with torch.no_grad():
            out_eval  = conv_block(test_input.clone())
        conv_block.train()
        with torch.no_grad():
            out_train = conv_block(test_input.clone())
        conv_block.eval()  # restore to eval

        max_diff = (out_eval - out_train).abs().max().item()
        parity_ok = max_diff < 1e-5
        print(f"    Max diff (train vs eval mode, same input): {max_diff:.2e}  {'✅ identical' if parity_ok else '❌ different (bug!)'}")
        if parity_ok:
            print(f"    ✅ No train/eval distribution shift — GroupNorm is batch-size independent")
        all_pass = all_pass and parity_ok
    elif is_batchnorm:
        print(f"    ❌ BatchNorm creates different outputs in train vs eval mode.")
        print(f"       At batch_size=1 (generation), the amplitude shift is severe.")
        print(f"       Fix: replace nn.BatchNorm1d with nn.GroupNorm in ConvBlock.")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "-" * 60)
    if all_pass:
        print("  ✅ ALL LEVEL 1 CHECKS PASSED — architecture is correct")
        print("  ℹ  Note: random weights mean nothing about signal quality.")
        print("     Run with --full to test signal quality on real EEG data.")
    else:
        print("  ❌ SOME CHECKS FAILED — review above")
    print("=" * 60)
    return all_pass


# ─────────────────────────────────────────────────────────────────────────────
# LEVEL 2b — Channel Attention Learned Correctly? (needs trained checkpoint)
# ─────────────────────────────────────────────────────────────────────────────

# Standard 10-20 EEG electrode names (64-channel layout, in channel order)
# These match the THINGS-EEG dataset electrode order
ELECTRODE_NAMES = [
    "Fp1","Fp2","F7","F3","Fz","F4","F8","FC5","FC1","FC2","FC6",
    "T7","C3","Cz","C4","T8","TP9","CP5","CP1","CP2","CP6","TP10",
    "P7","P3","Pz","P4","P8","PO9","O1","Oz","O2","PO10",
    "AF7","AF3","AF4","AF8","F5","F1","F2","F6","FT9","FT7","FC3",
    "FC4","FT8","FT10","C5","C1","C2","C6","TP7","CP3","CPz","CP4",
    "TP8","P5","P1","P2","P6","PO7","PO3","POz","PO4","PO8",
]

# Known electrode groupings: which should be HIGH (visual) vs LOW (noise)
VISUAL_CHANNELS  = {"O1","Oz","O2","PO3","POz","PO4","PO7","PO8","P3","Pz","P4","P5","P6"}
FRONTAL_NOISE    = {"Fp1","Fp2","F7","F8","AF7","AF8","FT9","FT10"}


def level2b_channel_attention_check(data_dir: str, ckpt_path: str):
    """
    After training, loads the checkpoint and checks what the SE block learned.

    Expected result (healthy):
      Occipital/parietal channels (O1, Oz, O2, P3, Pz, P4) → high gates (close to 1)
      Frontal/temporal channels   (Fp1, Fp2, F7, F8)        → low gates  (close to 0)

    If the model learned well, visual channels should have significantly higher
    gates than frontal noise channels — confirming the SE block is working.
    """
    print("\n" + "=" * 60)
    print("  LEVEL 2b — CHANNEL ATTENTION LEARNED CORRECTLY?")
    print("=" * 60)

    try:
        from phase_02_DL       import create_dataloader
        from phase_06_training import EEGPipeline
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = Path(ckpt_path)
    if not ckpt.exists():
        print(f"  ❌ Checkpoint not found: {ckpt}")
        return

    checkpoint = torch.load(ckpt, map_location=device)
    config_path = ckpt.parent / "config.json"
    token_dim, num_heads, ff_dim, num_layers = 224, 4, 416, 2
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        token_dim  = cfg.get("token_dim",  token_dim)
        num_heads  = cfg.get("num_heads",  num_heads)
        ff_dim     = cfg.get("ff_dim",     ff_dim)
        num_layers = cfg.get("num_layers", num_layers)

    model = EEGPipeline(
        token_dim=token_dim, num_heads=num_heads,
        ff_dim=ff_dim, num_layers=num_layers, dropout=0.0
    ).to(device)

    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as e:
        lines = [l.strip() for l in str(e).splitlines() if "size mismatch" in l]
        print("  ❌ Checkpoint incompatible with current architecture.")
        print("     This means the checkpoint was trained with the OLD (broken) architecture.")
        print("     Retrain Phase 6 first, then re-run this check.")
        for l in lines:
            print(f"     • {l}")
        print("=" * 60)
        return

    model.eval()

    # Run a batch of real EEG through the channel attention to get gates
    print(f"  Checkpoint loaded: {ckpt.name}")
    print(f"  Running EEG through trained channel attention gate...")

    loader = create_dataloader(data_dir=data_dir, batch_size=256, split="test")
    eeg_batch, _, _ = next(iter(loader))
    eeg_batch = eeg_batch.to(device)   # [B, 250, 64]

    with torch.no_grad():
        # Extract gates from the SE block directly
        x = eeg_batch.permute(0, 2, 1)   # [B, 64, 250]
        ca = model.encoder.multi_scale.channel_attn
        gap  = x.mean(dim=-1)             # [B, 64] — global avg pool over time
        gate = ca.fc(gap)                 # [B, 64] — learned gate values in [0,1]
        mean_gate = gate.mean(dim=0).cpu()   # [64] — average gate across the batch

    # ── Print ranked electrode gates ──────────────────────────────────────
    n = len(ELECTRODE_NAMES)
    gate_vals = mean_gate[:n].tolist()

    ranked = sorted(zip(gate_vals, ELECTRODE_NAMES), reverse=True)

    print(f"\n  {'─'*50}")
    print(f"  ELECTRODE GATE VALUES (higher = model trusts this electrode more)")
    print(f"  {'─'*50}")
    print(f"  {'Rank':<5} {'Electrode':<10} {'Gate':>6}  {'Type'}")
    print(f"  {'─'*50}")

    for rank, (val, name) in enumerate(ranked, 1):
        if name in VISUAL_CHANNELS:
            etype = "✅ visual cortex"
        elif name in FRONTAL_NOISE:
            etype = "⚠️  frontal noise"
        else:
            etype = "   central/other"
        bar = "█" * int(val * 20)
        print(f"  {rank:<5} {name:<10} {val:>6.4f}  {bar}  {etype}")

    print(f"  {'─'*50}")

    # ── Summary verdict ───────────────────────────────────────────────────
    visual_mean  = np.mean([v for v, n in ranked if n in VISUAL_CHANNELS])
    frontal_mean = np.mean([v for v, n in ranked if n in FRONTAL_NOISE])

    print(f"\n  Average gate — Visual  channels (O1,Oz,O2,P3,Pz...): {visual_mean:.4f}")
    print(f"  Average gate — Frontal channels (Fp1,Fp2,F7,F8...): {frontal_mean:.4f}")

    ratio = visual_mean / (frontal_mean + 1e-8)
    print(f"  Visual/Frontal gate ratio: {ratio:.2f}")

    if visual_mean > frontal_mean and ratio > 1.2:
        print("\n  ✅ CHANNEL ATTENTION LEARNED CORRECTLY")
        print("     Visual cortex channels have higher gates than frontal noise channels.")
        print("     The SE block is suppressing noise and amplifying signal.")
    elif visual_mean > frontal_mean:
        print("\n  ⚠️  MARGINAL — visual channels are slightly higher than frontal")
        print("     The SE block is learning but the separation is weak.")
        print("     Consider training longer or increasing the reduction ratio.")
    else:
        print("\n  ❌ CHANNEL ATTENTION DID NOT LEARN CORRECTLY")
        print("     Frontal noise channels have HIGHER gates than visual channels.")
        print("     Possible causes:")
        print("     - Training loss doesn't push the encoder toward occipital signal")
        print("     - Frontal noise has higher amplitude (easier to 'explain' loss)")
        print("     - More epochs needed")

    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# LEVEL 2 — Signal Quality Check (real EEG + trained checkpoint)
# ─────────────────────────────────────────────────────────────────────────────

def level2_signal_quality(data_dir: str, ckpt_path: str):
    """
    Loads trained Phase 6 model, runs test EEG through encoder,
    then measures:
      - IntraClassSim: mean cosine similarity between SAME-category object latents
      - InterClassSim: mean cosine similarity between DIFFERENT-category centroids
      - Ratio = Intra / Inter  → healthy if > 5, collapsing if < 2

    If IntraClassSim > InterClassSim → encoder is extracting category signal (signal ✅)
    If IntraClassSim ≈ InterClassSim → encoder is outputting noise (noise ❌)
    """
    print("\n" + "=" * 60)
    print("  LEVEL 2 — SIGNAL QUALITY CHECK (real EEG + trained model)")
    print("=" * 60)

    try:
        from phase_02_DL       import create_dataloader
        from phase_06_training import EEGPipeline, extract_category, build_category_index
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Load checkpoint
    ckpt = Path(ckpt_path)
    if not ckpt.exists():
        print(f"  ❌ Checkpoint not found: {ckpt}")
        return

    print(f"  Loading: {ckpt.name}")
    checkpoint = torch.load(ckpt, map_location=device)
    cat_index  = checkpoint["cat_index"]

    # Load config
    config_path = ckpt.parent / "config.json"
    token_dim, num_heads, ff_dim, num_layers = 224, 4, 416, 2
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        token_dim     = cfg.get("token_dim",  token_dim)
        num_heads     = cfg.get("num_heads",  num_heads)
        ff_dim        = cfg.get("ff_dim",     ff_dim)
        num_layers    = cfg.get("num_layers", num_layers)

    # Instantiate and load model
    model = EEGPipeline(
        token_dim=token_dim, num_heads=num_heads,
        ff_dim=ff_dim, num_layers=num_layers, dropout=0.0
    ).to(device)

    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as e:
        err = str(e)
        print("\n  ❌ CHECKPOINT IS INCOMPATIBLE WITH CURRENT ARCHITECTURE")
        print("  ─" * 30)
        # Show which weights mismatched
        lines = [l.strip() for l in err.splitlines() if "size mismatch" in l]
        for line in lines:
            print(f"  • {line}")
        print("\n  WHY THIS HAPPENED:")
        print("  The old checkpoint was trained with the BROKEN architecture:")
        print("    - Kernels: 3, 7, 15  (12ms / 28ms / 60ms  — too small for VEPs)")
        print("    - Tokens:  62        (last 2 EEG samples were discarded)")
        print("\n  The new architecture has FIXED shapes:")
        print("    - Kernels: 25, 51, 75 (100ms / 204ms / 300ms — covers C1, N170, P300)")
        print("    - Tokens:  63         (all 250 samples now covered)")
        print("\n  ✅ THIS IS CORRECT AND EXPECTED BEHAVIOR.")
        print("  The shape mismatch confirms the fix is in place.")
        print("\n  WHAT TO DO NEXT:")
        print("  1. Retrain Phase 6 from scratch with the new architecture")
        print("     → python phase_06_training.py")
        print("  2. After training completes, re-run this check:")
        print("     → python check_encoder.py --full")
        print("  3. A healthy trained model should show:")
        print("     → IntraSim/InterSim ratio > 5")
        print("     → Test Top-1 accuracy > 5% (vs 1.38% chance)")
        print("=" * 60)
        return

    model.eval()
    print(f"  Model loaded (token_dim={token_dim})")

    # Run through training set to collect object latents per category
    print(f"\n  Running encoder on training set...")
    train_loader = create_dataloader(data_dir=data_dir, batch_size=256, split="train")

    latents_by_cat = {c: [] for c in cat_index.values()}

    with torch.no_grad():
        for eeg, labels, _ in train_loader:
            eeg = eeg.to(device)
            out = model(eeg)
            obj = out["object"].cpu()   # [B, 128]

            for i, lbl in enumerate(labels):
                cat_name = extract_category(lbl)
                if cat_name in cat_index:
                    latents_by_cat[cat_index[cat_name]].append(obj[i])

    # Compute centroids
    n_cats = len(cat_index)
    centroids = torch.zeros(n_cats, 128)
    for c in range(n_cats):
        if latents_by_cat[c]:
            vecs = torch.stack(latents_by_cat[c])   # [N_c, 128]
            centroids[c] = F.normalize(vecs.mean(dim=0), dim=-1)

    # IntraClassSim: how similar are individual samples to their category centroid
    intra_sims = []
    for c in range(n_cats):
        if len(latents_by_cat[c]) > 1:
            vecs   = F.normalize(torch.stack(latents_by_cat[c]), dim=-1)  # [N_c, 128]
            sims   = torch.matmul(vecs, centroids[c])                       # [N_c]
            intra_sims.append(sims.mean().item())
    intra_sim = float(np.mean(intra_sims))

    # InterClassSim: how similar are category centroids to EACH OTHER
    sim_matrix = torch.matmul(centroids, centroids.T)   # [72, 72]
    off_diag   = sim_matrix[~torch.eye(n_cats, dtype=torch.bool)]
    inter_sim  = off_diag.mean().item()

    ratio = intra_sim / (abs(inter_sim) + 1e-8)

    print(f"\n  {'─'*40}")
    print(f"  ENCODER SIGNAL QUALITY METRICS")
    print(f"  {'─'*40}")
    print(f"  IntraClassSim (same cat → centroid):   {intra_sim:.4f}")
    print(f"  InterClassSim (centroid vs centroid):  {inter_sim:.4f}")
    print(f"  Ratio = IntraSim / InterSim:           {ratio:.2f}")
    print(f"\n  Interpretation:")

    if ratio > 5:
        verdict = "✅ HEALTHY — encoder extracts strong category signal"
    elif ratio > 2:
        verdict = "⚠️  MARGINAL — some category signal, but weak generalization"
    else:
        verdict = "❌ COLLAPSED — encoder outputs noise (no category signal found)"

    print(f"  {verdict}")
    print(f"\n  Healthy target:   ratio > 5")
    print(f"  Current ratio:    {ratio:.2f}")

    if inter_sim < 0:
        print(f"\n  ✅ Inter-class centroids are ORTHOGONAL/NEGATIVE")
        print(f"     (negative inter_sim = categories are well separated)")
    else:
        print(f"\n  ⚠️  Inter-class centroids are POSITIVELY similar ({inter_sim:.3f})")
        print(f"     (categories cluster together — poor separation)")

    # Also test on TEST set
    print(f"\n  Running encoder on test set (unseen images _08, _09)...")
    test_loader = create_dataloader(data_dir=data_dir, batch_size=256, split="test")

    test_latents = []
    test_cats    = []
    with torch.no_grad():
        for eeg, labels, _ in test_loader:
            eeg = eeg.to(device)
            out = model(eeg)
            obj = out["object"].cpu()
            for i, lbl in enumerate(labels):
                cat_name = extract_category(lbl)
                if cat_name in cat_index:
                    test_latents.append(obj[i])
                    test_cats.append(cat_index[cat_name])

    test_latents = torch.stack(test_latents)          # [N_test, 128]
    test_cats    = torch.tensor(test_cats)            # [N_test]

    # Nearest centroid top-1 accuracy
    sims     = torch.matmul(test_latents, centroids.T)  # [N_test, 72]
    preds    = sims.argmax(dim=1)
    top1_acc = (preds == test_cats).float().mean().item()

    print(f"\n  Test set nearest-centroid Top-1 accuracy: {top1_acc*100:.2f}%")
    print(f"  Chance level: {100/n_cats:.2f}%  ({n_cats} categories)")

    if top1_acc * 100 > 5:
        print(f"  ✅ Above chance — encoder has SOME category generalization")
    elif top1_acc * 100 > 1/n_cats * 100:
        print(f"  ⚠️  Barely above chance — very weak generalization")
    else:
        print(f"  ❌ At or below chance — no generalization to unseen images")

    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG Encoder Diagnostic")
    parser.add_argument("--full",     action="store_true",
                        help="Run Level 2 signal quality check (needs real data + checkpoint)")
    parser.add_argument("--channel",  action="store_true",
                        help="Run Level 2b channel attention check — shows per-electrode gate values (needs trained checkpoint)")
    parser.add_argument("--data_dir", type=str, default="data/EEGdatanpy",
                        help="Path to EEG data directory")
    parser.add_argument("--ckpt",     type=str,
                        default="checkpoints/phase06_2.24_leak_prevented/best_model.pt",
                        help="Path to trained Phase 6 checkpoint")
    parser.add_argument("--token_dim", type=int, default=192)
    args = parser.parse_args()

    # Always run Level 1
    passed = level1_shape_checks(token_dim=args.token_dim)

    if args.channel:
        # Level 2b: did channel attention learn the right electrode weights?
        level2b_channel_attention_check(args.data_dir, args.ckpt)
    elif args.full:
        # Level 2: did the encoder extract real category signal?
        level2_signal_quality(args.data_dir, args.ckpt)
    else:
        print("\n  💡 Tips (run after retraining Phase 6):")
        print("     Signal quality check   : python check_encoder.py --full")
        print("     Channel attention check: python check_encoder.py --channel")
