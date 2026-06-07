"""Connect Phase 2 DataLoader → Phase 3 EEGEncoder → Phase 4 Attention.

Pulls one batch of real EEG trials, passes them through the full
pipeline up to Phase 4, and runs checks at each stage to confirm
data is transforming correctly at every step.

Flow:
    DataLoader [B, 250, 64]
        ↓ Phase 3: EEGEncoder
    Tokens [B, 62, 256]
        ↓ Phase 4: EEGTransformerEncoder
    Shared Embedding [B, 62, 256]
"""

import torch
from phase_02_mse_tok import EEGEncoder
from phase_04_attention import EEGTransformerEncoder
from phase_02_DL import create_dataloader

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Models ────────────────────────────────────────────────────────────────────
encoder   = EEGEncoder().to(device).eval()            # Phase 3
attention = EEGTransformerEncoder().to(device).eval() # Phase 4

# ── DataLoader ────────────────────────────────────────────────────────────────
loader = create_dataloader(data_dir="data/EEGdatanpy", batch_size=64, split="all")

# ── Real batch ────────────────────────────────────────────────────────────────
eeg, labels, subjects = next(iter(loader))
eeg = eeg.to(device)

with torch.no_grad():
    tokens    = encoder(eeg)       # Phase 3 → [64, 62, 256]
    embedding = attention(tokens)  # Phase 4 → [64, 62, 256]

print("\n" + "=" * 60)
print("  PIPELINE CHECK: DataLoader → Phase 3 → Phase 4")
print("=" * 60)

# ── CHECK 1: Dimension check at every transformation step ─────────────────────
# Verifies each axis (Batch / Time / Feature) at every stage

B_eeg,   T_eeg,   C_eeg   = eeg.shape        # [64, 250, 64]
B_tok,   T_tok,   D_tok   = tokens.shape      # [64,  62, 256]
B_emb,   T_emb,   D_emb   = embedding.shape   # [64,  62, 256]

# --- Batch dimension: must stay 64 through the whole pipeline
batch_ok = (B_eeg == B_tok == B_emb == 64)

# --- Time/Tokens: 250 raw time steps → 62 tokens (÷4 stride compression)
time_compressed = (T_eeg == 250 and T_tok == 62)
time_preserved  = (T_tok == T_emb)           # Phase 4 must NOT change token count

# --- Feature/Channel dimension
channels_ok  = (C_eeg == 64)                  # 64 EEG channels in
features_ok  = (D_tok == 256)                 # 256-dim tokens out of Phase 3
embedding_ok = (D_emb == 256)                 # 256-dim preserved through Phase 4

print(f"\n[1] DIMENSION CHECK:")
print(f"    ── Batch dimension ──────────────────────────────────")
print(f"    EEG batch    : {B_eeg}   Tokens batch : {B_tok}   Emb batch : {B_emb}")
print(f"    Batch consistent throughout pipeline: {'✅' if batch_ok else '❌'}")

print(f"\n    ── Time / Token dimension ───────────────────────────")
print(f"    EEG time steps : {T_eeg}  (250Hz × 1 second)")
print(f"    Token count    : {T_tok}  (250 ÷ 4 stride = 62 tokens)  {'✅' if time_compressed else '❌'}")
print(f"    Token count after attention: {T_emb}  (must stay 62)  {'✅' if time_preserved else '❌'}")

print(f"\n    ── Feature / Channel dimension ──────────────────────")
print(f"    EEG channels   : {C_eeg}  (64 electrodes)          {'✅' if channels_ok  else '❌'}")
print(f"    Token dim      : {D_tok}  (enriched by encoder)   {'✅' if features_ok  else '❌'}")
print(f"    Embedding dim  : {D_emb}  (preserved by attention) {'✅' if embedding_ok else '❌'}")

# --- Attention-specific dimension rules
# Phase 4 MUST preserve ALL 3 dimensions — it is not allowed to change any of them
# [B, 62, 256] in → [B, 62, 256] out — only the values change, not the shape
attn_batch_ok = (B_tok == B_emb)
attn_time_ok  = (T_tok == T_emb)      # 62 tokens in → must be 62 tokens out
attn_dim_ok   = (D_tok == D_emb)      # 256 dim in → must be 256 dim out

# Positional embedding compatibility: pos_emb shape must match token sequence
pos_emb_shape = tuple(attention.pos_embedding.shape)  # should be [1, 62, 256]
pos_compat_ok = (pos_emb_shape == (1, T_tok, D_tok))

print(f"\n    ── Attention Dimension Check (Phase 4) ──────────────")
print(f"    Input  to attention  : [{B_tok}, {T_tok}, {D_tok}]")
print(f"    Output of attention  : [{B_emb}, {T_emb}, {D_emb}]")
print(f"    Batch preserved      : {B_tok} → {B_emb}  {'✅' if attn_batch_ok else '❌'}")
print(f"    Token count preserved: {T_tok} → {T_emb}  {'✅' if attn_time_ok  else '❌'} (attention MUST NOT change this)")
print(f"    Embed dim preserved  : {D_tok} → {D_emb}  {'✅' if attn_dim_ok   else '❌'} (attention MUST NOT change this)")
print(f"    Positional emb shape : {pos_emb_shape}  {'✅' if pos_compat_ok else '❌'} (must match [1, {T_tok}, {D_tok}])")

eeg_ok   = batch_ok and channels_ok and T_eeg == 250
token_ok = batch_ok and time_compressed and features_ok
embed_ok = batch_ok and attn_batch_ok and attn_time_ok and attn_dim_ok and pos_compat_ok



# ── CHECK 2: Phase 3 token sanity (upstream health — catches regressions) ──────
tok_nan = not tokens.isnan().any().item()
tok_inf = not tokens.isinf().any().item()
tok_mean_ok = abs(tokens.mean().item()) < 0.5
tok_std_ok  = 0.01 < tokens.std().item() < 2.0

print(f"\n[2] Phase 3 Token Sanity (upstream check):")
print(f"    NaN in tokens : {'None ✅' if tok_nan else 'FOUND ❌'}")
print(f"    Inf in tokens : {'None ✅' if tok_inf else 'FOUND ❌'}")
print(f"    Token Min     : {tokens.min().item():.4f}")
print(f"    Token Max     : {tokens.max().item():.4f}")
print(f"    Token Mean    : {tokens.mean().item():.4f}   {'✅' if tok_mean_ok else '❌'} (ideal near 0)")
print(f"    Token Std     : {tokens.std().item():.4f}   {'✅' if tok_std_ok  else '❌'} (ideal 0.01–2.0)")
print(f"    Token variance across time: {tokens.var(dim=1).mean().item():.6f}")

# ── CHECK 3: No NaN / Inf at Phase 4 output ───────────────────────────────────
nan_ok = not embedding.isnan().any().item()
inf_ok = not embedding.isinf().any().item()
print(f"\n[3] NaN in embedding: {'None ✅' if nan_ok else 'FOUND ❌'}")
print(f"    Inf in embedding: {'None ✅' if inf_ok else 'FOUND ❌'}")


# ── CHECK 4: Attention actually transformed the tokens ────────────────────────
mean_diff   = (embedding - tokens).abs().mean().item()
transformed = mean_diff > 1e-4
print(f"\n[4] Mean abs diff (tokens vs embedding): {mean_diff:.6f}")
print(f"    Attention transformed data: {'✅ YES' if transformed else '❌ NO — output same as input'}")

# ── CHECK 5: Embedding stats (health check) ───────────────────────────────────
emb_mean = embedding.mean().item()
emb_std  = embedding.std().item()
mean_ok  = abs(emb_mean) < 0.5
std_ok   = 0.01 < emb_std < 2.0
print(f"\n[5] Embedding mean : {emb_mean:.4f}   {'✅' if mean_ok else '❌'} (ideal near 0)")
print(f"    Embedding std  : {emb_std:.4f}   {'✅' if std_ok  else '❌'} (ideal 0.01–2.0)")

# ── CHECK 6: Attention redistributed information ──────────────────────────────
var_tokens    = tokens.var(dim=1).mean().item()
var_embedding = embedding.var(dim=1).mean().item()
redistributed = abs(var_embedding - var_tokens) > 1e-5
print(f"\n[6] Token variance (Phase 3)    : {var_tokens:.6f}")
print(f"    Embedding variance (Phase 4) : {var_embedding:.6f}")
print(f"    Attention redistributed info : {'✅' if redistributed else '❌'}")

# ── Sample info ───────────────────────────────────────────────────────────────
print(f"\n    Sample label  : {labels[0]}")
print(f"    Sample subject: {subjects[0]}")

# ── Final verdict ─────────────────────────────────────────────────────────────
all_ok = all([eeg_ok, token_ok, embed_ok, tok_nan, tok_inf, tok_mean_ok, tok_std_ok, nan_ok, inf_ok, transformed, mean_ok, std_ok, redistributed])
print(f"\n{'✅ Pipeline verified — Phase 3 → Phase 4 working on ' + str(device) + '!' if all_ok else '❌ Issues found — review above'}")
print("=" * 60 + "\n")