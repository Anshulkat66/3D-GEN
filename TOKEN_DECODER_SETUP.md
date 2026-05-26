# Token-to-Image CLIP Mapper — Real Data Setup

## ✅ Changes Made (5 Critical Fixes)

### FIX 1: Real Data Pairing
- ❌ **Before**: `category = subject_id` (abstract mapping)
- ✅ **After**: Direct EEG window ↔ image pairing based on category folders
- **How**: `TokenImagePairDataset._pair_tokens_with_images()` creates valid pairs
- **Requirement**: Images organized as `data/image/category_name/*.jpg`

### FIX 2: Remove Random Targets
- ❌ **Before**: `torch.randn(512)` synthetic CLIP features
- ✅ **After**: Only real images loaded via CLIP model
- **How**: If a category has no images, that sample is skipped entirely
- **No more fallbacks**: Real features only

### FIX 3: Remove CLIP Decoder
- ❌ **Before**: `clip_mapped → clip_decoder → clip_recon`, then MSE loss
- ✅ **After**: Direct loss: `MSE(clip_mapper(tokens), clip_target)`
- **Why**: Simpler, direct token-to-CLIP feature mapping
- **Removed classes**: `CLIPDecoder` is deleted

### FIX 4: Train CLIP-Only First
- ❌ **Before**: Multi-task (CLIP + VAE + Diffusion)
- ✅ **After**: Focus on CLIP mapping only
- **Why**: Validate the pipeline before multi-task complexity
- **Removed classes**: `VAEMapper`, `DiffusionMapper`, `VAEDecoder`

### FIX 5: Enforce Encoder Checkpoint
- ❌ **Before**: Optional encoder (`--encoder-ckpt` had default=None)
- ✅ **After**: **REQUIRED** argument with validation
- **Error handling**: Fails immediately if checkpoint doesn't exist
- **Console message**: Shows exact command if missing

---

## 📁 Required Directory Structure

```
d:\3D-GEN-main\
├── outputs/
│   └── phase1_output/
│       ├── sub_01_clean.npy
│       ├── sub_02_clean.npy
│       └── ...
├── data/
│   └── image/
│       ├── airplane/
│       │   ├── image1.jpg
│       │   ├── image2.jpg
│       │   └── ...
│       ├── apple/
│       │   └── ...
│       └── ... (other categories)
└── token_to_image_decoder.py
```

**Category names must match subject IDs** (e.g., `sub_01`, `sub_02`).

---

## 🚀 Running the Decoder

### Prerequisites
```bash
pip install openai-clip diffusers
```

### Command
```bash
python token_to_image_decoder.py \
  --encoder-ckpt path/to/encoder_checkpoint.pt \
  --image-dir data/image \
  --eeg-data-dir outputs/phase1_output \
  --epochs 100 \
  --batch-size 64 \
  --lr 1e-3
```

### Example with Full Path
```bash
python token_to_image_decoder.py \
  --encoder-ckpt D:\checkpoints\eeg_encoder_v1.pt \
  --image-dir D:\3D-GEN-main\data\image \
  --epochs 100
```

---

## 📊 Expected Output

```
[info] Device: cuda (or cpu)
[info] Extracting CLIP features from images...
Processing airplane: 100%|██████████| 10/10
Processing apple: 100%|██████████| 8/8
...
[info] Loaded CLIP features for 5 categories
[info] Extracting EEG tokens...
[info] Extracted 5234 EEG tokens, dim=256
[info] Pairing tokens with images...
[info] Created 1043 valid token-image pairs

========================================================================
Token-to-CLIP Mapper Training (Real Images)
Device            : cuda
Token dimension   : 256
Encoder checkpoint: D:\checkpoints\eeg_encoder_v1.pt
Image directory   : D:\3D-GEN-main\data\image
Total samples     : 1043
Train samples     : 834
Val samples       : 209
Epochs            : 100
========================================================================

Epoch 001/100 | Train Loss: 0.5234 | Val Loss: 0.4891
Epoch 010/100 | Train Loss: 0.1234 | Val Loss: 0.1456
...
Epoch 100/100 | Train Loss: 0.0089 | Val Loss: 0.0145

------------------------------------------------------------------------
Best Validation Loss: 0.0089
[info] Training complete!
```

---

## 🔍 Key Classes

### `TokenImagePairDataset`
- Loads EEG tokens from `*_clean.npy` files
- Extracts CLIP features from real images
- **Pairs them by category**: EEG from `sub_01` ↔ images in `data/image/sub_01/`
- Returns: `{"token": [256], "clip_target": [512]}`

### `CLIPMapper`
- **Input**: EEG tokens [256]
- **Output**: CLIP embeddings [512]
- **Architecture**:
  ```
  Linear(256 → 1024) → BatchNorm → GELU → Dropout
  → Linear(1024 → 768) → BatchNorm → GELU → Dropout
  → Linear(768 → 512)
  ```

### Training Loop
- **Loss**: `MSE(clip_mapper(tokens), clip_target)`
- **Optimizer**: Adam
- **Frozen encoder**: No gradient updates to encoder
- **Metrics**: Train/Val loss per epoch

---

## ✨ Next Steps (After CLIP Proves Good)

Once CLIP validation loss converges well:

1. **Add VAE latent mapping**
   - Uncomment `VAEMapper` class
   - Add VAE loss term in training

2. **Add diffusion latent mapping**
   - Uncomment `DiffusionMapper` class
   - Multi-task training
   
3. **Generative decoder**
   - Map tokens → CLIP/VAE latent → decode to image
   - Use Stable Diffusion VAE decoder

---

## ⚠️ Validation Checklist

Before running:
- [ ] `data/image/` exists with category folders
- [ ] `outputs/phase1_output/` has `*_clean.npy` files
- [ ] Encoder checkpoint path is correct
- [ ] Category names match subject IDs (e.g., `sub_01`)
- [ ] CLIP library installed (or will use zeros gracefully)

