from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader, TensorDataset


# =========================================================
# DEVICE
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


def _validate_npy_header(path: Path) -> None:
    """Fail fast if the file is not a valid NumPy .npy binary."""
    with path.open("rb") as f:
        magic = f.read(6)

    if magic != b"\x93NUMPY":
        raise ValueError(
            f"{path} is not a valid .npy array file. "
            "It may be corrupted or replaced with non-array content."
        )


def _load_numpy(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing required file: {p}")
    _validate_npy_header(p)
    return np.load(p)


# =========================================================
# LOAD PRECOMPUTED DATA (🔥 CORE FIX)
# =========================================================
clip_np = _load_numpy("clip_embeddings.npy")
latent_np = _load_numpy("latent_label.npy")

eeg_ids = _load_numpy("eeg_ids.npy")
window_ids = _load_numpy("window_ids.npy")
image_names = _load_numpy("image_names.npy")

if clip_np.shape[0] != latent_np.shape[0]:
    raise ValueError(
        "Sample count mismatch: "
        f"clip_embeddings={clip_np.shape[0]} vs latent_label={latent_np.shape[0]}"
    )

if clip_np.shape[0] != len(eeg_ids) or len(eeg_ids) != len(window_ids) or len(window_ids) != len(image_names):
    raise ValueError(
        "Metadata length mismatch across clip embeddings / eeg_ids / window_ids / image_names."
    )

clip_embeddings = torch.tensor(clip_np).float()
latent_label = torch.tensor(latent_np).float()
latent_label = latent_label / latent_label.abs().max()

# =========================================================
# LOAD VAE (ONLY FOR DECODING)
# =========================================================
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
vae.eval()

# 🔥 correct scaling for VAE
# training
# latent_label = latent_label * vae.config.scaling_factor
# normalize CLIP
clip_embeddings = clip_embeddings / clip_embeddings.norm(dim=1, keepdim=True)
# clip_embeddings = torch.clamp(clip_embeddings, -1, 1)

print("CLIP:", clip_embeddings.shape)
print("LATENT:", latent_label.shape)





# =========================================================
# MODEL (CLIP → LATENT)
# =========================================================
import torch.nn as nn

class ClipToLatent(nn.Module):
    def __init__(self):
        super().__init__()

        # Step 1: MLP → small feature map
        self.fc = nn.Sequential(
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),

            nn.Linear(2048, 16384),
            nn.ReLU(),
            nn.BatchNorm1d(16384)
        )

        # Step 2: reshape → (256, 8, 8)
        self.unflatten = nn.Unflatten(1, (256, 8, 8))

        # Step 3: Upsampling (ConvTranspose)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 8 → 16
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # 16 → 32
            nn.ReLU(),

            nn.ConvTranspose2d(64, 4, 4, stride=2, padding=1) ,
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = self.unflatten(x)
        x = self.decoder(x)
        return x




model = ClipToLatent().to(device)
def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight)

model.apply(init_weights)


# =========================================================
# LOSS
# =========================================================
def loss_fn(pred, target):
    l1 = F.l1_loss(pred, target)
    cos = 1 - F.cosine_similarity(pred.flatten(1), target.flatten(1)).mean()
    var_loss = -pred.std()
    return l1 + 0.1 * cos + 0.01 * var_loss

# =========================================================
# TRAIN
# =========================================================
loader = DataLoader(
    TensorDataset(clip_embeddings, latent_label),
    batch_size=16,
    shuffle=True,
    num_workers=0,
)
print(latent_label.min().item(), latent_label.max().item())

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
print("\n[INFO] Training CLIP → LATENT mapper...\n")

for epoch in range(50):
    total = 0
    

    for c, l in loader:
        c, l = c.to(device), l.to(device)

        pred = model(c)
        loss = F.l1_loss(pred, l)
        optimizer.zero_grad()
        loss.backward()

# ✅ ADD THIS LINE
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total += loss.item()
        scheduler.step()

    print(f"Epoch {epoch+1} Loss {total:.4f}")


# =========================================================
# GENERATE IMAGES
# =========================================================
out_dir = Path("outputs/decoder_results")
out_dir.mkdir(parents=True, exist_ok=True)

print("\n[INFO] Generating images...\n")

model.eval()

batch_size = 1   # 🔥 adjust if needed

with torch.no_grad(): 
    for batch_idx in range(0, len(clip_embeddings), 1):
        clip = clip_embeddings[batch_idx:batch_idx+1].to(device) 
        latents = model(clip)
        image = vae.decode(latents).sample 
        image = (image / 2 + 0.5).clamp(0, 1) 
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0] 
        image = (image * 255).astype(np.uint8) 
        name = f"{eeg_ids[batch_idx]}_w{window_ids[batch_idx]}_{image_names[batch_idx]}.png" 
        Image.fromarray(image).save(out_dir / name) 
        if batch_idx % 100 == 0: print(f"[SAVED] {batch_idx}") 
        print("\n✅ Done!")

        
print("\n✅ Images saved in outputs/decoder_results\n")
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open("outputs/decoder_results/" + name)
plt.imshow(img)
plt.axis("off")
plt.show()