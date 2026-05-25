import torch
import numpy as np
from pathlib import Path
from PIL import Image
from diffusers import AutoencoderKL

# ======================================================
# DEVICE
# ======================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ======================================================
# LOAD VAE
# ======================================================
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
vae.eval()

# ======================================================
# LOAD IMAGE LIST (SAME ORDER AS MAPPING FILE)
# ======================================================
image_dir = Path("image")

all_images = []
for cat in sorted(image_dir.iterdir()):
    if cat.is_dir():
        all_images += sorted(cat.glob("*.png"))

print("Total images:", len(all_images))

# ======================================================
# ENCODE EACH IMAGE ONCE
# ======================================================
image_latents = []

for i, img_path in enumerate(all_images):
    print(f"[ENCODING IMAGE] {i}/{len(all_images)}")

    img = Image.open(img_path).convert("RGB").resize((512, 512))
    import torchvision.transforms as T

    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),                    # [0,1]
        T.Normalize([0.5, 0.5, 0.5],    # IMPORTANT: 3 channels
                    [0.5, 0.5, 0.5])    # → [-1,1]
    ])

    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        latent = vae.encode(x).latent_dist.sample()

        print("BEFORE SCALE:", latent.min().item(), latent.max().item())

        latent = latent * vae.config.scaling_factor

        print("AFTER SCALE:", latent.min().item(), latent.max().item())

    image_latents.append(latent.squeeze(0).cpu())

    # with torch.no_grad():
    #     latent = vae.encode(x).latent_dist.sample()
    #     latent = latent * vae.config.scaling_factor

    # print("DEBUG LATENT:", latent.min().item(), latent.max().item())

image_latents = torch.stack(image_latents)



# ======================================================
# LOAD WINDOW IDS (THIS IS THE REAL KEY)
# ======================================================
window_ids = np.load("window_ids.npy")

# ======================================================
# MAP WINDOW → IMAGE DIRECTLY
# ======================================================
expanded_latents = image_latents[window_ids]

# ======================================================
# SAVE FINAL LATENTS
# ======================================================
np.save("latent_label.npy", expanded_latents.numpy())
print(expanded_latents.min().item(), expanded_latents.max().item())

print("\n✅ FINAL latent_label.npy saved")
print("Shape:", expanded_latents.shape)

print("image_latents:", image_latents.shape)
print("window_ids:", window_ids.shape)
