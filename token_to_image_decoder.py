from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from tkinter import image_names
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from phase_02_mse_tok import MultiScaleEEGEncoder as EEGEncoder


# =========================================================
# CLIP EXTRACTOR
# =========================================================
class ImageFeatureExtractor:
    def __init__(self, device, clip_model_name="ViT-B/32"):
        self.device = device

        clip_module = importlib.import_module("clip")
        self.model, self.preprocess = clip_module.load(clip_model_name, device=device)
        self.model.eval()

        self.cache = {}

    @torch.no_grad()
    def extract(self, image_path):
        if image_path in self.cache:
            return self.cache[image_path]

        image = Image.open(image_path).convert("RGB")
        x = self.preprocess(image).unsqueeze(0).to(self.device)
        feat = self.model.encode_image(x)
        feat = feat / feat.norm(dim=-1, keepdim=True)

        out = feat.squeeze(0).cpu()
        self.cache[image_path] = out
        return out


# =========================================================
# DATASET
# =========================================================
class TokenToClipDataset(Dataset):
    def __init__(self, eeg_dir, image_dir, encoder, device, extractor):
        self.eeg_files = sorted(Path(eeg_dir).glob("*_clean.npy"))
        self.image_dir = Path(image_dir)

        self.encoder = encoder.to(device)
        self.device = device
        self.extractor = extractor

        self.tokens = []
        self.targets = []

        self._build_mapping()
        self._build_pairs()

    def _build_mapping(self):
        all_images = []

        for cat in sorted(self.image_dir.iterdir()):
            if cat.is_dir():
                all_images += sorted(cat.glob("*.png"))

        self.map = {}
        for i, f in enumerate(self.eeg_files):
            sid = f.stem.replace("_clean", "")
            self.map[sid] = all_images[i]

    def _windows(self, eeg, window_size=600, stride=600):
        return [
            eeg[i:i+window_size]
            for i in range(0, len(eeg) - window_size + 1, stride)
        ]

    def _build_pairs(self):
        self.encoder.eval()

        for file in self.eeg_files:
            sid = file.stem.replace("_clean", "")
            eeg = np.load(file).astype(np.float32)

            windows = self._windows(eeg)
            if len(windows) == 0:
                continue

            img_path = self.map[sid]
            clip_target = self.extractor.extract(img_path)

            for w in windows:
                w = torch.tensor(w.T).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    tokens = self.encoder(w)
                    pooled = tokens.mean(dim=2)
                    token = pooled.squeeze(0).cpu()

                self.tokens.append(token)
                self.targets.append(clip_target)

        self.tokens = torch.stack(self.tokens)
        self.targets = torch.stack(self.targets)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx], self.targets[idx]


# =========================================================
# MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eeg_data_dir", default="outputs/phase1_output")
    parser.add_argument("--image_dir", default="image")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    encoder = EEGEncoder().to(device)
    extractor = ImageFeatureExtractor(device)

    dataset = TokenToClipDataset(
        args.eeg_data_dir,
        args.image_dir,
        encoder,
        device,
        extractor,
    )

    print("\n[INFO] Saving mapping outputs...\n")

    # =========================================================
    # SAVE CLIP EMBEDDINGS
    # =========================================================
    np.save("clip_embeddings.npy", dataset.targets.numpy())

    # =========================================================
    # BUILD METADATA
    # =========================================================
    eeg_ids = []
    window_ids = []
    class_names = []
    image_names = []
    image_dir = Path("image")

    all_images = []
    for cat in sorted(image_dir.iterdir()):
        if cat.is_dir():
            all_images += sorted(cat.glob("*.png")) 

    for file in dataset.eeg_files:
        sid = file.stem.replace("_clean", "")
        eeg = np.load(file)

        windows = dataset._windows(eeg)

        for w_idx in range(len(windows)):

            img = all_images[w_idx]   # 🔥 CORRECT LINE

            cls = img.parent.name
            img_name = img.name

            eeg_ids.append(sid)
            window_ids.append(w_idx)
            class_names.append(cls)
            image_names.append(img_name)


    np.save("eeg_ids.npy", np.array(eeg_ids))
    np.save("window_ids.npy", np.array(window_ids))
    np.save("class_names.npy", np.array(class_names))
    np.save("image_names.npy", np.array(image_names))
    print("✅ CLIP + metadata saved")


if __name__ == "__main__":
    main()