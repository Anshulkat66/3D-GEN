"""Phase 2: EEG DataLoader.

Loads pre-epoched, pre-normalized EEG trials from all subjects via phase_01pp
and wraps them in a PyTorch Dataset / DataLoader for model training.

Each sample returned is:
    eeg_tensor  : [250, 64]  float32  (Time × Channels)
    label       : str        stimulus image name  (e.g. 'airplane_03')
    subject_id  : str        subject folder name  (e.g. 'sub01')
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from phase_01pp import load_all_subjects


class EEGDataset(Dataset):
    """PyTorch Dataset over all pre-epoched EEG trials across all subjects.

    Args:
        data_dir : path to data/EEGdatanpy (passed to load_all_subjects)
        split    : 'all'   → every trial (default, use for tokenization / Phase 2-3)
                   'train' → labels ending _00 to _07
                   'test'  → labels ending _08 or _09
    """

    def __init__(self, data_dir: str = "data/EEGdatanpy", split: str = "all"):
        self.split = split

        # Load all subjects once into RAM as a dict of {sub_id: {eeg_data, labels, ...}}
        subjects_data = load_all_subjects(data_dir)

        # Build a flat index: list of (eeg_trial [64,250], label_str, subject_id)
        self.samples = []

        for sub_id, record in subjects_data.items():
            eeg_data = record["eeg_data"].astype(np.float32)  # [1728, 64, 250]  float64 → float32
            labels   = record["labels"]                        # [1728]  strings

            for i in range(len(labels)):
                label = str(labels[i])

                # Apply split filter
                if split == "train" and (label.endswith("_08") or label.endswith("_09")):
                    continue
                if split == "test" and not (label.endswith("_08") or label.endswith("_09")):
                    continue

                self.samples.append((eeg_data[i], label, sub_id))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        trial, label, sub_id = self.samples[idx]

        # trial shape: [64, 250] → transpose to [250, 64] (Time × Channels) for the encoder
        eeg_tensor = torch.tensor(trial.T, dtype=torch.float32)

        return eeg_tensor, label, sub_id



def create_dataloader(
    data_dir: str = "data/EEGdatanpy",
    batch_size: int = 64,
    split: str = "all",
) -> DataLoader:
    """Create a DataLoader over the EEGDataset."""

    dataset = EEGDataset(data_dir=data_dir, split=split)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    return loader


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  PHASE 2 — DATALOADER VERIFICATION")
    print("=" * 60)

    # ── CHECK 1: Total trial count ────────────────────────────────
    ds_all = EEGDataset(split="all")
    expected_total = 12 * 1728  # 20,736
    status = "✅" if len(ds_all) == expected_total else "❌"
    print(f"\n[1] Total trials : {len(ds_all)} (expected {expected_total}) {status}")

    # ── CHECK 2: Train / Test split counts ───────────────────────
    ds_train = EEGDataset(split="train")
    ds_test  = EEGDataset(split="test")
    expected_train = 12 * 1152   # 13,824
    expected_test  = 12 * 576    # 6,912
    s_tr = "✅" if len(ds_train) == expected_train else "❌"
    s_te = "✅" if len(ds_test)  == expected_test  else "❌"
    print(f"[2] Train trials : {len(ds_train)} (expected {expected_train}) {s_tr}")
    print(f"    Test  trials : {len(ds_test)}  (expected {expected_test})  {s_te}")

    # ── CHECK 3: Batch shape and dtype ───────────────────────────
    loader = create_dataloader(split="all")
    eeg, labels, subjects = next(iter(loader))
    shape_ok = eeg.shape == torch.Size([64, 250, 64])
    dtype_ok = eeg.dtype == torch.float32
    print(f"[3] Batch shape  : {tuple(eeg.shape)} {'✅' if shape_ok else '❌'} (expected [64, 250, 64])")
    print(f"    Dtype        : {eeg.dtype}   {'✅' if dtype_ok else '❌'} (expected float32)")

    # ── CHECK 4: No NaN / Inf in a batch ─────────────────────────
    no_nan = not eeg.isnan().any().item()
    no_inf = not eeg.isinf().any().item()
    print(f"[4] NaN in batch : {'None ✅' if no_nan else 'FOUND ❌'}")
    print(f"    Inf in batch : {'None ✅' if no_inf else 'FOUND ❌'}")

    # ── CHECK 5: Shuffle — batch has multiple subjects mixed ──────
    unique_subjects = len(set(subjects))
    shuffle_ok = unique_subjects > 1
    print(f"[5] Subjects in batch: {unique_subjects} unique {'✅ (shuffled)' if shuffle_ok else '❌ (not shuffled)'}")
    print(f"    Sample label  : {labels[0]}")
    print(f"    Sample subject: {subjects[0]}")

    print("\n" + "=" * 60)
    print("  VERIFICATION COMPLETE")
    print("=" * 60 + "\n")