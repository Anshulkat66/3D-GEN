import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class EEGDataset(Dataset):
    def __init__(self, data_dir, window_size=1024, stride=512):
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.stride = stride

        self.files = sorted(self.data_dir.glob("*.npy"))

        self.samples = []  # stores (file_path, start_idx)

        self._prepare_index()

    def _prepare_index(self):
        for file_path in self.files:
            eeg = np.load(file_path, mmap_mode='r')  # memory efficient
            length = eeg.shape[0]

            for start in range(0, length - self.window_size, self.stride):
                self.samples.append((file_path, start))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, start = self.samples[idx]

        eeg = np.load(file_path)

        window = eeg[start:start + self.window_size]

        # convert to tensor
        window = torch.tensor(window, dtype=torch.float32)

        return window


def create_dataloader(data_dir, batch_size=8, window_size=1024, stride=512):
    dataset = EEGDataset(data_dir, window_size, stride)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    return loader


if __name__ == "__main__":
    data_dir = "outputs/phase1_output"

    loader = create_dataloader(data_dir)

    for batch in loader:
        print("Batch shape:", batch.shape)
        break