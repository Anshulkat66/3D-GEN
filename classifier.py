from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

from phase_02_mse_tok import EEGEncoder


class TokenProbeDataset(Dataset):
	"""Create pooled token features with a frozen pretrained encoder.

	Pipeline:
	EEG window [T, C] -> encoder -> tokens [T_tokens, D] -> mean over T_tokens -> [D]
	"""

	def __init__(
		self,
		data_dir: str | Path,
		encoder: nn.Module,
		encoder_device: torch.device,
		window_size: int = 1024,
		stride: int = 512,
		inference_batch_size: int = 128,
	):
		self.data_dir = Path(data_dir)
		self.window_size = int(window_size)
		self.stride = int(stride)
		self.encoder = encoder
		self.encoder_device = encoder_device
		self.inference_batch_size = int(inference_batch_size)

		self.files = sorted(self.data_dir.glob("*_clean.npy"))
		if not self.files:
			raise FileNotFoundError(f"No cleaned EEG files found in: {self.data_dir}")

		self.subject_ids = [file_path.stem.replace("_clean", "") for file_path in self.files]
		self.subject_to_label = {subject_id: idx for idx, subject_id in enumerate(self.subject_ids)}

		self.pooled_features: torch.Tensor
		self.labels: torch.Tensor
		self.feature_dim: int

		self._build_feature_cache()

	def _iter_windows(self, eeg: np.ndarray) -> np.ndarray:
		length = int(eeg.shape[0])
		if length < self.window_size:
			return np.empty((0, self.window_size, eeg.shape[1]), dtype=np.float32)

		windows = [eeg[start : start + self.window_size] for start in range(0, length - self.window_size + 1, self.stride)]
		return np.asarray(windows, dtype=np.float32)

	def _encode_windows(self, windows_np: np.ndarray) -> torch.Tensor:
		if windows_np.shape[0] == 0:
			return torch.empty(0, 0, dtype=torch.float32)

		all_features: list[torch.Tensor] = []
		with torch.no_grad():
			for start in range(0, windows_np.shape[0], self.inference_batch_size):
				end = min(start + self.inference_batch_size, windows_np.shape[0])
				batch = torch.from_numpy(windows_np[start:end]).to(self.encoder_device)
				tokens = self.encoder(batch)  # [B, T_tokens, D]
				pooled = tokens.mean(dim=1)  # [B, D]
				all_features.append(pooled.cpu())

		return torch.cat(all_features, dim=0)

	def _build_feature_cache(self) -> None:
		feature_chunks: list[torch.Tensor] = []
		label_chunks: list[torch.Tensor] = []

		self.encoder.eval()

		for file_path in self.files:
			subject_id = file_path.stem.replace("_clean", "")
			label = self.subject_to_label[subject_id]
			eeg = np.load(file_path).astype(np.float32)

			windows_np = self._iter_windows(eeg)
			if windows_np.shape[0] == 0:
				continue

			pooled = self._encode_windows(windows_np)
			feature_chunks.append(pooled)
			label_chunks.append(torch.full((pooled.shape[0],), label, dtype=torch.long))

		if not feature_chunks:
			raise RuntimeError("No windows were created. Check window-size/stride versus signal lengths.")

		self.pooled_features = torch.cat(feature_chunks, dim=0).float()
		self.labels = torch.cat(label_chunks, dim=0).long()
		self.feature_dim = int(self.pooled_features.shape[1])

	def get_window_labels(self) -> np.ndarray:
		return self.labels.numpy()

	def __len__(self) -> int:
		return int(self.labels.shape[0])

	def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
		return self.pooled_features[idx], self.labels[idx]


class DeepMLPClassifier(nn.Module):
	def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.3):
		super().__init__()

		self.net = nn.Sequential(
			nn.Linear(input_dim, 512),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Dropout(dropout),
			nn.Linear(512, 256),
			nn.BatchNorm1d(256),
			nn.ReLU(inplace=True),
			nn.Dropout(dropout),
			nn.Linear(256, 128),
			nn.ReLU(inplace=True),
			nn.Dropout(dropout),
			nn.Linear(128, num_classes),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)


def load_frozen_encoder(
	checkpoint_path: str | Path | None,
	device: torch.device,
) -> EEGEncoder:
	encoder = EEGEncoder().to(device)

	if checkpoint_path:
		checkpoint_path = Path(checkpoint_path)
		if not checkpoint_path.exists():
			raise FileNotFoundError(f"Encoder checkpoint not found: {checkpoint_path}")

		state = torch.load(checkpoint_path, map_location=device)
		if isinstance(state, dict) and "state_dict" in state:
			state = state["state_dict"]

		clean_state = {}
		for key, value in state.items():
			new_key = key.replace("module.", "")
			clean_state[new_key] = value

		missing, unexpected = encoder.load_state_dict(clean_state, strict=False)
		if missing:
			print(f"[warn] Missing encoder keys: {len(missing)}")
		if unexpected:
			print(f"[warn] Unexpected encoder keys: {len(unexpected)}")
	else:
		print("[warn] No --encoder-ckpt provided. Using randomly initialized encoder weights.")

	for param in encoder.parameters():
		param.requires_grad = False

	encoder.eval()
	return encoder


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
	model.eval()
	criterion = nn.CrossEntropyLoss()

	total_loss = 0.0
	total_correct = 0
	total_samples = 0

	with torch.no_grad():
		for features, labels in loader:
			features = features.to(device)
			labels = labels.to(device)

			logits = model(features)
			loss = criterion(logits, labels)

			preds = logits.argmax(dim=1)
			total_correct += int((preds == labels).sum().item())
			batch_size = int(labels.size(0))
			total_samples += batch_size
			total_loss += float(loss.item() * batch_size)

	avg_loss = total_loss / max(total_samples, 1)
	accuracy = total_correct / max(total_samples, 1)
	return avg_loss, accuracy


def train_mlp_classifier(
	data_dir: str | Path,
	encoder_ckpt: str | Path | None = None,
	window_size: int = 1024,
	stride: int = 512,
	batch_size: int = 64,
	inference_batch_size: int = 128,
	epochs: int = 25,
	learning_rate: float = 1e-3,
	val_split: float = 0.2,
	random_state: int = 42,
) -> None:
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	encoder = load_frozen_encoder(encoder_ckpt, device=device)

	dataset = TokenProbeDataset(
		data_dir=data_dir,
		encoder=encoder,
		encoder_device=device,
		window_size=window_size,
		stride=stride,
		inference_batch_size=inference_batch_size,
	)

	all_indices = np.arange(len(dataset))
	all_labels = dataset.get_window_labels()

	train_idx, val_idx = train_test_split(
		all_indices,
		test_size=val_split,
		random_state=random_state,
		stratify=all_labels,
	)

	train_set = Subset(dataset, train_idx.tolist())
	val_set = Subset(dataset, val_idx.tolist())

	train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
	val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

	model = DeepMLPClassifier(input_dim=dataset.feature_dim, num_classes=len(dataset.subject_ids)).to(device)

	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

	print("=" * 72)
	print("MLP Token Probe Classifier")
	print(f"Device            : {device}")
	print(f"Data dir          : {Path(data_dir)}")
	print(f"Encoder checkpoint: {encoder_ckpt if encoder_ckpt else 'None'}")
	print(f"Subjects/classes  : {len(dataset.subject_ids)}")
	print(f"Total windows     : {len(dataset)}")
	print(f"Train windows     : {len(train_set)}")
	print(f"Validation windows: {len(val_set)}")
	print(f"Token dim (D)     : {dataset.feature_dim}")
	print("=" * 72)

	best_val_acc = 0.0

	for epoch in range(1, epochs + 1):
		model.train()
		running_loss = 0.0
		running_correct = 0
		running_samples = 0

		for features, labels in train_loader:
			features = features.to(device)
			labels = labels.to(device)

			optimizer.zero_grad()
			logits = model(features)
			loss = criterion(logits, labels)
			loss.backward()
			optimizer.step()

			preds = logits.argmax(dim=1)
			batch_size_now = int(labels.size(0))
			running_samples += batch_size_now
			running_correct += int((preds == labels).sum().item())
			running_loss += float(loss.item() * batch_size_now)

		train_loss = running_loss / max(running_samples, 1)
		train_acc = running_correct / max(running_samples, 1)

		val_loss, val_acc = evaluate(model, val_loader, device)
		best_val_acc = max(best_val_acc, val_acc)

		print(
			f"Epoch {epoch:02d}/{epochs} | "
			f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
			f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
		)

	print("-" * 72)
	print(f"Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc * 100:.2f}%)")
	print("Class mapping (label -> subject):")
	for subject_id, label in dataset.subject_to_label.items():
		print(f"  {label:02d} -> {subject_id}")



def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train a deep MLP classifier on pooled encoder tokens.")
	parser.add_argument("--data-dir", type=str, default="outputs/phase1_output", help="Directory with *_clean.npy files")
	parser.add_argument("--encoder-ckpt", type=str, default=None, help="Path to pretrained EEG encoder checkpoint")
	parser.add_argument("--window-size", type=int, default=1024, help="Window length in time samples")
	parser.add_argument("--stride", type=int, default=512, help="Window stride")
	parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
	parser.add_argument("--inference-batch-size", type=int, default=128, help="Batch size for encoder feature extraction")
	parser.add_argument("--epochs", type=int, default=1024, help="Number of training epochs")
	parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
	parser.add_argument("--val-split", type=float, default=0.2, help="Validation split fraction")
	parser.add_argument("--seed", type=int, default=42, help="Random seed")
	return parser.parse_args(argv)


def main() -> None:
	args = parse_args()

	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	train_mlp_classifier(
		data_dir=args.data_dir,
		encoder_ckpt=args.encoder_ckpt,
		window_size=args.window_size,
		stride=args.stride,
		batch_size=args.batch_size,
		inference_batch_size=args.inference_batch_size,
		epochs=args.epochs,
		learning_rate=args.lr,
		val_split=args.val_split,
		random_state=args.seed,
	)


if __name__ == "__main__":
	main()
