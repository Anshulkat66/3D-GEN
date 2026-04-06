"""Phase 1: EEG data preparation.

This module focuses only on the first stage of the pipeline:
- load EEG CSV files
- attach subject IDs and optional metadata
- apply light denoising
- apply subject-wise normalization
- optionally apply band-pass filtering

The project data currently stores one CSV per subject in `eeg_of_subjects/`.
Each file uses columns like `ch_0`, `ch_1`, ... and rows correspond to time samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd
from pandas.errors import ParserError

try:
	from scipy.signal import butter, filtfilt
except Exception:  # pragma: no cover - keeps the module importable if SciPy is unavailable
	butter = None
	filtfilt = None


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_EEG_DIR = PROJECT_ROOT / "data" / "EEGdata"


@dataclass(frozen=True)
class Phase1Config:
	"""Configuration for EEG phase-1 preprocessing."""

	eeg_dir: Path = DEFAULT_EEG_DIR
	sampling_rate: int = 250
	use_bandpass: bool = False
	low_cut_hz: float = 0.5
	high_cut_hz: float = 45.0
	denoise_window: int = 3
	epsilon: float = 1e-8


def infer_subject_id(file_path: Path) -> str:
	"""Infer subject ID from a file name like `eeg_Sub_01_250Hz.csv`."""

	match = re.search(r"Sub_(\d+)", file_path.stem, flags=re.IGNORECASE)
	if match:
		return f"sub_{match.group(1)}"
	return file_path.stem.lower()


def load_eeg_csv(file_path: Path) -> pd.DataFrame:
	"""Load a single EEG CSV file into a numeric dataframe."""

	try:
		frame = pd.read_csv(file_path)
	except ParserError:
		# Fallback for occasional malformed lines or C-engine tokenization issues.
		frame = pd.read_csv(file_path, engine="python", on_bad_lines="skip")
	frame = frame.apply(pd.to_numeric, errors="coerce")
	frame = frame.dropna(axis=0, how="any")
	return frame


def light_denoise(signal_array: np.ndarray, window: int = 3) -> np.ndarray:
	"""Apply a small moving-average smoothing filter along the time axis."""

	if window <= 1:
		return signal_array.copy()

	kernel = np.ones(window, dtype=np.float64) / float(window)
	denoised = np.empty_like(signal_array, dtype=np.float64)
	for channel_idx in range(signal_array.shape[1]):
		denoised[:, channel_idx] = np.convolve(signal_array[:, channel_idx], kernel, mode="same")
	return denoised


def subject_wise_normalize(signal_array: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
	"""Z-score normalize a subject's EEG across time for each channel."""

	mean = signal_array.mean(axis=0, keepdims=True)
	std = signal_array.std(axis=0, keepdims=True)
	normalized = (signal_array - mean) / (std + epsilon)
	normalized = np.clip(normalized,-5,5)
	return normalized

def bandpass_filter(signal_array: np.ndarray, sampling_rate: int, low_cut_hz: float, high_cut_hz: float) -> np.ndarray:
	"""Apply a Butterworth band-pass filter if SciPy is available."""

	if butter is None or filtfilt is None:
		raise RuntimeError("SciPy is required for band-pass filtering.")

	nyquist = 0.5 * sampling_rate
	low = low_cut_hz / nyquist
	high = high_cut_hz / nyquist
	b, a = butter(N=4, Wn=[low, high], btype="band")
	return filtfilt(b, a, signal_array, axis=0)


def preprocess_subject_signal(frame: pd.DataFrame, config: Phase1Config) -> np.ndarray:
	"""Run the phase-1 preprocessing pipeline on a single subject."""

	signal_array = frame.to_numpy(dtype=np.float64, copy=True)
	signal_array = light_denoise(signal_array, window=config.denoise_window)

	if config.use_bandpass:
		signal_array = bandpass_filter(
			signal_array,
			sampling_rate=config.sampling_rate,
			low_cut_hz=config.low_cut_hz,
			high_cut_hz=config.high_cut_hz,
		)

	signal_array = subject_wise_normalize(signal_array, epsilon=config.epsilon)
	return signal_array.astype(np.float32, copy=False)


def collect_eeg_files(eeg_dir: Path) -> list[Path]:
	"""Collect EEG CSV files from the subject directory."""

	if not eeg_dir.exists():
		raise FileNotFoundError(f"EEG directory not found: {eeg_dir}")
	return sorted(path for path in eeg_dir.glob("*.csv") if path.is_file())


def build_phase1_dataset(config: Phase1Config) -> list[dict[str, Any]]:
	"""Build the phase-1 dataset as a list of subject records.

	Each record contains:
	- subject_id
	- source_file
	- eeg_clean: cleaned EEG array with shape [time, channels]
	"""

	dataset: list[dict[str, Any]] = []
	for file_path in collect_eeg_files(config.eeg_dir):
		raw_frame = load_eeg_csv(file_path)
		clean_signal = preprocess_subject_signal(raw_frame, config)
		dataset.append(
			{
				"subject_id": infer_subject_id(file_path),
				"source_file": str(file_path),
				"eeg_clean": clean_signal,
				"n_samples": int(clean_signal.shape[0]),
				"n_channels": int(clean_signal.shape[1]),
			}
		)
	return dataset


def save_phase1_outputs(dataset: Iterable[dict[str, Any]], output_dir: Path) -> None:
	"""Save cleaned subject arrays and a compact manifest for later phases."""

	output_dir.mkdir(parents=True, exist_ok=True)
	manifest: list[dict[str, Any]] = []

	for record in dataset:
		subject_id = str(record["subject_id"])
		clean_signal = np.asarray(record["eeg_clean"], dtype=np.float32)
		np.save(output_dir / f"{subject_id}_clean.npy", clean_signal)

		manifest.append(
			{
				"subject_id": subject_id,
				"source_file": record["source_file"],
				"clean_file": f"{subject_id}_clean.npy",
				"n_samples": record["n_samples"],
				"n_channels": record["n_channels"],
			}
		)

	pd.DataFrame(manifest).to_csv(output_dir / "phase1_manifest.csv", index=False)


def phase1_prepare_data(config: Phase1Config | None = None, save_dir: Path | None = None) -> list[dict[str, Any]]:
	"""Run the complete phase-1 data preparation pipeline."""

	config = config or Phase1Config()
	dataset = build_phase1_dataset(config)

	if save_dir is not None:
		save_phase1_outputs(dataset, save_dir)

	return dataset


def _demo() -> None:
	"""Quick local smoke test for phase 1."""

	config = Phase1Config(use_bandpass=False)
	dataset = phase1_prepare_data(config=config, save_dir=PROJECT_ROOT / "outputs" / "phase1_output")
	print(f"Prepared {len(dataset)} subject files.")
	for record in dataset[:3]:
		print(
			{
				"subject_id": record["subject_id"],
				"shape": tuple(record["eeg_clean"].shape),
				"source_file": Path(record["source_file"]).name,
			}
		)


if __name__ == "__main__":
	_demo()
