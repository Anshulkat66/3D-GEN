"""Phase 1: EEG data preparation.

This module focuses on loading and indexing preprocessed EEG trials:
- Locate subjects under data/EEGdatanpy/
- Load process_data_1s_250Hz.npy and name_label.npy for each subject
- Map trials to their respective labels/image names
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "EEGdatanpy"


def load_subject_data(subject_dir: Path) -> Dict[str, Any]:
    """Load preprocessed EEG trials and labels for a single subject folder.

    Expected structure:
    subject_dir/subject_dir/process_data_1s_250Hz.npy
    subject_dir/subject_dir/name_label.npy
    """
    sub_name = subject_dir.name
    inner_dir = subject_dir / sub_name

    eeg_path = inner_dir / "process_data_1s_250Hz.npy"
    label_path = inner_dir / "name_label.npy"

    if not eeg_path.exists():
        raise FileNotFoundError(f"EEG file not found: {eeg_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    # Load data
    eeg_data = np.load(eeg_path)  # shape: [1728, 64, 250]
    labels = np.load(label_path, allow_pickle=True)  # shape: [1728]

    # Verify alignment consistency
    if len(eeg_data) != len(labels):
        raise ValueError(
            f"Mismatch between EEG trials ({len(eeg_data)}) and labels ({len(labels)}) for {sub_name}."
        )

    return {
        "subject_id": sub_name,
        "eeg_data": eeg_data,       # [1728, 64, 250]
        "labels": labels,           # [1728]
        "n_trials": len(labels),
        "n_channels": eeg_data.shape[1],
        "n_samples": eeg_data.shape[2],
    }


def load_all_subjects(data_dir: Path | str = DEFAULT_DATA_DIR) -> Dict[str, Dict[str, Any]]:
    """Scan the data directory and load trials/labels for all subjects."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    subject_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    dataset = {}

    for sub_dir in subject_dirs:
        sub_name = sub_dir.name
        # Skip hidden or configuration folders
        if sub_name.startswith("."):
            continue
        try:
            dataset[sub_name] = load_subject_data(sub_dir)
        except Exception as e:
            print(f"[Warning] Failed to load data for subject {sub_name}: {e}")

    return dataset


if __name__ == "__main__":
    # Quick smoke test when running directly (without execution as requested)
    print("Scanning subject directories...")
    try:
        data = load_all_subjects()
        print(f"Successfully loaded {len(data)} subjects:")
        for sub_id, record in data.items():
            print(
                f"  - {sub_id}: EEG shape {record['eeg_data'].shape}, "
                f"Labels length {len(record['labels'])}"
            )
    except Exception as e:
        print(f"Error loading subjects: {e}")
