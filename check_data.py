from pathlib import Path

import numpy as np


def _find_latent_file() -> Path | None:
	"""Return the first existing latent target file candidate."""
	candidates = [
		Path("latent_targets.npy"),
		Path("outputs/latent_targets.npy"),
		Path("outputs/phase1_output/latent_targets.npy"),
	]

	for path in candidates:
		if path.exists():
			return path
	return None


clip = np.load("clip_embeddings.npy")
# latent_path = _find_latent_file()
print("CLIP length:", len(clip))

latent = np.load("latent_label.npy",allow_pickle=True)
# print("LATENT file:", latent_path)
print("LATENT length:", len(latent))