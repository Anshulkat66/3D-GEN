import matplotlib
matplotlib.use('TkAgg')  # or 'Agg' if Tk doesn't workimport matplotlib
matplotlib.use('TkAgg')  # or 'Agg' if Tk doesn't work
import torch

from phase_02_mse_tok import EEGEncoder
from phase_02_DL import create_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = EEGEncoder().to(device)
model.eval()  # no training yet


data_dir = "outputs/phase1_output"

loader = create_dataloader(
    data_dir,
    batch_size=4,     # keep small initially
    window_size=1024,
    stride=512
)


for batch in loader:
    batch = batch.to(device)

    with torch.no_grad():
        tokens = model(batch)

    print("Input shape :", batch.shape)
    print("Output shape:", tokens.shape)
    break

#sanity checks :

print("Min:", tokens.min().item())
print("Max:", tokens.max().item())
print("Mean:", tokens.mean().item())
print("Std:", tokens.std().item())

#
print("Token variance across time:", tokens.var(dim=1).mean().item())

#
batch2 = next(iter(loader)).to(device)

with torch.no_grad():
    tokens2 = model(batch2)

print((tokens - tokens2).abs().mean().item())

#

print(tokens[0, 0, :10])


tokens_np = tokens.detach().cpu().numpy()  # [B, T, D]

# flatten → [B*T, D]
tokens_flat = tokens_np.reshape(-1, tokens_np.shape[-1])

print("Tokens shape:", tokens_flat.shape)


#PCA :
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)

sample_tokens = tokens_flat[:5000]  # limit for speed
tokens_2d = pca.fit_transform(sample_tokens)

plt.figure()
plt.scatter(tokens_2d[:, 0], tokens_2d[:, 1], s=5)
plt.title("PCA of Raw Tokens")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

#cosine similarity

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

sample = tokens_flat[:1000]

sim_matrix = cosine_similarity(sample)

# remove self-similarity
sim_matrix = sim_matrix - np.eye(sim_matrix.shape[0])

avg_sim = sim_matrix.mean()
max_sim = sim_matrix.max()

print("Average similarity:", avg_sim)
print("Max similarity:", max_sim)

# histogram:
import matplotlib.pyplot as plt

values = tokens_flat.flatten()

plt.figure()
plt.hist(values, bins=100)
plt.title("Token Value Distribution")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()