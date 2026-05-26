# import matplotlib
# matplotlib.use('TkAgg')  # or 'Agg' if Tk doesn't workimport matplotlib
# matplotlib.use('TkAgg')  # or 'Agg' if Tk doesn't work
# import torch

# from phase_02_mse_tok import EEGEncoder
# from phase_02_DL import create_dataloader

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using device:", device)

# model = EEGEncoder().to(device)
# model.eval()  # no training yet


# data_dir = "outputs/phase1_output"

# loader = create_dataloader(
#     data_dir,
#     batch_size=4,     # keep small initially
#     window_size=1024,
#     stride=512
# )


# for batch in loader:
#     batch = batch.to(device)

#     with torch.no_grad():
#         tokens = model(batch)

#     print("Input shape :", batch.shape)
#     print("Output shape:", tokens.shape)
#     break

# #sanity checks :

# print("Min:", tokens.min().item())
# print("Max:", tokens.max().item())
# print("Mean:", tokens.mean().item())
# print("Std:", tokens.std().item())

# #
# print("Token variance across time:", tokens.var(dim=1).mean().item())

# #
# batch2 = next(iter(loader)).to(device)

# with torch.no_grad():
#     tokens2 = model(batch2)

# print((tokens - tokens2).abs().mean().item())

# #

# print(tokens[0, 0, :10])


# tokens_np = tokens.detach().cpu().numpy()  # [B, T, D]

# # flatten → [B*T, D]
# tokens_flat = tokens_np.reshape(-1, tokens_np.shape[-1])

# print("Tokens shape:", tokens_flat.shape)


# #PCA :
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt

# pca = PCA(n_components=2)

# sample_tokens = tokens_flat[:5000]  # limit for speed
# tokens_2d = pca.fit_transform(sample_tokens)

# plt.figure()
# plt.scatter(tokens_2d[:, 0], tokens_2d[:, 1], s=5)
# plt.title("PCA of Raw Tokens")
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.show()

# #cosine similarity

# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# sample = tokens_flat[:1000]

# sim_matrix = cosine_similarity(sample)

# # remove self-similarity
# sim_matrix = sim_matrix - np.eye(sim_matrix.shape[0])

# avg_sim = sim_matrix.mean()
# max_sim = sim_matrix.max()

# print("Average similarity:", avg_sim)
# print("Max similarity:", max_sim)

# # histogram:
# import matplotlib.pyplot as plt

# values = tokens_flat.flatten()

# plt.figure()
# plt.hist(values, bins=100)
# plt.title("Token Value Distribution")
# plt.xlabel("Value")
# plt.ylabel("Frequency")
# plt.show()


# #random token difference:

# import torch
# import matplotlib.pyplot as plt
# import random

# diffs = []

# B, T, D = tokens.shape

# for i in range(200):
#     b = random.randint(0, B-1)
#     t1 = random.randint(0, T-1)
#     t2 = random.randint(0, T-1)

#     d = torch.mean(torch.abs(tokens[b, t1] - tokens[b, t2])).item()
#     diffs.append(d)

# # scatter plot
# plt.figure()
# plt.scatter(range(len(diffs)), diffs, s=10)
# plt.title("Random Token Differences (Scatter)")
# plt.xlabel("Sample Index")
# plt.ylabel("Difference")
# plt.show()

# #temporal scatter plot:
# temporal_diffs = []

# b = 0  # pick one sample

# for i in range(tokens.shape[1] - 1):
#     d = torch.mean(torch.abs(tokens[b, i] - tokens[b, i+1])).item()
#     temporal_diffs.append(d)

# # scatter plot
# plt.figure()
# plt.scatter(range(len(temporal_diffs)), temporal_diffs, s=10)
# plt.title("Temporal Token Differences (Scatter)")
# plt.xlabel("Token Index (Time)")
# plt.ylabel("Difference")
# plt.show()


# #distance vs difference
# distances = []
# diffs = []

# for i in range(tokens.shape[1]):
#     for j in range(i+1, tokens.shape[1]):
#         d_time = j - i
#         d_feat = torch.mean(torch.abs(tokens[0, i] - tokens[0, j])).item()

#         distances.append(d_time)
#         diffs.append(d_feat)

# # scatter
# import matplotlib.pyplot as plt

# plt.scatter(distances, diffs, s=5)
# plt.xlabel("Time Distance")
# plt.ylabel("Feature Difference")
# plt.title("Distance vs Difference")
# plt.show()

#CHANNEL ABALATION :

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
batch = next(iter(loader))
print(type(batch))
x = batch.to(device)
if isinstance(batch, tuple):
    print(batch[0].shape)
elif isinstance(batch, dict):
    print(batch["eeg"].shape)
else:
    print(batch.shape)
# for batch in loader:
#     batch = batch.to(device)

#     with torch.no_grad():
#         tokens = model(batch)

#     print("Input shape :", batch.shape)
#     print("Output shape:", tokens.shape)
#     break

# import torch
# import torch.nn.functional as F

# def cosine_similarity(a, b):
#     a = F.normalize(a, dim=-1)
#     b = F.normalize(b, dim=-1)
#     return (a * b).sum(dim=-1)

# def mean_abs_diff(a, b):
#     return (a - b).abs().mean().item()


# def channel_ablation_test(encoder, x, channel_idx):
#     x_orig = x.clone()

#     x_ablate = x.clone()
#     x_ablate[:, :, channel_idx] = 0

#     with torch.no_grad():
#         t1 = encoder(x_orig)
#         t2 = encoder(x_ablate)

#     diff = mean_abs_diff(t1, t2)
#     print(f"Channel {channel_idx} diff: {diff:.6f}")
#     return diff

# for ch in [0, 10, 20, 30, 40, 50]:
#     channel_ablation_test(model, x, ch)

#Frequency Sensitivity test:

import torch

def fft_filter(x, low=None, high=None, fs=128):
    # x: [B, T, C]
    Xf = torch.fft.rfft(x, dim=1)
    freqs = torch.fft.rfftfreq(x.size(1), d=1/fs).to(x.device)

    mask = torch.ones_like(freqs)

    if low is not None:
        mask = mask * (freqs >= low)
    if high is not None:
        mask = mask * (freqs <= high)

    Xf_filtered = Xf * mask[None, :, None]

    return torch.fft.irfft(Xf_filtered, n=x.size(1), dim=1)

def frequency_test(encoder, x):
    with torch.no_grad():
        t_orig = encoder(x)

        x_low = fft_filter(x, low=0.5, high=4)     # delta band
        x_high = fft_filter(x, low=30, high=45)    # gamma band

        t_low = encoder(x_low)
        t_high = encoder(x_high)

    diff_low = (t_orig - t_low).abs().mean().item()
    diff_high = (t_orig - t_high).abs().mean().item()

    print(f"Low freq diff  (0.5–4 Hz):  {diff_low:.6f}")
    print(f"High freq diff (30–45 Hz): {diff_high:.6f}")

    return diff_low, diff_high

frequency_test(model, x)