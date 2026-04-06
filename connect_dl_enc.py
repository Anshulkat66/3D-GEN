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

