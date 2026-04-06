import numpy as np
from pathlib import Path

DATA_DIR = Path("outputs/phase1_output")  # adjust if needed

files = sorted(DATA_DIR.glob("*.npy"))

if not files:
    print("❌ No .npy files found!")
    exit()

print(f"✅ Checking {len(files)} subjects...\n")


# 🔥 TRACK GLOBAL STATS
all_shapes = set()
nan_files = []
inf_files = []
bad_mean = []
bad_std = []
extreme_values = []

global_min = float("inf")
global_max = float("-inf")


for f in files:
    eeg = np.load(f)

    # -------------------------
    # SHAPE CHECK
    # -------------------------
    all_shapes.add(eeg.shape)

    # -------------------------
    # NaN / INF CHECK
    # -------------------------
    if np.isnan(eeg).any():
        nan_files.append(f.name)

    if np.isinf(eeg).any():
        inf_files.append(f.name)

    # -------------------------
    # NORMALIZATION CHECK
    # -------------------------
    mean = eeg.mean()
    std = eeg.std()

    if abs(mean) > 0.1:
        bad_mean.append((f.name, mean))

    if not (0.8 < std < 1.2):
        bad_std.append((f.name, std))

    # -------------------------
    # VALUE RANGE CHECK
    # -------------------------
    min_val = eeg.min()
    max_val = eeg.max()

    global_min = min(global_min, min_val)
    global_max = max(global_max, max_val)

    if abs(min_val) > 10 or abs(max_val) > 10:
        extreme_values.append((f.name, min_val, max_val))


# =========================
# 📊 FINAL REPORT
# =========================

print("📊 ===== DATASET SUMMARY =====\n")

# Shapes
print("🔹 Unique shapes found:")
for shape in all_shapes:
    print("  ", shape)

# NaN / INF
print("\n🔹 NaN files:", nan_files if nan_files else "None")
print("🔹 Inf files:", inf_files if inf_files else "None")

# Mean
print("\n🔹 Mean issues:")
if bad_mean:
    for name, m in bad_mean[:5]:
        print(f"  {name}: mean={m:.4f}")
else:
    print("  None")

# Std
print("\n🔹 Std issues:")
if bad_std:
    for name, s in bad_std[:5]:
        print(f"  {name}: std={s:.4f}")
else:
    print("  None")

# Range
print("\n🔹 Global value range:")
print(f"  Min: {global_min:.4f}")
print(f"  Max: {global_max:.4f}")

# Extreme values
print("\n🔹 Extreme value files:")
if extreme_values:
    for name, mn, mx in extreme_values[:5]:
        print(f"  {name}: min={mn:.2f}, max={mx:.2f}")
else:
    print("  None")

#random subject visual check :
# import matplotlib.pyplot as plt
# import random

# sample_files = random.sample(files, 3)

# for f in sample_files:
#     eeg = np.load(f)
#     plt.plot(eeg[:500, 0])
#     plt.title(f.name)
#     plt.show()

print("\n✅ CHECK COMPLETE")