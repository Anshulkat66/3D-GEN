import numpy as np

# # ===============================
# # LOAD WINDOW INFO
# # ===============================
# window_ids = np.load("window_ids.npy")   # shape (N,)
# eeg_ids = np.load("eeg_ids.npy")         # shape (N,)

# # ===============================
# # PARAMETERS (VERY IMPORTANT)
# # ===============================
# stride = 600                 # same as used in windowing
# sampling_rate = 250          # Hz
# time_per_image = 2.4         # seconds

# samples_per_image = int(sampling_rate * time_per_image)  # 600
# num_images = 720

# print("Samples per image:", samples_per_image)

# # ===============================
# # BUILD IMAGE IDS
# # ===============================
# image_ids = np.zeros(len(window_ids), dtype=int)

# for i in range(len(window_ids)):

#     # 1. get window start position in EEG signal
#     start_sample = window_ids[i] * stride

#     # 2. convert sample index → image index
#     img_idx = start_sample // samples_per_image

#     # 3. safety clamp
#     if img_idx >= num_images:
#         img_idx = num_images - 1

#     image_ids[i] = img_idx

# # ===============================
# # SAVE
# # ===============================
# np.save("image_ids.npy", image_ids)

# print("✅ image_ids.npy saved")
# print("Example:", image_ids[:20])

image_ids = np.load("image_ids.npy")

clip_embeddings = np.load("clip_embeddings.npy")
clip_targets = clip_embeddings[image_ids]

image_latents = np.load("latent_label.npy")
latent_targets = image_latents[image_ids]

image_names = np.load("image_names.npy")
# print(image_ids)
# print(class_names[image_ids])

for i in range(50):
    img_id = image_ids[i]
    print(i, "-> image:", img_id, "-> class:", image_names[img_id])