import torch
import numpy as np

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

# Check label indices:
print("LABELS and indices:")
for i, l in enumerate(LABELS):
    print(f"  {i}: {l}")

assert LABELS[2] == "Medial Meniscus"
assert LABELS[3] == "Lateral Meniscus"
assert LABELS[4] == "Medial OA"
assert LABELS[5] == "Lateral OA"

# Test mirror tensor logic:
# x has shape: [batch_size, N_SLOTS, k, 3, SIZE, SIZE]
# slots:
# plane 0: Sagittal (slots 0 and 3)
# plane 1: Coronal (slots 1 and 4)
# plane 2: Axial (slots 2 and 5)

B, N_SLOTS, K, C, H, W = 2, 6, 16, 3, 384, 384
x = torch.randn(B, N_SLOTS, K, C, H, W)

# Mirroring function:
def mirror_volume(x_vol):
    x_m = x_vol.clone()
    # Sagittal slots (0 and 3): reverse slice triplet channel order [2, 1, 0]
    x_m[:, [0, 3], :, :, :, :] = torch.flip(x_m[:, [0, 3], :, :, :, :], dims=[3])
    # Coronal & Axial slots (1, 2, 4, 5): horizontal flip along image width (last dimension)
    x_m[:, [1, 2, 4, 5], :, :, :, :] = torch.flip(x_m[:, [1, 2, 4, 5], :, :, :, :], dims=[-1])
    return x_m

x_mirrored = mirror_volume(x)
assert x_mirrored.shape == x.shape
print("Mirror volume shape verification PASSED:", x_mirrored.shape)

# Test label swapping logic:
def swap_bilateral_labels(p):
    # p shape: [batch, 12]
    p_swapped = p.clone()
    p_swapped[:, 2] = p[:, 3] # Medial Meniscus <- Lateral Meniscus
    p_swapped[:, 3] = p[:, 2] # Lateral Meniscus <- Medial Meniscus
    p_swapped[:, 4] = p[:, 5] # Medial OA <- Lateral OA
    p_swapped[:, 5] = p[:, 4] # Lateral OA <- Medial OA
    return p_swapped

p_dummy = torch.arange(12, dtype=torch.float32).unsqueeze(0).repeat(B, 1)
p_sw = swap_bilateral_labels(p_dummy)
assert (p_sw[:, 2] == p_dummy[:, 3]).all()
assert (p_sw[:, 3] == p_dummy[:, 2]).all()
assert (p_sw[:, 4] == p_dummy[:, 5]).all()
assert (p_sw[:, 5] == p_dummy[:, 4]).all()
assert (p_sw[:, 0] == p_dummy[:, 0]).all() # ACL unchanged
assert (p_sw[:, 1] == p_dummy[:, 1]).all() # MCL unchanged
assert (p_sw[:, 6] == p_dummy[:, 6]).all() # PF OA unchanged
print("Label swapping logic verification PASSED!")
