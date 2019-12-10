import numpy as np

x = np.random.rand(8, 3, 20, 40)
# print(arr)
B, C, H, W = x.shape
print(B, C, W, H)

if H > W:
    h = H // 2
    x = x[:, :, h - W // 2:h + W // 2, :]
else:
    h = W // 2
    x = x[:, :, :, h - H // 2:h + H // 2]

print(x.shape)