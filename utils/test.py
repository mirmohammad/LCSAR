import torch

y = torch.arange(1, 10)

print(y)

print(y.unfold(0, 3, 4))

x = torch.randn(1, 500, 500, 500)  # batch, c, h, w
kc, kh, kw = 64, 64, 64  # kernel size
dc, dh, dw = 64, 64, 64  # stride
patches = x.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
print(patches.size())
unfold_shape = patches.size()
patches = patches.contiguous().view(patches.size(0), -1, kc, kh, kw)
print(patches.shape)

# Reshape back
patches_orig = patches.view(unfold_shape)
output_c = unfold_shape[1] * unfold_shape[4]
output_h = unfold_shape[2] * unfold_shape[5]
output_w = unfold_shape[3] * unfold_shape[6]
patches_orig = patches_orig.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
patches_orig = patches_orig.view(1, output_c, output_h, output_w)

# Check for equality
print((patches_orig == x[:, :output_c, :output_h, :output_w]).all())