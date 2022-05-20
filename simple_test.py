import numpy
import numpy as np
import torch
from tqdm import tqdm
import time
# batch: scale * pts
# 3x3
# a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# w = np.array([10, 20, 30])
#
# w2 = numpy.reshape(w, (-1, 1))
# o = a*w2


# a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
# aa = np.array([a, a, a, a, a])  # 5x4x3
# w = np.array([[10, 10, 10], [20, 20, 20], [30, 30, 30], [40, 40, 40], [50, 50, 50]])  # 5x3
# aa2 = torch.from_numpy(aa).transpose(1, 2)  # 5x3x4
# w1 = w.reshape((5, 3, 1))  # 5x3x1
# w2 = torch.from_numpy(w1)
# o = aa2 + w2


a = torch.arange(3, 12).view(1, 3, 3)
b = torch.Tensor([10, 20, 30]).view(1, 3, 1)
b1 = torch.Tensor([11, 21, 31]).view(1, 3, 1)
b2 = torch.Tensor([12, 22, 32]).view(1, 3, 1)
b3 = torch.Tensor([13, 23, 33]).view(1, 3, 1)
aa = torch.cat((a, a, a, a), 0)
bb = torch.cat((b, b1, b2, b3), 0)
print(aa.shape)
print(bb.shape)
c = torch.cat((aa, bb), 2)

bottom = torch.zeros(aa.shape[0], 1, 4)
bottom[:, :, 3] = 1
d = torch.cat((c, bottom), 1)

print(c)
print(d)
print(d[0])




