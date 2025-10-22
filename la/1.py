import torch

# TYPES OF TENSORS

# 1.SCALAR(0-D TENSOR)
# It has no axes or directions. It's a single point information like a temperature or grade.

s1 = torch.tensor(10)
s2 = torch.tensor(99)
print(s1 + s2)

# ----------------------------------------------------------------------------------------------------------------
# 2. VECTOR(1-D TENSOR)
# single dimension/axis/direction.
# best to describe one information with many features like a "house" with [price, room, size].

v1 = torch.tensor([1, 2, 3])
v2 = torch.tensor([4, 5, 6])
print(v1 + v2)

# ----------------------------------------------------------------------------------------------------------------
# 3. MATRIX(2-D TENSOR)
#
