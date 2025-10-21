import torch

# TYPES OF TENSORS

# 1.SCALAR(0-D TENSOR)
# It has no axes or directions. It's a single point information like a temperature or grade.

s1 = torch.tensor(10)
s2 = torch.tensor(99)

print(s1 + s2)  # addition
# similarly other operations can be performed on scalars and it will behave as expected.

# ----------------------------------------
# this is in vim mode
