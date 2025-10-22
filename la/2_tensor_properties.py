import torch

# TOPIC : PROPERTIES OF TENSORS

s = torch.tensor(25)  # SCALAR
v = torch.tensor([10, 20, 30])  # VECTOR
m = torch.tensor([[1, 2, 3], [4, 5, 6]])  # MATRIX

# 1. RANKS : It's the simplest property, "rank" is the number of dimensions of a tensor.
# NDIM = Number of Dimensions
print(f"Scalar rank : {s.ndim}")
print(f"Vector rank : {v.ndim}")
print(f"Matrix rank : {m.ndim}")


# 2. SHAPE
