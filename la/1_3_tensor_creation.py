import torch

# CREATION OF TENSORS

# 1. CONVERTING EXISTING DATA(for eg list).
# PYOTORCH WILL INFER THE SHAPE AND DTYPE FROM THE DATA
a = torch.tensor([1, 2, 3])
print(f"Vector : {a}")
print("")

# 2. FORCING A DTYPE
b = torch.tensor([4, 5, 6], dtype=torch.float32)
print(f"Forced dtype on a vector : {b}")
print("")

# 3. RANDOMS: rand AND randn
r1 = torch.rand(2, 2)  # GENERATES BETWEEN 0 AND 1
r2 = torch.randn(2, 2)  # GENERATES IN A BELL CURVE WITH 0 AS CENTER
print(f"Using rand : {r1}")
print(f"Using randn : {r2}")
print("")

# 4. USING "like" ON "zeros", "ones", "rand" AND "randn"
a = torch.tensor([[1, 2], [3, 4]])
b = torch.ones_like(a)  # dtype OPTIONAL
print(f"Using like : {b}")
