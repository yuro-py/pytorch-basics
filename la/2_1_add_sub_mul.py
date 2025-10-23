import torch


a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[10, 20], [30, 40]])

# 1.. ADDITION/SUBSTRACTION
print(f"Tensor 1 :-\n{a}")
print(f"Tensor 2 :-\n{b}")
print("")
print(f"Add :\n {a + b}")
print(f"Sub :\n {a - b}")
print("")

# 2. ELEMENTAL MULTIPLICATION(OR HADAMARD PRODUCT)
print(f"Hadamard elemental product :\n{a * b}")
print("")
