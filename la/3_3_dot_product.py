import torch

# DOT PRODUCT : MATRIX-MATRIX MULTIPLICATION
# CONDITION : COLUMNS OF MATRIX A == ROWS OF MATRIX B
# shape(m,n) @ shape(n,p) -> shape(m,p)

a = torch.tensor([[1, 2, 3], [4, 5, 6]])
b = torch.tensor([[10, 11], [20, 21], [30, 31]])
# a's columns = b's rows. compatible.

print(f"Using @ : {a @ b}")
print(f"Using torch.matmul : {torch.matmul(a, b)}")
