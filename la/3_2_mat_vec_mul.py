import torch

# DIFFERNENT THAN ELEMENT-WISE MULTIPLICATION

a = torch.tensor([[1, 2, 3], [4, 5, 6]])  # SHAPE = (2,3)
b = torch.tensor([10, 20, 30])  # SHAPE = (3,)

# a's COLUMNS AND b's NUMBER OF ELEMENTS MATCH. THIS WILL WORK.
# USE "@" OR "torch.matmul()" for MULTIPLICATION.

print(f"Using @ : {a @ b}")
print(f"Using torch.matmul : {torch.matmul(a, b)}")

# SLIGHT DIFFERENCE BETWEEN USING "@" AND "torch.matmul" BUT EXCLUSIVE FOR SPECIFIC CASES, NOT GENERAL ONES.
