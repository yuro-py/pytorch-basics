import torch

# 1. INDEXING
t = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
print(f"The tensor we are working on :-\n {t}")
print("----------------------")
print(f"Getting a single element : {t[1, 2]}")
print("")
print(f"Getting a row : {t[0]}")
print("----------------------")

# 2. SLICING
t = torch.tensor([[5, 6, 7, 8], [1, 2, 3, 4]])
print(f"All rows, specific columns : {t[:, 2]}")
print("")
print(f"Specific rows, all columns : {t[1, :]}")
print("")
print(f"Sub-matrix :-\n {t[0:2, 1:3]}")
print("")
print(f"Negative indexing : {t[-2]}")
