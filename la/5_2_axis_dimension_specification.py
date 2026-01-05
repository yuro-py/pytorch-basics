import torch

# AXIS/DIMENSION SPECIFICATION(basically using "dim" parameter)
a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(f"The tensor :-\n{a}\n")

# ROW COLLAPSE(either sum or mean) :-
b = torch.sum(a, dim=0)  # using dimension 0, which is "rows"
print(f"Sum of rows : {b}")
b = torch.mean(a, dim=0)  # using dim 0
print(f"Mean of rows : {b}")

# COLUMN COLLAPSE(either sum or mean) :-
b = torch.sum(a, dim=1)  # dimension 1 = column
print(f"Sum of columns : {b}")
b = torch.mean(a, dim=1)
print(f"Mean of columns : {b}")

# MAIN GOAL HERE ACHIEVED WAS COLLAPSING, NOT AN ENTIRE TENSOR, BUT SELECTIVELY ROWS OR COLUMNS
