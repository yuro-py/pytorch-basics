import torch

# REDUCING AN ENTIRE TENSOR TO A SCALAR VALUE.
# USE CASE IN DL WHERE THE LOSS IS CALCULATED BY ADDING UP OR AVERAGING THE TENSOR.

a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # USING FLOATS IS A GOOD PRACTICE
print(f"The tensor :-\n{a}")
print("")

# FEW DAYS TO ACHIEVE THIS :-
# 1. SUMMING UP ALL ELEMENTS DIRECTLY :-
print(f"Summing up all elements : {torch.sum(a)}")
# outputs "21"

# 2. MEAN/AVERAGE :-
print(f"Average of all elements : {torch.mean(a)}")
# outputs "3.5000"

# 3. MIN/MAX ELEMENT IN THE TENSOR:-
print(f"MIN & MAX : {torch.min(a)} and {torch.max(a)}")
# outputs "1.0" and "6.0"
