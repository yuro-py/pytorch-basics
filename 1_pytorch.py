# https://youtu.be/V_xro1bcAuA?si=fkVRlmOqcXsu7FBr following this video
import torch
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

a = torch.tensor([2, 2], dtype=torch.int32, device="cpu", requires_grad=False)
# print(a)
# device can be cpu, gpu, cuda etc. two co-operational tensors should have the same device.
# requires_grad is a bool var. it asks if tracking of the tensor for gradient computation. this feature is called autograd(automatic differentiation).

# --------------------------------------------------------
# MATRIX MULTIPLICATION
# torch.matmul = torch.mm = @
# inner values(a's row and b's column) should be same. or else error in multiplication.
# use http://matrixmultiplication.xyz/ to visualize
# --------------------------------------------------------

# --------------------------------------------------------
# for reducing the size of the tensor : sum, mean, min, max; can specify the dim asw.
# for finding the index quantity use "var.argmax()" or "var.argmin()"
# --------------------------------------------------------

# --------------------------------------------------------
# Manipulating shapes/dimensions :-
# 1. reshaping/adding dimensions : reshapes a tensor. for eg : shape(1,10) can be "reshape(9,1)".
# x = torch.arange(1, 10)  # 1-9
x = torch.tensor([[1, 2], [3, 4], [5, 6]])
print(x)
# xr = x.reshape(1, 9)
# print(xr)  # print(x.shape, xr.shape)
print()

# 2. view : without creating a new one in memory, returns another reshaped view of an input tensor.
# z = x.view(1, 9)  # same as reshape using x's memory. hence, changing z changes x.
# print(z)

# 3. stacking : combine multiple tensors on top. torch.stack(for basic stack), torch.vstack(for vertical) and torch.hstack(for side by side).
x_stack = torch.stack(
    [x, x, x, x], dim=2
)  # IMPORTANT : the higher the dim value, the inner the brackets go. lowest dim = outermost bracket.
print(x_stack)
# print(x.shape)
# print()

# 4. squeeze :
# print()

# 5. unsqueeze :
# print()

# 6. permute : returns a view of tensor with permuted dimensions.
# print()
