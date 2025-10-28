# COMMENTED OUT STUFF. UNCOMMENT WHATEVER PART YOU WANT TO RUN.
# NOT MADE FOR RUNNING IN ONE GO. WILL ADD HEADING AND MAKE IT A CLEAR "RUN-IN-ONE-GO" THING.

# https://youtu.be/V_xro1bcAuA?si=fkVRlmOqcXsu7FBr following this video
import torch
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# a = torch.tensor([2, 2], dtype=torch.int32, device="cpu", requires_grad=False)
# print(a)
# device can be cpu, gpu, cuda etc. two co-operational tensors should have the same device.
# requires_grad is a bool var. it asks if tracking of the tensor for gradient computation. this feature is called autograd(automatic differentiation).

# ---------------------------------------------------------------------------------------------------------------
# MATRIX MULTIPLICATION
# torch.matmul = torch.mm = @
# inner values(a's row and b's column) should be same. or else error in multiplication.
# use http://matrixmultiplication.xyz/ to visualize
# ---------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------
# for reducing the size of the tensor : sum, mean, min, max; can specify the dim asw.
# for finding the index quantity use "var.argmax()" or "var.argmin()"
# ---------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------
# Manipulating shapes/dimensions :-
# 1. reshaping/adding dimensions : reshapes a tensor. for eg : shape(1,10) can be "reshape(9,1)".
# x = torch.arange(1, 10)  # 1-9
# x = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
# print(x.shape)
# xr = x.reshape(1, 9)
# print(xr)  # print(x.shape, xr.shape)
# print()

# 2. view : without creating a new one in memory, returns another reshaped view of an input tensor.
# z = x.view(1, 9)  # same as reshape using x's memory. hence, changing z changes x.
# print(z)

# 3. stacking : combine multiple tensors on top. torch.stack(for basic stack), torch.vstack(for vertical) and torch.hstack(for horizontal).
# x_stack = torch.stack(
#    [x, x, x, x], dim=2
# )  # IMPORTANT : the higher the dim value, the inner the brackets go. lowest dim = outermost bracket. highest dim = treatment as scalars.
# print(x_stack)
# print(x.shape)
# print()

# 4. squeeze : removes dimensions(compress)
# xs = x.squeeze(0)
# print(xs, xs.shape)
# print()

# 5. unsqueeze : adds dimensions(expand)
# xu = xs.unsqueeze(0)
# print(xu, xu.shape)
# print()

# 6. permute : returns a view of tensor with permuted dimensions. basically changes the order of the shape.
# example "shape(0,1,2)". the dimensions can be ordered by permuting and made into "shape(2,0,1)".
# xp = x.permute(1, 0, 2)
# print(xp.shape)
# print()
# ---------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------
# 1. INDEXING/SLICING :-
t = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
print(f"INDEXING/SLICING on this tensor :-\n{t}\n\n")
print(f"Getting a single element -> {t[0, 1].item()}\n\n")
print(f"Getting 0th row ->{t[0, :]}\n\n")
print(f"All rows specific columns ->{t[:, 2]}\n\n")
print(f"Specific rows all columsn :-\n{t[0, :]}\n\n")
print(f"Sub matrix :-\n{t[0:2, 0:0]}")  # 3 7 1 5
print(f"Negative indexing -> \n{t[-1]}")
