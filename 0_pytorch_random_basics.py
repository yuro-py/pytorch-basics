# NOTICE -> COMMENTED OUT STUFF. UNCOMMENT WHATEVER PART YOU WANT TO RUN.
# NOT MADE FOR RUNNING IN ONE GO. WILL ADD HEADING AND MAKE IT A CLEAR "RUN-IN-ONE-GO" THING LATER.

# https://youtu.be/V_xro1bcAuA?si=fkVRlmOqcXsu7FBr following this video

import torch
import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# a = torch.tensor([2, 2], dtype=torch.int32, device="cpu", requires_grad=False)
# print(a)
# device can be cpu, gpu, cuda etc. two co-operational tensors should have the same device.
# requires_grad is a bool var. it asks if tracking of the tensor for gradient computation. this feature is called autograd(automatic differentiation).
a = torch.tensor
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
# INDEXING/SLICING :-
t = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
# print(f"INDEXING/SLICING on this tensor :-\n{t}\n\n")
# print(f"Getting a single element -> {t[0, 1].item()}\n\n")
# print(f"Getting 0th row ->{t[0, :]}\n\n")
# print(f"All rows specific columns ->{t[:, 2]}\n\n")
# print(f"Specific rows all columsn :-\n{t[0, :]}\n\n")
# print(f"Sub matrix :-\n{t[0:2, 0:0]}")  # 3 7 1 5
# print(f"Negative indexing -> \n{t[-1]}")

# ---------------------------------------------------------------------------------------------------------------
# 1. "NUMPY TO TENSOR" and "TENSOR TO NUMPY"
# a.
array = np.arange(1.0, 10.0)
tensor = torch.from_numpy(array)  # THIS!
# after conversion, specify any type using eg ".type(torch.float32)" you want in the last
# print(array, tensor)
# print(array.shape, tensor.shape)
# b.
tensor = torch.ones(9)
array = tensor.numpy()  # THIS!
# print(tensor, array)
# print(tensor.shape, array.shape)


# 2. REPRODUCIBILITY : trying to take a random out of a random
# NOTES -> HOW A NEURAL NETWORK LEARNS :-
# STARTS WITH RANDOM NUMS -> TENSOR OPS -> UPDATE RANDOM NUMBERS TO MAKE THEM A BETTER REPRESENTATIONS OF THE DATA -> REPEAT...
# TO REDUCE RANDOMNESS EACH TIME : INTRODUCE A "RANDOM SEED"
torch.manual_seed(42)
t1 = torch.rand(3, 4)
torch.manual_seed(42)
# RESET THE SEED BEFORE EACH NEW RANDOM VARIABLE TO ACHIEVE REPRODUCIBILITY
t2 = torch.rand(3, 4)
# print(t1)
# print(t2)
# print(t1 == t2)

# 3. CHECK GPU AVAILABILITY THROUGH PYTORCH
# a. check availability
# print(torch.cuda.is_available())
# b. device agnostic setup
device = "cuda" if torch.cuda.is_available() else "cpu"
# print(device)
# in future there will be moments where we use this device variable
# c. count gpus
# print(torch.cuda.device_count())
# d. lot more weird gpu accessing methods. will reference that section later.
# ---------------------------------------------------------------------------------------------------------------
