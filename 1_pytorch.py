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
#
