import numpy as np
import torch

#create tensor
data = ([1,2,3,4])
t = torch.tensor(data)
#print(t)

# from numpy
n = np.array(data)
t = torch.from_numpy(n)
#print(t)

# tensor-to-tensor : ones, zeroes or rand
x_ones = torch.ones_like(t, dtype = torch.float)
#print(x_ones)

tensor = torch.rand(3,4)
#print(f"shape : {tensor.shape}")
#print(f"data type : {tensor.dtype}")
#print(f"stored on device : {tensor.device}")

# operations
x = torch.linspace(0,5,101)
y1, y2 = x.sin(), x ** x.cos()
print(f"y1 : {y1}")
print(f"y2 : {y2}")

y3 = y2 - y1
y4 = y3.min()
print(y1, y2)
print(y3, y4)
