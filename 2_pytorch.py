import torch
import numpy as np

# 1. numpy to tensor
array = np.arange(1.0, 10.0)
tensor = torch.from_numpy(array)  # THIS!
# after conversion, specify any type using eg ".type(torch.float32)" you want in the last
# print(array, tensor)
# print(array.shape, tensor.shape)

# 2. tensor to numpy
tensor = torch.ones(9)
array = tensor.numpy()  # THIS!
# print(tensor, array)
# print(tensor.shape, array.shape)

# 3. Reproducibility : trying to take a random out of a random
# NOTES -> HOW A NEURAL NETWORK LEARNS :-
# STARTS WITH RANDOM NUMS -> TENSOR OPS -> UPDATE RANDOM NUMBERS TO MAKE THEM A BETTER REPRESENTATIONS OF THE DATA -> REPEAT...
# TO REDUCE RANDOMNESS EACH TIME : INTRODUCE A "RANDOM SEED"
torch.manual_seed(42)
t1 = torch.rand(3, 4)
torch.manual_seed(42)
# RESET THE SEED BEFORE EACH NEW RANDOM VARIABLE TO ACHIEVE REPRODUCIBILITY
t2 = torch.rand(3, 4)
print(t1)
print(t2)
print(t1 == t2)
