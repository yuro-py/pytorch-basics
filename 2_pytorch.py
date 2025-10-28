import torch
import numpy as np

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
print(torch.cuda.is_available())
# b. device agnostic setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
# in future there will be moments where we use this device variable
# c. count gpus
print(torch.cuda.device_count())
