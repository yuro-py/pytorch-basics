import torch

start = 0
end = 1
step = 0.02
print(torch.arange(start, end, step).unsqueeze(dim=1))
