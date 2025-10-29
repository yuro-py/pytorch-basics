import torch

A = torch.randn(2048, 2048, dtype=torch.bfloat16)
B = torch.randn(2048, 2048, dtype=torch.bfloat16)
ref = torch.mm(A, B)
for _ in range(1000):
    assert (torch.mm(A, B) - ref).abs().max().item() == 0
