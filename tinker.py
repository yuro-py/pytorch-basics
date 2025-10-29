import torch

a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
b = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
ref = torch.mm(a, b)
for _ in range(1000):
    assert (torch.mm(a, b) - ref).abs().max().item() == 0
