import torch

# BROADCASTING : PERFORMING ELEMENTAL OPERATIONS OF TENSORS WITH DIFFERENT SHAPES.
# THE SCALAR * MATRIX WAS A SIMPLER VERSION OF BROADCASTING.

# RULES FOR BROADCASTING. WHEN COMPARING TWO DIMENSIONS FROM THE RIGHT-MOST DIMENSION:-
# 1. THEY ARE EQUAL, OR
# 2. ONE OF THEM IS 1
# IF NONE OF THESE, THEN IT'S AN ERROR.

# COMPATIBLE MATCH -> 3 = 3
a = torch.rand([2, 3])
b = torch.rand([3])
print(f"1 Performing an operation to check errors : {a * b}")

# COMPATIBLE MATCH -> SCALAR IS [1,1] BY DEFAULT
a = torch.rand([2, 3])
b = torch.rand([])
print(f"2 Performing an operation to check errors : {a * b}")

# INCOMPATIBLE MATCH -> 2 != 3
a = torch.rand([3, 2])
b = torch.rand([3])
print(f"3 Performing an operation to check errors : {a * b}")
