import torch

# 1. TRANSPOSE = FLIPS. ROW BECOMES COLUMNS AND COLUMNS BECOMES ROWS.
a = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[6, 5, 4], [3, 2, 1]]])
print("a matrix :- \n", a)
print(f"Transpose of a :-\n{a.mT}")
# ".T" FOR TRANSPOSING.
# ".mT" FOR BIGGER TRANSPOSING WITH DIMENSIONS > 2.
print("")

# 2. RESHAPE : CHANGING THE SHAPE OF THE TENSOR WHILE RETAINING THE NUMBER OF ELEMENTS.
# shape(12) or (3,4) and be changed to (4,3) or (2,6) or (12,1).
b = torch.rand(2, 3, 4, 4)
c = b.reshape(2, -1)
print(b.shape)
print(c.shape)
