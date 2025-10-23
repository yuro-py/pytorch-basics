import torch

# TYPES OF TENSORS


# 1.SCALAR(0-D TENSOR)
# It has no axes or directions. It's a single point information like a temperature or grade.
def scalar():
    s1 = torch.tensor(10)
    s2 = torch.tensor(99)
    print("Scalars :-")
    print(s1)
    print(s2)
    print("Addition :")
    print(s1 + s2)
    print("------------------------")


# ----------------------------------------------------------------------------------------------------------------
# 2. VECTOR(1-D TENSOR)
# single dimension/axis/direction.
# best to describe one information with many features like a "house" with [price, room, size].
def vector():
    v1 = torch.tensor([1, 2, 3])
    v2 = torch.tensor([4, 5, 6])
    print("Vectors :-")
    print(v1)
    print(v2)
    print("Addition :-")
    print(v1 + v2)
    print("------------------------")


# ----------------------------------------------------------------------------------------------------------------
# 3. MATRIX(2-D TENSOR)
# two dimensions/axes/directions.
# a grid of rows and columns.
#     can represent a greyscale image(each number is a pixel),
#     or a batch of data where each row is a data point.
def matrix():
    m1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
    m2 = torch.tensor([[7, 8, 9], [1, 2, 3]])
    print("Matrix :-")
    print(m1)
    print(m2)
    print("Addition :-")
    print(m1 + m2)
    print("------------------------")


# ----------------------------------------------------------------------------------------------------------------
# 4. N-D TENSOR
# 3-D is a cube of numbers or a stack of matrices.
# 4-D is a collection of 3Ds and used in images.
# 5-D is a collection of 4Ds and used in videos.
# etc
def t3d():  # a batch of two matrices
    t3d1 = torch.tensor(
        [
            [[1, 2], [3, 4]],  # MATRIX 1
            [[5, 6], [7, 8]],  # MATRIX 2
        ]
    )
    t3d2 = torch.tensor(
        [
            [[1, 3], [5, 7]],  # MATRIX 1
            [[2, 4], [6, 8]],  # MATRIX 2
        ]
    )
    print("3D")
    print(t3d1)
    print(t3d2)
    print("Addition :-")
    print(t3d1 + t3d2)
    print("------------------------")


# execution block
while True:
    print("what to perform?")
    print("1. scalar")
    print("2. vector")
    print("3. matrix")
    print("4. t3d")
    i = int(input("-> "))
    print("")
    if i == 1:
        scalar()
    elif i == 2:
        vector()
    elif i == 3:
        matrix()
    elif i == 4:
        t3d()
    else:
        exit("over")
