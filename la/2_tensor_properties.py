import torch

# TOPIC : PROPERTIES OF TENSORS

s = torch.tensor(25)  # SCALAR
v = torch.tensor([10, 20, 30])  # VECTOR
m = torch.tensor([[1, 2, 3], [4, 5, 6]])  # MATRIX

print("Scalar :", s)
print("Vector :", v)
print("Matrix :\n", m)
print("")


# 1. RANKS : It's the simplest property, "rank" is the number of dimensions of a tensor.
# NDIM = Number of Dimensions
def rank():
    print(f"Scalar rank : {s.ndim}")
    print(f"Vector rank : {v.ndim}")
    print(f"Matrix rank : {m.ndim}")


# 2. SHAPE : It's a tuple which tells the dimensions in the form of "rows x columns x etc"
def shape():
    print(f"Scalar shape : {s.shape}")
    print(f"Vector shape : {v.shape}")
    print(f"Matrix shape : {m.shape}")


# 3. DATA TYPES : Tells the type of numbers inside a tensor.
def datype():
    t_1 = torch.tensor([1.0, 2.0])  # has decimals
    print(t_1)
    print(t_1.dtype)
    print("")

    # manually specifying a dtype
    t_2 = torch.tensor([1, 2], dtype=torch.float64)
    print(t_2)
    print(t_2.dtype)
    print("")

    # bool
    t_3 = torch.tensor(1, dtype=torch.bool)
    print(t_3)
    print(t_3.dtype)
    print("")


while True:
    print("""what to perform?
        1. rank checking
        2. shape checking
        3. datatype checking""")
    a = int(input("-> "))
    if a == 1:
        rank()
    elif a == 2:
        shape()
    elif a == 3:
        datype()
    else:
        exit("over")
