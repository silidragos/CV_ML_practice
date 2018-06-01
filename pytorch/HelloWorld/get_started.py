import torch

# x = torch.empty(3,5)
# print(x)

# x = torch.rand(5, 3)
# print(x)

# x = torch.zeros(5, 3, dtype = torch.long)
# print(x)

# x = torch.tensor([5.5, 3])
# print(x)

# x = x.new_ones(5, 3, dtype=torch.double)
# print(x)
# x = torch.randn_like(x, dtype=torch.float)
# print(x)


###Operations
# x = torch.rand(5, 3)
# y = torch.rand(5, 3)
# print(x+y)
# print(torch.add(x, y))

# result = torch.empty(5, 3)
# torch.add(x, y, out=result)
# print(result)

# print(x[:, 1])


###.numpy() .from_numpy()

# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))
else:
    print("No cuda for you :(")