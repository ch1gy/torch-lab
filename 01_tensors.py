import torch

one = torch.rand(3, 3)
two = torch.rand(3, 3)
a = one + two
b = one * two
c = one.reshape(9,)

three = torch.rand(9)
four = torch.rand(9)
d = torch.dot(three, four)
print("Tensor:", a, b, c, d)
print("Shape:", a.shape)
print("Dtype:", a.dtype)
print("Device:", a.device)