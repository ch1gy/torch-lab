import torch

x = torch.tensor([4.0], requires_grad=True)
y = x ** 2
y.backward(retain_graph=True)
y.backward()


z = 0.1 * x.grad

print("Tensor:", x)
print(x.grad)
print(y)
print(z)
print("Shape:", x.shape)
print("Dtype:", x.dtype)
print("Device:", x.device)