import torch

a = [i%5 for i in range(0, 16 * 16)]
a = torch.Tensor(a).reshape(-1, 16)

b = [i%5 + i % 4 for i in range(0, 16 * 16)]
b = torch.Tensor(b).reshape(-1, 16)

print(a)
print(b)
print(torch.matmul(a, b)[-1].tolist())
