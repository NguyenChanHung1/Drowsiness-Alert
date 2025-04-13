import torch

a = torch.Tensor([[1,2,3]])
b = torch.Tensor([[4,5,6]])
c = [a,b]
# d = torch.Tensor(c)
d = torch.vstack([a,b])
print(d.shape)