import pandas as pd
import torch

a = torch.Tensor([1,2,3,4,5,6])
print(a.reshape(-1,2))
