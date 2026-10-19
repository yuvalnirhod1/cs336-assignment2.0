import torch.nn as nn

class ToyModel(nn.Module):
     def __init__(self, in_features: int, out_features: int):
         super().__init__()
         self.fc1 = nn.Linear(in_features, 10, bias=False)
         self.ln = nn.LayerNorm(10)
         self.fc2 = nn.Linear(10, out_features, bias=False)
         self.relu = nn.ReLU()

     def forward(self, x):
         print(x.dtype)
         x = self.relu(self.fc1(x))
         print(x.dtype)
         x = self.ln(x)
         print(x.dtype)
         x = self.fc2(x)
         print(x.dtype)
         return x

