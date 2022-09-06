from torch.nn import Linear
import torch

w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

#y = w*x + b

def forward(x):

    y = w*x + b

    return y

x = torch.tensor(2)
print(forward(x))

torch.manual_seed(1)
model = Linear(in_features=1, out_features=1)
print(model.bias, model.weight)

