import torch

v = torch.tensor([1,2,3,4,5,6])
print(v, v.dtype, v[1:4])

f = torch.FloatTensor([1,2,3,4,5,6])
print(f, f.dtype, f[1:4])

# Reshape
print(v.view(6,1))

# Numpy to Tensor
import numpy as np

a = np.array([1,2,3,4,5])
tensor_converted = torch.from_numpy(a)
print(tensor_converted, tensor_converted.type())

# Tensor to Numpy
numpy_converted = tensor_converted.numpy()
print(numpy_converted)

# Print out 100 equal spaced numbers between 0 to 10
print(torch.linspace(0,10, steps = 100))
x = torch.linspace(0,10, steps = 100)
y = torch.exp(x)
siny = torch.sin(x)
import matplotlib.pyplot as plt

img = plt.plot(x.numpy(), y.numpy())
