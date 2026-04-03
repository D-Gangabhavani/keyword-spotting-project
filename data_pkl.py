
import torch
import numpy as np

# dummy input
x = torch.randn(1, 1, 40, 100)

with torch.no_grad():
    y = model(x)

print("Output shape:", y.shape)
print("Raw output:", y)
