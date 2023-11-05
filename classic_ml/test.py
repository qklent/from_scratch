import numpy as np


w = np.random.rand(100, 1)
g = np.random.rand(100).reshape(100, 1)
q = w - g
print(q.shape)
