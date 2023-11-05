import numpy as np
from sklearn.linear_model import LinearRegression
import torch
from torch import nn


seed = 1337
np.random.seed(seed)
torch.manual_seed(seed)

batch_size = 32
in_features = 100
num_iters = 100000
lr = 1e-2

# my model
weights = np.random.rand(in_features, 1)
bias = 0
X = np.random.rand(batch_size, in_features)
y_true = np.random.randn(batch_size, 1).round()

y_pred = X @ weights


def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).sum() / len(y_pred)


prev_loss = 0

for i in range(num_iters):
    y_pred = X @ weights + bias
    grad = -2 / len(y_pred) * (y_true - y_pred) * X
    grad = grad.mean(axis=0).reshape(weights.shape)

    weights -= lr * grad

    bias_grad = -2 / len(y_pred) * (y_true - y_pred) * 1
    bias = lr * (bias - bias_grad.mean())

print(f"loss for my model = {mse_loss(y_true, y_pred)}")


# sklearn model
# model = LinearRegression()
# model.fit(X, y_true)

# y_pred = model.predict(X)

# print (f"loss for sklearn model = {mse_loss(y_true, y_pred)}")


# # pytorch model
# model = nn.Sequential(
#     nn.Linear(in_features, 1)
# )
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
# loss_fn = torch.nn.MSELoss()
# X = torch.from_numpy(X).view(batch_size, in_features).to(torch.float32)
# y_true = torch.from_numpy(y_true).to(torch.float32)
# # print(y_pred.squeeze(dim=-1).shape)
# for i in range(num_iters):
#     y_pred = model(X)
#     loss = loss_fn(y_pred, y_true)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
# # print (loss.item())
# print (f"loss for torch model = {loss}")
