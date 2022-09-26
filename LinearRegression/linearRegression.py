

# 1. Design model (input, output size, forward pass)
# 2. Construct loss and optimizer
# 3. Training loop
#   - forward pass: compute prediction
#   - backward pass: gradients
#   - update weights


import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


# 0) prepare data
X, Y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=2)

X = torch.from_numpy(X.astype(np.float32))
y = torch.from_numpy(Y.astype(np.float32))
y = y.view(y.shape[0], 1)

n_samples, n_features = X.shape

# 1) model
input_size = n_features
output_size = 1


# simple lInear regression model
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        # define layers
        self.ln = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.ln(x)

model = LinearRegression(input_size, output_size)


# 2) loss and optimizer
learning_rate = 0.1
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


###############
# Training
##############
num_epochs = 100
for epoch in range(num_epochs):
    # forward pass and loss
    y_pred = model(X)

    loss = criterion(y_pred, y)

    # gradients = backward pass
    loss.backward() # dloss/dw

    # update weights and bias
    optimizer.step()

    # zero gradients
    optimizer.zero_grad()
    
    if (epoch + 1) % 10 == 0:
        print(f'epoch: {epoch + 1}, loss= {loss.item():.4f}')


# plot
# detach from graph and convert tenor to numpy array
predicted = model(X).detach().numpy()
plt.plot(X, y, 'ro')
plt.plot(X, predicted, 'b')
plt.show()