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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# 0) prepare data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

n_samples, n_features = X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# scaling dataset
# with a zero mean and one variance
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

# reshape y to valid dims (N, 1)
y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)


# 1) model
# input -> model: w*x + b -> sigmoid(out)
class LogisticRegression(nn.Module):
    def __init__(self, input_dim) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))


model = LogisticRegression(n_features)

# 2) loss and optimizer
learning_rate = 0.01
criterion = nn.BCELoss() # binary classification
optim = torch.optim.SGD(model.parameters(), lr=learning_rate)

###########
# Training
##########
num_epochs = 100
for epoch in range(num_epochs):
    # forward pass and loss
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)

    # backward pass : compute gradient
    loss.backward()

    # update weights
    optim.step()

    # zero gradients
    optim.zero_grad()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch: {epoch+1}, loss= {loss.item():.4f}")


# Evaluate model
# It should not be with gradient and check its accuracy
with torch.no_grad():
    y_predicted = model(X_test) 
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f"Accuracy = {acc:.4f}")