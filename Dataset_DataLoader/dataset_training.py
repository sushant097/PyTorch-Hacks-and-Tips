from turtle import forward
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler


class WineDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        # data loading 
        xy = np.loadtxt('../data/wine.csv', delimiter=",", dtype=np.float32, skiprows=1)
        x = xy[:, 1:]
        x = x.reshape(-1, x.shape[1]).astype('float32')
        print(x.shape)
        y = xy[:, 0].reshape(-1, 1)
        print(y.shape)
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y) # (N, 1)
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples
    

#########
# MODEL
#########

n_classes = 3 # 3 wine categories
class MultiLabelClassification(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
    
        self.in_layer = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.out_layer = nn.Linear(64, n_classes)

    def forward(self,x):
        x = self.in_layer(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.out_layer(x)
        return F.log_softmax(x)


############
# DataLoader
############
batch_size = 4
dataset = WineDataset()
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


input_size, _ = iter(dataloader).next()
input_size = input_size.shape[1]
output_size = 1
print(input_size)
model = MultiLabelClassification(input_size, output_size)



#################
# Hyperparameters
#################
lr = 0.01
criterion = nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr=lr)


#########
# Training
############
num_epochs = 2
total_samples = len(dataset)
n_iterations = total_samples // batch_size
print(total_samples, n_iterations)

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        # forward + backward => update
        y_pred = model(inputs)
        print(y_pred)
        print(labels)
        loss = criterion(y_pred, labels)
        print(loss.item())
        loss.backward()

        optim.step()

        optim.zero_grad()


        if (i+1) % 5 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iterations}, Loss {loss.item()}')


# Evaluate model
# It should not be with gradient and check its accuracy
with torch.no_grad():
    y_predicted = model(X_test) 
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f"Accuracy = {acc:.4f}")