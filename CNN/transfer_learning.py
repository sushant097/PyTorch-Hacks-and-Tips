from cProfile import label
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms, models, datasets
import matplotlib.pyplot as plt
import time
import os 
import copy
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
}

data_dir = '../data/hymenoptera_data'

# Create dataset and dataloader for both Train and Validation
dataset_train = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train']) 
dataset_val = datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val']) 
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=4, shuffle=True, num_workers=0)
valid_loader = torch.utils.data.DataLoader(dataset_val, batch_size=4, num_workers=0)
class_names = dataset_train.classes

print(len(train_loader), len(valid_loader))
print(class_names)

def imshow(inp, title):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.title(title)
    plt.show()

def plot_data():
    # get a batch of training data
    inputs, classes = next(iter(train_loader))
    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[class_names[x] for x in classes])


# plot_data()

def train_evaluate_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-' * 10)

        # Each epoch has a 
        ##################
        #Training
        ##################
        model.train()
        running_loss = 0.0
        running_corrects = 0

        # Iterate over data
        for data, labels in train_loader:
            data = data.to(device)
            labels = labels.to(device)

            # forward
            # track history if only in train
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # backward + optimize 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * data.size(0)
            running_corrects += torch.sum(preds == labels.data)

        
        scheduler.step()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = running_corrects / len(train_loader)

        print(' Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))


        running_loss, running_corrects = 0, 0
        ##############
        ## Evaluation
        ##############
        model.eval()
        with torch.no_grad():
            for data, labels in valid_loader:
                data = data.to(device)
                labels = labels.to(device)

                outputs = model(data)

                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)


                running_loss += loss.item()* data.size(0)
                running_corrects += torch.sum(preds == labels.data)

            
            epoch_loss = running_loss / len(valid_loader)
            epoch_acc = running_corrects.double() / len(valid_loader)

            print(' Validation Loss: {:.4f} Accuracy: {:4f}'.format(epoch_loss, epoch_acc))

            if epoch_acc > best_acc:
                best_model_wts = copy.deepcopy(model.state_dict())
            
            print("===============================\n")

    
    time_elapsed = time.time() - since
    print('Training complete in {:0f}m {:0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:.4f}'.format(best_acc))

    # load best model weights
    model.load_state_dic(best_model_wts)
    return model


###################
## FineTuning
###################

# Load a pretrained model and reset final fully connnected layer.
model = models.resnet18(pretrained=True)
num_filters = model.fc.in_features
# output is number of classes is 2
model.fc = nn.Linear(num_filters, len(class_names))
# transfer model to the device
model = model.to(device)
# define criterion
criterion = nn.CrossEntropyLoss()

# USE SGD as optimizer on all the parameters are being optimized
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# StepLR Decays the learning rate of each parameter group by gamma every step_size epochs
# Decay LR by a factor of 0.1 every 7 epochs
# Learning rate scheduling should be applied after optimizer's update
# e.g
# for epoch in range(epochs):
#   train(...)
#   validate(...)
#   scheduler.step()

step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# trained model with best updated weights
model = train_evaluate_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=25)

# Pretrained resnet18 model
# Freeze the network except the final layer so that gradient are not computed in backward().
model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules
num_filters = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_filters, len(class_names))

# move model to the device
model_conv = model_conv.to(device)
criterion = nn.CrossEntropyLoss()

# Only parameters of final layer are being optimized as
optimizer_conv = torch.optimSGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

model_conv = train_evaluate_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=25)

