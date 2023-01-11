import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import os
import numpy as np

# download data from here: https://download.pytorch.org/tutorial/hymenoptera_data.zip

# Data augmentationand normalization for training
####
data_transforms = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    "val": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }



data_dir = "data/hymenoptera_data"
# Create a dictionary that contains the information of the images in both the training and validation set
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
# Create a dictionary that contains the data loader
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True) for x in ['train', 'val']}
# Create a dictionary that contains the size of each dataset (training and validation)
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
# Get the lass names
class_names = image_datasets['train'].classes
# Print out the results
print("Class Names: {}".format(class_names))
print("There are {} batches in the training set".format(len(dataloaders['train'])))
print("There are {} batches in the test set".format(len(dataloaders['val'])))
print("There are {} training images ".format(dataset_sizes["train"]))
print("There are {} testing images".format(dataset_sizes['val']))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the resnet
model_conv = torchvision.models.resnet18(pretrained=True)

# Freeze all layers in the network
for param in model_conv.parameters():
    param.requires_grad = False
    
# Get the number of inputs of the last layer (or number of neurons in the layer preceding the last layer)
num_ftrs = model_conv.fc.in_features
# Reconstruct the last layer (output layer) to have only two classes
model_conv.fc = nn.Linear(num_ftrs, 2)

if torch.cuda.is_available():
    model_conv = model_conv.cuda()
    
# Understand what's happening
iteration = 0
correct = 0
for inputs, labels in dataloaders['train']:
    if iteration == 1:
        break
    inputs = Variable(inputs)
    labels = Variable(labels)
    if torch.cuda.is_available():
        inputs = inputs.cuda()
        labels = labels.cuda()
    print("For one iteration, this is what happens:")
    print("Input Shape: ", inputs.shape)
    print("Labels Shape: ",labels.shape)
    print("Labels are: {}".format(labels))
    output = model_conv(inputs)
    print("Output Tensor: ", output)
    print("Outputs Shape", output.shape)
    _, predicted = torch.max(output, 1)
    print("Predicted: ", predicted)
    print("Predicted Shape", predicted.shape)
    correct += (predicted == labels).sum()
    print("Correct predictions: ", correct)
    
    iteration += 1



criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_conv.fc.parameters(), lr=0.01, momentum=0.9)
# Try experimenting with: optim.Adam(model_conv.fc.parameters(), lr=0.01)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=0.7, gamma=0.1)

# This is to demonstrate what happens in the backgroud of scheduler.step()

# def lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7):
#     """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
#     lr = init_lr * (0.1 **(epoch // lr_decay_epoch))
    
#     if epoch % lr_decay_epoch == 0:
#         print("LR is set to {}".format(lr))
    
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
        
#     return optimizer




# Train the model
num_epochs = 25
for epoch in range(num_epochs):
    exp_lr_scheduler.step()
    # Reset the correct to 0 after passing through all the dataset
    correct = 0
    for images, labels in dataloaders['train']:
        images = Variable(images)
        labels = Variable(labels)
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
            
        optimizer.zero_grad()
        outputs = model_conv(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum()
        
    train_acc = 100 * correct / dataset_sizes['train']
    print("Epoch [{}/{}], Loss: {:.4f}, Train Accuracy: {}%".format(epoch+1, num_epochs, loss.item(), train_acc))
    

# Test the model
model_conv.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in dataloaders['val']:
        images = Variable(images)
        labels = Variable(labels)
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
            
        optimizer.zero_grad()
        outputs = model_conv(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum()
        
    test_acc = 100 * correct / dataset_sizes['val']
    print("Test Accuracy: {}%".format(test_acc))
            
# Visualize some predictions
import matplotlib.pyplot as plt
fig = plt.figure()
shown_batch = 0
index = 0

with torch.no_grad():
    for (images, labels) in dataloaders['val']:
        if shown_batch == 1:
            break
        shown_batch += 1
        images = Variable(images)
        labels = Variable(labels)
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
            
        outputs = model_conv(images)
        _, preds = torch.max(outputs, 1)
        
        for i in range(4):
            index += 1
            ax = plt.subplot(2, 2, index)
            ax.axis("off")
            ax.set_title("Predicted Label: {}".format(class_names[preds[i]]))
            input_img = images.cpu().data[i]
            inp = input_img.numpy().transpose((1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array( [0.229, 0.224, 0.225])
            inp = std * inp + mean
            inp = np.clip(inp, 0, 1)
            plt.imshow(inp)