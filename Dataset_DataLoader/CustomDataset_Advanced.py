# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 19:59:50 2022

@author: susha
"""

# Download the dataset from here: https://download.pytorch.org/tutorial/faces.zip 

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode


# load landmarks
landmarks_frame = pd.read_csv('../../data/faces/face_landmarks.csv')

n = 65
img_name = landmarks_frame.iloc[n, 0]
landmarks = landmarks_frame.iloc[n, 1:]
landmarks = np.asarray(landmarks)
landmarks = landmarks.astype('float').reshape(-1, 2)

# Visualize on image
print('Image name: {}'.format(img_name))
print('Landmarks shape: {}'.format(landmarks.shape))
print('First 4 Landmarks: {}'.format(landmarks[:4]))


def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001) # pause a bit so that plots are updated

plt.figure()
show_landmarks(io.imread(os.path.join('../../data/faces', img_name)), landmarks)
plt.show()


# Dataset class
class FaceLandmarksDataset(Dataset):
    """Face landmarks Dataset."""
    def __init__(self, csv_file, root_dir, transform=None):
        """

        Parameters
        ----------
        csv_file : string
            Path to the csv file with annotations.
        root_dir : string
            Directory with all the images.
        transform : callable, optional
            Optional transform to be applied on a sample. The default is None.

        Returns
        -------
        None.

        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform 
        
    def __len__(self):
        return len(self.landmarks_frame)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[index, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[index, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image':image, 'landmarks':landmarks}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


face_dataset = FaceLandmarksDataset(csv_file='../../data/faces/face_landmarks.csv', root_dir='../../data/faces')

fig = plt.figure()

for i in range(len(face_dataset)):
    sample = face_dataset[i]
    
    print(i, sample['image'].shape, sample['landmarks'].shape)
    
    ax = plt.subplot(1, 4, i+1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    show_landmarks(**sample)

    if i == 3:
        plt.show()
        break


# Issue is: they are not of same size. How to make fixed image size?
# Make a preprocess function that have three transforms: Rescale, RandomCrop, ToTensor

# Write them as callable class

class Rescale(object):
    """
    Rescale the image in a sample to a given size.
    
    """
    def __init__(self, output_size):
        """    
        Parameters
        ----------
        output_size : tuple or int
            Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
        """
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        
        h, w = image. shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h/w , self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
                
        else:
            new_h, new_w = self.output_size
            
        new_h, new_w = int(new_h), int(new_w)
        
        img = transform.resize(image, (new_h, new_w))
        
        # h and w are swapped for landmarks because for images, 
        # x and y axes are axis 1 and 0 respectively
        
        landmarks = landmarks * [new_w / w, new_h / h]
        
        return {'image': img, 'landmarks': landmarks}
    
class RandomCrop(object):
    "Crop randomly the image in a sample"
    
    def __init__(self, output_size):
        """

        Parameters
        ----------
        output_size : (tuple or int)
            Desired output size. If int,  square crop is made.  

        Returns
        -------
        None.

        """
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        
        image = image[top: top + new_h,
                      left: left + new_w]
        
        landmarks = landmarks - [left, top]
        
        return {'image':image, 'landmarks':landmarks}
    

class ToTensor(object):
    "Convert ndarrrays in sample to Tensors."
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return {'image':torch.from_numpy(image),
                'landmarks':torch.from_numpy(landmarks)}
    

scale = Rescale(256)
crop = RandomCrop(128)
composed = transforms.Compose([Rescale(256),
                               RandomCrop(224)])

# Apply each of the above transforms on sample.
# fig = plt.figure()
# sample = face_dataset[65]
# for i, tsfrm in enumerate([scale, crop, composed]):
#     transformed_sample = tsfrm(sample)

#     ax = plt.subplot(1, 3, i + 1)
#     plt.tight_layout()
#     ax.set_title(type(tsfrm).__name__)
#     show_landmarks(**transformed_sample)

# plt.show()

# Iterating through dataset
transformed_dataset = FaceLandmarksDataset(csv_file='../../data/faces/face_landmarks.csv',
                                           root_dir='../../data/faces',
                                           transform=transforms.Compose([
                                               Rescale(256),
                                               RandomCrop(224),
                                               ToTensor()
                                               ]))
for i in range(len(transformed_dataset)):
    sample = transformed_dataset[i]
    print(i, sample['image'].size(), sample['landmarks'].size())
    if i == 3: 
        break
    
# Loading Data
dataloader = DataLoader(transformed_dataset, batch_size=4, shuffle=True, num_workers=0)

# show in a batch
def show_landmarks_batch(sample_batched):
    "Show image with landmarks for a batch of samples."
    images_batch, landmarks_batch = sample_batched['image'], sample_batched['landmarks']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2
    
    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    
    for i in range(batch_size):
        plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size + (i+1) * grid_border_size, 
                    landmarks_batch[i, :, 1].numpy() + grid_border_size, 
                    s=10, marker='.', c='r')
        
        plt.title('Batch from dataloader')
        
# if __name__ == '__main__':
for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size(),
          sample_batched['landmarks'].size())

    # observe 4th batch and stop.
    if i_batch == 3:
        plt.figure()
        show_landmarks_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break
    
    



















