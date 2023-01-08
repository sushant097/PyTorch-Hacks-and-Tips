import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Hyper-parameters
latent_size = 20
hidden_size = 400
image_size = 784
num_epochs = 200
batch_size = 100
lr = 1e-3


# Create directory if not exists
sample_dir = "samples"
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)


transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]) # for grayscale
])

mnist = torchvision.datasets.MNIST(
    root='./data/',
    train=True,
    transform=transforms,
    download=True
)

dataloader = torch.utils.data.DataLoader(dataset=mnist,
                                         batch_size=batch_size,
                                         shuffle=True)


class Encoder(nn.Module):
    def __init__(self, input_dim=784, h_size=400, z_size=20):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, h_size)
        self.FC_mean = nn.Linear(h_size, z_size)
        self.FC_var = nn.Linear(h_size, z_size)

        self.leakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        h = self.leakyReLU(self.fc1(x))
        return self.FC_mean(h), self.FC_var(h)


class Decoder(nn.Module):
    def __init__(self, h_size, z_size, output_dim):
        super(Decoder, self).__init__()
        self.fc4 = nn.Linear(z_size, h_size)
        self.output = nn.Linear(h_size, output_dim)

        self.leakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        h = self.leakyReLU(self.fc4(x))
        x_hat = torch.sigmoid(self.output(h))
        return x_hat


class VAE(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(VAE, self).__init__()
        self.encoder = Encoder
        self.decoder = Decoder

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(device)  # sampling epsilon
        z = mean + var * epsilon # reparameterization trick
        return z

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var => var)
        x_hat = self.decoder(z)
        return x_hat, mean, log_var


# intantiate encoder, decoder
encoder = Encoder(input_dim=image_size, h_size=hidden_size, z_size=latent_size)
decoder = Decoder(hidden_size, latent_size, image_size)

model = VAE(encoder, decoder).to(device)

# define loss criterion and optimizer
bce_loss = nn.BCELoss()


def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD


optimizer = torch.optim.Adam(model.parameters(), lr=lr)

print("Start training VAE.....")
model.train()

for epoch in range(num_epochs):
    overall_loss = 0
    for batch_idx, (x, _) in enumerate(dataloader):
        x = x.reshape(batch_size, -1)
        x = x.to(device)

        # zero grad optimizer
        optimizer.zero_grad()

        x_hat, mean, log_var = model(x)
        loss = loss_function(x, x_hat, mean, log_var)

        overall_loss += loss.item()

        loss.backward()
        optimizer.step()

    print("Epoch: [{}/{}], Loss: {}".format(num_epochs, epoch+1, overall_loss/ (batch_size*batch_idx)))

print("DOne!")

# look: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/variational_autoencoder/main.py
# and : https://github.com/Jackson-Kang/Pytorch-VAE-tutorial/blob/master/01_Variational_AutoEncoder.ipynb
# learn more from https://pytorch.org/tutorials/