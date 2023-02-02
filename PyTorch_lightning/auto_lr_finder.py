

import torch
import pytorch_lightning as pl
import os

from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchmetrics import Accuracy
from torchvision.datasets import MNIST
from torchvision import datasets, transforms
from pytorch_lightning.callbacks.progress import TQDMProgressBar
import matplotlib.pyplot as plt

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64

class MNISTModel(pl.LightningModule):
    def __init__(self, lr=0.001):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.l1 = torch.nn.Linear(28 * 28, 10)
        self.lr = lr

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        return loss
    
    def test_step(self, batch):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = PATH_DATASETS):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        self.dims = (1, 28, 28)
        self.num_classes = 10

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=BATCH_SIZE)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=BATCH_SIZE)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=BATCH_SIZE)

# Init DataLoader from MNIST Dataset
# Init DataModule
datamodule = MNISTDataModule()
# Init model from datamodule's attributes
model = MNISTModel()

# Initialize a trainer
trainer = pl.Trainer(
                    enable_checkpointing=False,
                    auto_lr_find=False,
                    accelerator="auto",
                    max_epochs=5,
                )
lr_finder = trainer.tuner.lr_find(model, datamodule)

# set the suggested new lr
model.hparams.lr = lr_finder.suggestion()
        
print(f'Auto-find model LR is: {model.hparams.lr}')

# plot
fig = lr_finder.plot(suggest=True)
plt.show()
# Train with new lr
trainer = pl.Trainer(
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
    max_epochs=3,
    callbacks=[TQDMProgressBar(refresh_rate=20)],
)

# Train the model ⚡
print("Training ...\n")
trainer.fit(model, datamodule)

# save the model
torch.save(model.state_dict(), "model.pt")
print("Train Finished. Model Saved!")