from typing import Any, Dict, Optional, Tuple, List

import os
import subprocess
import torch
import torch.nn as nn
import json

import pytorch_lightning as pl
import torchvision.transforms as T

from pathlib import Path
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from torchmetrics import F1Score, Precision, Recall, ConfusionMatrix, MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import TQDMProgressBar
import torchvision
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import io
from PIL import Image
from datetime import datetime


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# for confusion matrix logging tensorboard
class IntHandler:
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        text = plt.matplotlib.text.Text(x0, y0, str(orig_handle))
        handlebox.add_artist(text)
        return text


class LitResnet(pl.LightningModule):

    def __init__(
        self,
        model_name='resnet18',
        num_classes=6,
        lr=0.05,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters()

        self.num_classes = num_classes

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, self.num_classes),
        )


        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task='multiclass', num_classes=self.num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=self.num_classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=self.num_classes)

        # some other metrics to be logged
        self.f1_score = F1Score(task="multiclass", num_classes=self.num_classes)
        self.precision_score = Precision(task="multiclass", average='macro', num_classes=self.num_classes)
        self.recall_score = Recall(task="multiclass", average='macro', num_classes=self.num_classes)

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.f1_score(preds, targets)
        self.precision_score(preds, targets)
        self.recall_score(preds, targets)
        self.log("val/loss", self.val_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=True, on_epoch=True, prog_bar=False)
        self.log("val/f1", self.val_acc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/precision", self.precision_score, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/recall", self.recall_score, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outs: List[Any]):

        "Optional - Logging Confusion matrix image in Tensorboard"
        ######
        # Reference logging confusion matrix in tensorboard: https://stackoverflow.com/a/73388839/7338066
        ######

        tb = self.logger.experiment  # noqa

        outputs = torch.cat([tmp['preds'] for tmp in outs])
        labels = torch.cat([tmp['targets'] for tmp in outs])

        confusion = ConfusionMatrix(num_classes=self.num_classes).to(device)
        confusion(outputs, labels)
        computed_confusion = confusion.compute().detach().cpu().numpy().astype(int)

        # confusion matrix
        df_cm = pd.DataFrame(
            computed_confusion,
            index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            columns=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.subplots_adjust(left=0.05, right=.65)
        sn.set(font_scale=1.2)
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='d', ax=ax)
        ax.legend(
            index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            columns=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
            handler_map={int: IntHandler()},
            loc='upper left',
            bbox_to_anchor=(1.2, 1)
        )
        buf = io.BytesIO()

        plt.savefig(buf, format='jpeg', bbox_inches='tight')
        buf.seek(0)
        im = Image.open(buf)
        im = torchvision.transforms.ToTensor()(im)
        tb.add_image("val_confusion_matrix", im, global_step=self.current_epoch)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        return {"optimizer": optimizer}


class IntelClassificationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "dataset/",
        batch_size: int = 256 if torch.cuda.is_available() else 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_dir = data_dir

        ######################
        # data transformations
        ######################
        # References: https://pytorch.org/vision/stable/auto_examples/plot_transforms.html
        self.transforms = T.Compose([
            T.RandomRotation(degrees=66),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=(0.1,0.6), contrast=1,saturation=0, hue=0.4),
            T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            # T.Resize((224, 224)), # not needed for MNIST
            T.RandomCrop(size=(128, 128)),
            T.ToTensor(),
            T.Normalize((0.1307,), (0.3081,)),
        ])

        self.mnist_train: Optional[Dataset] = None
        self.mnist_valid: Optional[Dataset] = None
        self.mnist_test: Optional[Dataset] = None

    @property
    def num_classes(self):
        return len(self.data_train.classes)

    @property
    def classes(self):
        return self.data_train.classes

    def prepare_data(self):
        """Download data if needed.
        Do not use it to assign state (self.x = y).
        """
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.mnist_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.mnist_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.mnist_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


def train_and_evaluate(model, datamodule):
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs")

    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="auto",
        logger=[tb_logger],
        callbacks=[TQDMProgressBar(refresh_rate=20)]
    )

    trainer.fit(model, datamodule)
    # test on the test dataset
    # calculating evaluation metrics


    idx_to_class = {k: v for v,k in datamodule.data_train.class_to_idx.items()}
    model.idx_to_class = idx_to_class

    # calculating per class accuracy
    nb_classes = datamodule.num_classes

    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    acc_all = 0
    with torch.no_grad():
        for i, (images, targets) in enumerate(datamodule.test_dataloader()):
            # images = images.to(device)
            # targets = targets.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(targets.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    """
    Simple Logic may be useful:
    acc = [0 for c in list_of_classes]
    for c in list_of_classes:
        acc[c] = ((preds == labels) * (labels == c)).float() / (max(labels == c).sum(), 1))
    """

    acc_all = acc_all / len(datamodule.test_dataloader())

    accuracy_per_class = {
        idx_to_class[idx]: val.item() * 100 for idx, val in enumerate(confusion_matrix.diag() / confusion_matrix.sum(1))
    }
    print(accuracy_per_class)
    print(acc_all)

    report_dict = {
        "multiclass_classification_metrics": {
            "accuracy": acc_all,
            "accuracy_per_class": accuracy_per_class
        },
    }

    with open("evaluation_metrics.json", "w") as f:
        json.dump(report_dict, f)


def save_scripted_model(model):
    script = model.to_torchscript()

    # save for use in production environment
    torch.jit.save(script,"model.scripted.pt")


if __name__ == '__main__':

    datamodule = IntelClassificationDataModule(data_dir="../dataset", num_workers=2)
    datamodule.setup()

    model = LitResnet(model_name='resnet18', num_classes=datamodule.num_classes)
    # model = model.to(device)


    print(":: Training ...")
    train_and_evaluate(model, datamodule)

    print(":: Saving Scripted Model")
    save_scripted_model(model)
