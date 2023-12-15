import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy
import torch
from torchvision.models import densenet121
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from timm import create_model


class AslLitModelMatrix(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.num_classes = num_classes

        # Define the CNN layers and fully connected layers according to the architecture
        self.layer1 = nn.Sequential(nn.Conv2d(3, 64, 3, stride=1, padding=1),
                                    nn.MaxPool2d(2),
                                    nn.BatchNorm2d(64))

        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, 3, stride=1, padding=1),
                                    nn.MaxPool2d(2),
                                    nn.BatchNorm2d(64))

        self.layer3 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=1, padding=1),
                                    nn.MaxPool2d(2),
                                    nn.BatchNorm2d(128))

        self.layer4 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=1, padding=1),
                                    nn.MaxPool2d(2),
                                    nn.BatchNorm2d(256))

        self.global_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Linear(4608, 1024)
        self.drop = nn.Dropout(0.1)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, self.num_classes)

        self.accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.learning_rate = learning_rate

    def forward(self, x):
        x = self.layer1(x)
        x = self.drop(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.relu(x)

        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        self.log('train_acc', self.accuracy(y_hat, y), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.accuracy(y_hat, y), prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', self.accuracy(y_hat, y), prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-3)

        def lambda_epoch(epoch):
            if epoch < 10:
                return 1.0
            elif 10 <= epoch < 20:
                return 0.5
            else:
                return 0.1

        scheduler_lr = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch)
        scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler_lr, 'interval': 'epoch'},
            'lr_scheduler': {'scheduler': scheduler_plateau, 'interval': 'epoch', 'monitor': 'val_loss'},
            'gradient_clip_val': 0.5,
        }


class AslLitModelMatrix2(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.num_classes = num_classes

        # Define the CNN layers and fully connected layers according to the architecture
        self.layer1 = nn.Sequential(nn.Conv2d(3, 64, 3, stride=1, padding=1),
                                    nn.MaxPool2d(2),
                                    nn.BatchNorm2d(64))

        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, 3, stride=1, padding=1),
                                    nn.MaxPool2d(2),
                                    nn.BatchNorm2d(64))

        self.layer3 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=1, padding=1),
                                    nn.MaxPool2d(2),
                                    nn.BatchNorm2d(128))

        self.layer4 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=1, padding=1),
                                    nn.MaxPool2d(2),
                                    nn.BatchNorm2d(256))

        self.global_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Linear(4608, 1024)
        self.drop = nn.Dropout(0.6)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, self.num_classes)

        self.accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.learning_rate = learning_rate

    def forward(self, x):
        x = self.layer1(x)
        x = self.drop(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.relu(x)

        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        self.log('train_acc', self.accuracy(y_hat, y), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.accuracy(y_hat, y), prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', self.accuracy(y_hat, y), prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-3)

        def lambda_epoch(epoch):
            if epoch < 10:
                return 1.0
            elif 10 <= epoch < 20:
                return 0.5
            else:
                return 0.1

        scheduler_lr = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch)
        scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler_lr, 'interval': 'epoch'},
            'lr_scheduler': {'scheduler': scheduler_plateau, 'interval': 'epoch', 'monitor': 'val_loss'},
            'gradient_clip_val': 0.5,
        }


