import torch
from torch import nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchvision import models
from torchmetrics import Accuracy
import pytorch_lightning as pl


class ImageClassifier(LightningModule):
    def __init__(self, num_classes: int = 250, learning_rate: float = 0.001):
        super().__init__()

        # Use pretrained model for robustness
        self.backbone = models.resnet50(pretrained=True)
        num_ftrs = self.backbone.fc.in_features
        self.num_classes = num_classes

        # Remove the final layer of the model
        layers = list(self.backbone.children())[:-1]
        self.backbone = nn.Sequential(*layers)

        # Replace it with a new linear layer (classification head)
        self.classifier = nn.Linear(num_ftrs, num_classes)
        self.learning_rate = learning_rate
        self.accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)


    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, start_dim=1)  # Flatten features vector
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

class BasicBlock(nn.Module):
    """
    Basic Block of ResNet
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # First convolution block of the Basic Block
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second convolution block of the Basic Block
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(pl.LightningModule):
    """
    ResNet class for constructing the model
    """
    def __init__(self, block, num_blocks, num_classes=250, learning_rate=0.001):
        super().__init__()
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.in_channels = 64


        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(3072, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        self.log('train_acc', self.accuracy(y_hat, y), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.accuracy(y_hat, y), prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
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

def ResNet50(num_classes=250, learning_rate=0.001):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, learning_rate)