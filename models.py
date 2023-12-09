import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy
import torch
import numpy
from torch.optim.lr_scheduler import ReduceLROnPlateau


class AslLitModel(pl.LightningModule):
    def __init__(self, input_shape, num_classes, learning_rate=3e-4):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.conv1 = nn.Conv3d(1, 32, 3, 1)
        self.conv2 = nn.Conv3d(32, 32, 3, 1)
        self.conv3 = nn.Conv3d(32, 64, 3, 1)
        self.conv4 = nn.Conv3d(64, 64, 3, 1)

        self.pool1 = nn.MaxPool3d(3)
        self.pool2 = nn.MaxPool3d(3)

        n_sizes=self._get_output_shape(input_shape)

        self.fc1 = nn.Linear(n_sizes, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, self.num_classes)

        self.accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)

    def _get_output_shape(self, shape):

        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output_feat = self._feature_extractor(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    def _feature_extractor(self, x):

        x = F.relu(self.conv1(x))
        x = self.pool1(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool2(F.relu(self.conv4(x)))
        return x

    def forward(self, x):

       x = self._feature_extractor(x)
       x = x.view(x.size(0), -1)
       x = F.relu(self.fc1(x))
       x = F.relu(self.fc2(x))
       x = F.log_softmax(self.fc3(x), dim=1)
       return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        # metric
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class AslLitModel2(pl.LightningModule):
    def __init__(self, input_shape, num_classes, learning_rate=3e-4):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.num_classes = num_classes

        self.conv1 = nn.Conv3d(1, 64, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.max_pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.batch_norm1 = nn.BatchNorm3d(64)

        self.conv2 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.max_pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.batch_norm2 = nn.BatchNorm3d(64)

        self.conv3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.max_pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.batch_norm3 = nn.BatchNorm3d(128)

        self.conv4 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.max_pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.batch_norm4 = nn.BatchNorm3d(256)

        self.global_pool = nn.AdaptiveMaxPool3d(1)
        self.fc1 = nn.Linear(256, 512)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, num_classes)

        #n_sizes = self._get_conv_output(input_shape)

        self.accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)

    def _get_conv_output(self, shape):
        with torch.no_grad():
            input = torch.autograd.Variable(torch.rand(1, *shape))
            output = self._feature_extractor(input)
            return int(numpy.prod(output.size()))

    def _feature_extractor(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        return x

    def forward(self, x):
        x = F.relu(self.batch_norm1(self.max_pool1(self.conv1(x))))
        x = F.relu(self.batch_norm2(self.max_pool2(self.conv2(x))))
        x = F.relu(self.batch_norm3(self.max_pool3(self.conv3(x))))
        x = F.relu(self.batch_norm4(self.max_pool4(self.conv4(x))))

        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}


class AslLitModel3(pl.LightningModule):
    def __init__(self, input_shape, num_classes, learning_rate=3e-4):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.num_classes = num_classes

        self.conv1 = nn.Conv3d(1, 64, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.max_pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.batch_norm1 = nn.BatchNorm3d(64)

        self.conv2 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.max_pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.batch_norm2 = nn.BatchNorm3d(64)

        self.conv3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.max_pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.batch_norm3 = nn.BatchNorm3d(128)

        self.conv4 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.max_pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.batch_norm4 = nn.BatchNorm3d(256)

        self.global_pool = nn.AdaptiveMaxPool3d(1)
        self.fc1 = nn.Linear(256, 512)
        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, num_classes)

        #n_sizes = self._get_conv_output(input_shape)

        self.accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)

    def _get_conv_output(self, shape):
        with torch.no_grad():
            input = torch.autograd.Variable(torch.rand(1, *shape))
            output = self._feature_extractor(input)
            return int(numpy.prod(output.size()))

    def _feature_extractor(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        return x

    def forward(self, x):
        x = F.relu(self.batch_norm1(self.max_pool1(self.conv1(x))))
        x = self.dropout1(x)
        x = F.relu(self.batch_norm2(self.max_pool2(self.conv2(x))))
        x = F.relu(self.batch_norm3(self.max_pool3(self.conv3(x))))
        x = F.relu(self.batch_norm4(self.max_pool4(self.conv4(x))))

        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = torch.sigmoid(self.fc2(x))
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
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}


class AslLitModel4(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.num_classes = num_classes

        # Define the CNN layers and fully connected layers according to the architecture
        self.layer1 = nn.Sequential(nn.Conv3d(1, 32, kernel_size=3, stride=2, padding=1), nn.BatchNorm3d(32))
        self.layer2 = nn.Sequential(nn.Conv3d(32, 64, kernel_size=3, padding=1), nn.MaxPool3d(2), nn.BatchNorm3d(64))
        self.layer3 = nn.Sequential(nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1), nn.MaxPool3d(2),
                                    nn.BatchNorm3d(128))
        self.layer4 = nn.Sequential(nn.Conv3d(128, 256, kernel_size=3, padding=1), nn.MaxPool3d(2), nn.BatchNorm3d(256))
        self.layer5 = nn.Sequential(nn.Conv3d(256, 256, kernel_size=3, stride=2, padding=1), nn.BatchNorm3d(256))
        self.layer6 = nn.Sequential(nn.Conv3d(256, 128, kernel_size=3, padding=1),
                                    nn.Upsample(scale_factor=2, mode='nearest'), nn.BatchNorm3d(128))
        self.layer7 = nn.Sequential(nn.Conv3d(128, 64, kernel_size=3, stride=2, padding=1),
                                    nn.Upsample(scale_factor=2, mode='nearest'), nn.BatchNorm3d(64))
        self.layer8 = nn.Conv3d(64, 32, kernel_size=3, padding=1)

        # Assuming the input size to the first FC layer is determined and is 'fc_input_size'
        fc_input_size = 256  # This needs to be calculated based on the input volume size
        self.fc1 = nn.Linear(fc_input_size, 256)
        self.fc2 = nn.Linear(256, self.num_classes)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.3)

        # Metrics
        self.accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.learning_rate = learning_rate

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)

        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return F.log_softmax(x, dim=1)

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


class AslLitModel5(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.num_classes = num_classes

        # Define the CNN layers and fully connected layers according to the architecture
        self.layer1 = nn.Sequential(nn.Conv3d(1, 64, 3, stride=1, padding=1),
                                    nn.MaxPool3d(2),
                                    nn.BatchNorm3d(64))

        self.layer2 = nn.Sequential(nn.Conv3d(64, 64, 3, stride=1, padding=1),
                                    nn.MaxPool3d(2),
                                    nn.BatchNorm3d(64))

        self.layer3 = nn.Sequential(nn.Conv3d(64, 128, 3, stride=1, padding=1),
                                    nn.MaxPool3d(2),
                                    nn.BatchNorm3d(128))

        self.layer4 = nn.Sequential(nn.Conv3d(128, 256, 3, stride=1, padding=1),
                                    nn.MaxPool3d(2),
                                    nn.BatchNorm3d(256))

        self.global_pool = nn.AdaptiveMaxPool3d(1)
        self.fc1 = nn.Linear(256, 128)
        self.drop = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, self.num_classes)

        self.accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.learning_rate = learning_rate

    def forward(self, x):
        x = self.layer1(x)
        x = self.drop(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

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
