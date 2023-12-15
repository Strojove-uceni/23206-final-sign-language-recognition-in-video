import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy
import torch.optim as optim
import torch
import numpy


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class AslCnnRnnModel(pl.LightningModule):
    def __init__(self, input_shape, hidden_dim, num_classes, n_layers, learning_rate=3e-4):
        super(AslCnnRnnModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.scaler = torch.cuda.amp.GradScaler()
        # CNN 

        self.cnn = nn.Sequential(nn.Conv2d(3, 64, 3, stride=1, padding=1),
                                    nn.MaxPool2d(2),
                                    nn.BatchNorm2d(64),
                                    nn.Conv2d(64, 64, 3, stride=1, padding=1),
                                    nn.MaxPool2d(2),
                                    nn.BatchNorm2d(64),
                                    nn.Conv2d(64, 128, 3, stride=1, padding=1),
                                    nn.MaxPool2d(2),
                                    nn.BatchNorm2d(128),
                                    nn.Conv2d(128, 256, 3, stride=1, padding=1),
                                    nn.MaxPool2d(2),
                                    nn.BatchNorm2d(256))

        self.cnn.add_module('flatten', Flatten())
        # self.cnn.add_module('global_pool', nn.AdaptiveMaxPool2d(1))
        self.cnn.add_module('fc1', nn.Linear(4608, 1024))
        self.cnn.add_module('drop', nn.Dropout(0.4))
        self.cnn.add_module('fc2', nn.Linear(1024, 512))

        # RNN 
        self.rnn = nn.LSTM(input_size=512, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
        # Classification layer
        self.classifier = nn.Linear(self.hidden_dim, self.num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)


    def forward(self, x):
        # CNN
        cnn_out = self.cnn(x)
        r_in = cnn_out.unsqueeze(1)

        # RNNs
        rnn_out, _ = self.rnn(r_in)
        rnn_out2 = rnn_out[:, -1, :]

        # Classifier
        out = self.classifier(rnn_out2)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', self.accuracy(y_hat, y), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', self.accuracy(y_hat, y), prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', self.accuracy(y_hat, y), prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)


class AslCnnRnnModel2(pl.LightningModule):
    def __init__(self, input_shape, hidden_dim, num_classes, n_layers, learning_rate=3e-4):
        super(AslCnnRnnModel2, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        # CNN 
        self.cnn = nn.Sequential(
            nn.Conv3d(1, 32, 3, 1),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool3d(3),
            # nn.Conv3d(32, 64, 3, 1),
            # nn.ReLU(),
            # nn.Conv3d(64, 64, 3, 1),
            # nn.ReLU(),
            # nn.MaxPool3d(3),
        )
        self.n_sizes = self._get_output_shape(input_shape)

        # Additional layers
        self.cnn.add_module('flatten', Flatten())
        self.cnn.add_module('linear1', nn.Linear(self.n_sizes, 256))

        # LSTM part
        self.lstm = nn.LSTM(
            input_size=32, # This should match the output size of the CNN
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, num_classes)

        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)

    def _get_output_shape(self, shape):
        batch_size = 4
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output_feat = self.cnn(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    def _feature_extractor(self, x):
        conv1 = nn.Conv3d(1, 32, 3, 1)
        conv2 = nn.Conv3d(32, 32, 3, 1)
        # conv3 = nn.Conv3d(32, 64, 3, 1)
        # conv4 = nn.Conv3d(64, 64, 3, 1)

        pool1 = nn.MaxPool3d(3)
        # pool2 = nn.MaxPool3d(3)

        x = F.relu(conv1(x))
        x = pool1(F.relu(conv2(x)))
        # x = F.relu(conv3(x))
        # x = pool2(F.relu(conv4(x)))
        return x

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        # c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(x)
        r_in = c_out.view(batch_size, timesteps, -1)

        # RNN
        r_out, (h_n, h_c) = self.lstm(r_in)
        r_out2 = r_out[:, -1, :]

        # Classifier
        out = self.classifier(r_out2)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', self.accuracy(y_hat, y), prog_bar=True)
        # Gradient clipping
        # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        # for name, param in self.named_parameters():
        #     if param.grad is not None:
        #         print(f"Layer: {name}, Gradient norm: {param.grad.norm().item()}")
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', self.accuracy(y_hat, y), prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', self.accuracy(y_hat, y), prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.01)
