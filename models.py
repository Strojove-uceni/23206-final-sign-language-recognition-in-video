import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy
import torch.optim as optim
import torch


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


class AsLModel2(pl.LightningModule):
    def __init__(self, input_shape, num_classes, learning_rate=3e-4):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.num_classes = num_classes

        self.conv1 = nn.Conv3d(1, 32, (3, 3, 3))
        nn.init.constant_(self.conv1.bias, 0.01)
        self.conv2 = nn.Conv3d(32, 32, (3, 3, 3))
        nn.init.constant_(self.conv2.bias, 0.01)
        self.conv3 = nn.Conv3d(32, 64, (3, 3, 3))
        self.conv4 = nn.Conv3d(64, 64, (2, 2, 2))

        self.pool = nn.MaxPool3d((2, 2, 2))
        self.dropout1 = nn.Dropout(0.6)
        self.dropout2 = nn.Dropout(0.7)
        self.dropout3 = nn.Dropout(0.5)

        n_sizes = self._get_conv_output(input_shape)

        self.fc1 = nn.Linear(n_sizes, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

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
        x = self._feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class AslCnnRnnModel(pl.LightningModule):
    def __init__(self, input_shape, hidden_dim, num_classes, n_outputs, learning_rate=3e-4):
        super(AslCnnRnnModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.n_outputs = n_outputs
        self.n_sizes = self._get_output_shape(input_shape)
        # CNN 
        self.cnn = nn.Sequential(
            nn.Conv3d(1, 32, 3, 1),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool3d(3),
            nn.Conv3d(32, 64, 3, 1),
            nn.ReLU(),
            nn.Conv3d(64, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool3d(3),
            nn.Linear(self.n_sizes, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
        )
        # RNN 
        self.rnn = nn.LSTM(input_size=128, hidden_size=hidden_dim, num_layers=num_classes, batch_first=True)
        # Classification layer
        self.classifier = nn.Linear(self.hidden_dim, self.n_outputs)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def _get_output_shape(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output_feat = self._feature_extractor(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    def _feature_extractor(self, x):
        conv1 = nn.Conv3d(1, 32, 3, 1)
        conv2 = nn.Conv3d(32, 32, 3, 1)
        conv3 = nn.Conv3d(32, 64, 3, 1)
        conv4 = nn.Conv3d(64, 64, 3, 1)

        pool1 = nn.MaxPool3d(3)
        pool2 = nn.MaxPool3d(3)

        x = F.relu(conv1(x))
        x = pool1(F.relu(conv2(x)))
        x = F.relu(conv3(x))
        x = pool2(F.relu(conv4(x)))
        return x

    def forward(self, x):
        # CNN
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, timesteps, -1)

        # RNN
        r_out, (h_n, h_c) = self.rnn(r_in)
        r_out2 = r_out[:, -1, :]

        # Classifier
        out = self.fc(r_out2)

        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y.view(-1, 1))
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y.view(-1, 1))
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.01)

