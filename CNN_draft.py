from torchvision import datasets
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import random_split, DataLoader
from torchmetrics import Accuracy
from torchvision import transforms
import os
import numpy as np
import torch
from torch.utils.data import Dataset

class NpyFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the .npy files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

        # List all .npy files and their corresponding labels
        self.npy_files = []
        self.labels = []
        self.class_to_idx = {}
        for idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                self.class_to_idx[class_name] = idx
                for file in os.listdir(class_dir):
                    if file.endswith('.npy'):
                        self.npy_files.append(os.path.join(class_dir, file))
                        self.labels.append(idx)

    def __len__(self):
        return len(self.npy_files)

    def __getitem__(self, idx):
        npy_path = self.npy_files[idx]
        label = self.labels[idx]
        # Load .npy file
        sample = np.load(npy_path)
        sample = sample.astype(np.float32)
        sample = np.expand_dims(sample, axis=0)
        if self.transform:
            sample = self.transform(sample)

        return sample, label


class AslDataModule(pl.LightningDataModule):
    def __init__(self, data_dir=r"E:\asl-signs\tensors", batch_size: int = 32, val_split: float = 0.2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.dims = (420, 70, 70)
        self.num_classes = 250
        self.val_split = val_split

    #def prepare_data(self):
    # tady potenciálně parquet_proccess

    def setup(self, stage: str=None):
        if stage == 'fit' or stage is None:
            full_dataset = NpyFolderDataset(root_dir=self.data_dir)
            print(f"Full dataset size: {len(full_dataset)}")
            val_size = int(len(full_dataset))
            train_size = len(full_dataset) - val_size
            self.train_dataset, self.val_dataset = random_split(full_dataset, [2000, len(full_dataset)-2000])
            print(f"Train dataset size: {len(self.train_dataset)}")  # Debug print
            print(f"Validation dataset size: {len(self.val_dataset)}")  # Debug print

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)

class AslLitModel(pl.LightningModule):
    def __init__(self, input_shape, num_classes, learning_rate=3e-4):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate=learning_rate

        self.conv1 = nn.Conv3d(1, 32, 3, 1)
        self.conv2 = nn.Conv3d(32, 32, 3, 1)
        self.conv3 = nn.Conv3d(32, 64, 3, 1)
        self.conv4 = nn.Conv3d(64, 64, 3, 1)

        self.pool1 = nn.MaxPool3d(3)
        self.pool2 = nn.MaxPool3d(3)

        n_sizes=self._get_output_shape(input_shape)

        self.fc1 = nn.Linear(n_sizes,512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

        self.accuracy = Accuracy(task="multiclass", num_classes=250)

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


def main():
    path = r"E:\asl-signs\tensors"
    dm = AslDataModule()
    dm.setup()
    model = AslLitModel((1, 70, 70, 420), dm.num_classes)  # Make sure the input shape matches your data
    trainer = pl.Trainer(max_epochs=2)
    trainer.fit(model, dm)


if __name__ == '__main__':
    main()