import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
import os
import numpy as np
from torch.utils.data import Dataset
import torch


class MatrixFolderDataset(Dataset):
    """
    Standard lightning dataset for sorted numpy arrays
    """
    def __init__(self, root_dir, transform=None):
        """
        Initialization of numpy dataset created with buffer.py
        :param root_dir: Path to folder with sorted numpy arrays
        :param transform: Default none for now
        """

        self.root_dir = root_dir
        self.transform = transform
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

    class NpyFolderDataset(Dataset):
        """
        Standard lightning dataset for sorted numpy arrays
        """

    def __init__(self, root_dir, transform=None):
            """
            Initialization of numpy dataset created with buffer.py
            :param root_dir: Path to folder with sorted numpy arrays
            :param transform: Default none for now
            """

            self.root_dir = root_dir
            self.transform = transform
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
            """
            Returns length of dataset
            :return:
            """
            return len(self.npy_files)

    def __getitem__(self, idx):
            """
            Getitem function loads numpy data and expands dimensions for torch libraries
            :param idx: idx from init
            :return: Sample and lable
            """
            npy_path = self.npy_files[idx]
            label = self.labels[idx]
            sample = np.load(npy_path)
            sample = sample.astype(np.float32)

            sample = np.transpose(sample, (2, 0, 1))

            if self.transform:
                sample = self.transform(sample)

            return sample, label

class AslMatrixDataModule(pl.LightningDataModule):
    def __init__(self, input_dim, num_classes, num_of_workers, data_dir=r"E:\asl-signs\matrix", batch_size: int = 32, val_split: float = 0.2):

        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.dims = input_dim
        self.num_classes = num_classes
        self.num_of_workers = num_of_workers
        self.val_split = val_split


    def setup(self, stage: str=None):
        if stage == 'fit' or stage is None:
            full_dataset = MatrixFolderDataset(root_dir=self.data_dir)
            print(f"Full dataset size: {len(full_dataset)}")
            val_size = int(len(full_dataset) * 0.2)
            train_size = int(len(full_dataset) * 0.7)
            test_size = len(full_dataset) - train_size - val_size
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
            print(f"Train dataset size: {len(self.train_dataset)}")  # Debug print
            print(f"Validation dataset size: {len(self.val_dataset)}")  # Debug print
            print(f"Test dataset size: {len(self.test_dataset)}")  # Debug print

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_of_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_of_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_of_workers, shuffle=False)
