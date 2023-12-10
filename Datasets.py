import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
import os
import numpy as np
from torch.utils.data import Dataset
from processing import ParquetProcess
import torch.nn.utils.rnn as rnn_utils
import torch


class ParquetFolderDataset(Dataset):
    """
    Standard lighntning dataset for parquet data
    """
    def __init__(self, root_dir, landmarks, transform=None):
        """
        Initialization of dataset made from parquet data
        :param root_dir: Path to folder with sorted parquet data. Sorting done with ShitSort. Data are sorted into
         folders with class name
        :param landmarks: List of selected landmark indices (for detail look at mediapipe indices)
        :param transform: Default none for now
        """
        self.root_dir = root_dir
        self.transform = transform
        self.landmarks = landmarks
        self.parquet_files = []
        self.labels = []
        self.class_to_idx = {}

        for idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                self.class_to_idx[class_name] = idx
                for file in os.listdir(class_dir):
                    if file.endswith('.parquet'):
                        self.parquet_files.append(os.path.join(class_dir, file))
                        self.labels.append(idx)

    def __len__(self):
        """
        Returns length of dataset
        :return:
        """
        return len(self.parquet_files)

    def __getitem__(self, idx):
        """
        Getitem function, calculates tensor representation of parquet file using ParquetProcess from processing.py
        :param idx: Idx from init
        :return: Sample and lable from parquet dataset folder
        """
        parquet_path = self.parquet_files[idx]
        label = self.labels[idx]
        readed_data = ParquetProcess(parquet_path, self.landmarks, 140)
        sample = readed_data.tensor
        sample = sample.astype(np.float32)
        sample = np.expand_dims(sample, axis=0)
        if self.transform:
            sample = self.transform(sample)

        return sample, label


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
        sample = np.expand_dims(sample, axis=0)
        if self.transform:
            sample = self.transform(sample)

        return sample, label


class AslParquetDataModule(pl.LightningDataModule):
    """
    Standard lightning data module
    """
    def __init__(self, landmarks_idx, input_dim, num_classes, num_of_workers, data_dir=r"E:\asl-signs\parquets_sorted", batch_size: int = 2, val_split: float = 0.2):
        """
        Initialization of data module
        :param landmarks_idx: List of selected landmark indices used for tensor representation
        :param input_dim: dimensions of input data (for example (70,70,420)
        :param num_classes: Number of classes
        :param num_of_workers: Number of workers for training of different systems
        :param data_dir: Path to folder with sorted parquet files
        :param batch_size: Batch size, default 2 because of small VRAM, implementation of gradient accumulation
         recommended
        :param val_split: Ratio of data used for validation, default 0.2
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.dims = input_dim
        self.num_classes = num_classes
        self.num_of_workers = num_of_workers
        self.val_split = val_split
        self.landmarks = landmarks_idx

    def setup(self, stage: str=None):
        if stage == 'fit' or stage is None:
            full_dataset = ParquetFolderDataset(root_dir=self.data_dir, landmarks=self.landmarks)
            print(f"Full dataset size: {len(full_dataset)}")
            val_size = int(len(full_dataset) * 0.2)
            train_size = int(len(full_dataset) * 0.7)
            test_size = len(full_dataset) - train_size - val_size
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
            print(f"Train dataset size: {len(self.train_dataset)}")  # Debug print
            print(f"Validation dataset size: {len(self.val_dataset)}")  # Debug print
            print(f"Test dataset size: {len(self.test_dataset)}")  # Debug print

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_of_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_of_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_of_workers)


class AslNpyDataModule(pl.LightningDataModule):
    """
    Standard lightning data module
    """
    def __init__(self, input_dim, num_classes, num_of_workers, data_dir=r"E:\asl-signs\tensors", batch_size: int = 2, val_split: float = 0.2):
        """
        Initialization of data module
        :param input_dim: dimensions of input data (for example (70,70,420)
        :param num_classes: Number of classes
        :param num_of_workers: Number of workers for training of different systems
        :param data_dir: Path to folder with sorted parquet files
        :param batch_size: Batch size, default 2 because of small VRAM, implementation of gradient accumulation
         recommended
        :param val_split: Ratio of data used for validation, default 0.2
        """
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
            full_dataset = NpyFolderDataset(root_dir=self.data_dir)
            print(f"Full dataset size: {len(full_dataset)}")
            val_size = int(len(full_dataset) * 0.2)
            train_size = int(len(full_dataset) * 0.7)
            test_size = len(full_dataset) - train_size - val_size
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
            print(f"Train dataset size: {len(self.train_dataset)}")  # Debug print
            print(f"Validation dataset size: {len(self.val_dataset)}")  # Debug print
            print(f"Test dataset size: {len(self.test_dataset)}")  # Debug print

    def custom_collate_fn(self, batch):
        data, labels = zip(*batch)

        # Find the longest sequence
        max_len = max([s.shape[3] for s in data])  # Assuming shape [C, H, W, Frames]

        # Pad each sequence to match the longest one
        padded_data = [torch.nn.functional.pad(torch.from_numpy(s), (0, max_len - s.shape[3])) for s in data]
        # padded_videos = [torch.nn.functional.pad(torch.from_numpy(frame), (0, 0, 0, 0, 0, 0, 0, max_len - frame.shape[3])) for frame in data]

        data = torch.stack(padded_data)
        # data = torch.stack(padded_videos)
        labels = torch.tensor(labels)

        return data, labels

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_of_workers, collate_fn=self.custom_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_of_workers, collate_fn=self.custom_collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_of_workers, collate_fn=self.custom_collate_fn)


#Example usage
#    selected_landmark_indices = [33, 133, 159, 263, 46, 70, 4, 454, 234, 10, 338, 297, 332, 61, 291, 0, 78, 14, 317,
#                                 152, 155, 337, 299, 333, 69, 104, 68, 398]
#params={"landmarks": selected_landmark_indices,
#        "input_dim": (70,70,420),
#        "num_classes": 10
#        "num_of_workers": 8
#        "path": r"E:\asl-signs\parquets_sorted"
#        "batch": 4
#        "val_split": 0.2}
#data_module=AslParquetDataModule(params["path"])
#data_module.setup()
