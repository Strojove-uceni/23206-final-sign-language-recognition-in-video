import pytorch_lightning as pl
import torch.cuda

import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import StochasticWeightAveraging
from Datasets import ParquetFolderDataset, NpyFolderDataset, AslParquetDataModule, AslNpyDataModule
from models import AslLitModel, AslLitModel2, AslLitModel3
from pytorch_lightning.tuner import Tuner
from torch.optim.lr_scheduler import ReduceLROnPlateau


def main():
    torch.cuda.empty_cache()
    selected_landmark_indices = [33, 133, 159, 263, 46, 70, 4, 454, 234, 10, 338, 297, 332, 61, 291, 0, 78, 14, 317,
                                 152, 155, 337, 299, 333, 69, 104, 68, 398]

    params = {"landmarks": selected_landmark_indices,
              "input_dim": (25,25,70),
              "model_input": (25,25,70,100),
              "num_classes": 10,
              "num_of_workers": 4,
              "path": r"E:\asl-signs\tensors",
              "batch": 16,
              "val_split": 0.2,
              "learning_rate": 3e-4,
              "epochs": 6,
              "precision": 16,
              "accumulated gradient batches": 2,
              "Stochastic Weight Averaging": 1e-1,
              "AutoSet": False,
              "log_key": '6b4346cf1caf31e2c470b0f4b7e338da7bf74825'}

    wandb.login(key=params["log_key"])

    dm = AslNpyDataModule(params["input_dim"], params["num_classes"], params["num_of_workers"], params["path"], params["batch"], params["val_split"])
    dm.setup()

    model = AslLitModel2(params["model_input"], dm.num_classes, params["learning_rate"])

    wandb_logger = WandbLogger(project='ASL', job_type='train')
    wandb_logger.experiment.config.update({
        "model_name": "Draft_no_acc_local",
        "input_dim": params["input_dim"],
        "model_input": params["model_input"],
        "num_classes": params["num_classes"],
        "num_of_workers": params["num_of_workers"],
        "batch": params["batch"],
        "val_split": params["val_split"],
        "learning_rate": params["learning_rate"],
        "epochs": params["epochs"],
        "precision": params["precision"],
        "accumulated gradient batches": params["accumulated gradient batches"],
        "Stochastic Weight Averaging": params["Stochastic Weight Averaging"],
    })
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',  # Metric to monitor
        dirpath=r'C:\Users\drend\Desktop\SU2',  # Path where to save models
        filename='best-model',  # Filename for the checkpoint
        save_top_k=1,  # Save the top 1 models
        mode='min',  # Mode 'min' for loss and 'max' for accuracy

    )
    trainer = pl.Trainer(logger=wandb_logger, max_epochs=params["epochs"],
                         callbacks=[checkpoint_callback]#, precision=params["precision"], accumulate_grad_batches=params["accumulated gradient batches"],
                         )
    if params["AutoSet"]:
        tuner = Tuner(trainer)
        tuner.scale_batch_size(model)
        tuner.lr_find(model)

    trainer.fit(model, dm)
    trainer.test(dataloaders=dm.test_dataloader())
    wandb.finish()


if __name__ == '__main__':
    main()
