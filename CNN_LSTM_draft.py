import pytorch_lightning as pl
import wandb
import torch
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import StochasticWeightAveraging
from Datasets import ParquetFolderDataset, NpyFolderDataset, AslParquetDataModule, AslNpyDataModule
from models import AslLitModel, AslCnnRnnModel, AslCnnRnnModel2
from pytorch_lightning.tuner import Tuner


def main():
    selected_landmark_indices = [33, 133, 159, 263, 46, 70, 4, 454, 234, 10, 338, 297, 332, 61, 291, 0, 78, 14, 317,
                                 152, 155, 337, 299, 333, 69, 104, 68, 398]

    params = {"landmarks": selected_landmark_indices,
              "input_dim": (70, 70, 420),
              "model_input": (1, 70, 70, 420),
              "num_classes": 4,
              "hidden_dim": 128,
              "n_layers": 1,
              "num_of_workers": 4,
              "path": r"C:/Skoda_Digital/Materials/Documents_FJFI/SU2/asl-signs/tensors",
              "batch": 4,
              "val_split": 0.2,
              "learning_rate": 1e-4,
              "gradient_clip": 0.1,
              "epochs": 3,
              "precision": 16,
              "accumulated gradient batches": 8,
              "Stochastic Weight Averaging": 1e-2,
              "log_key": '2c38e408badbb21d5150f743a54db048c3e9b943'}

    wandb.login(key=params["log_key"])

    dm = AslNpyDataModule(params["input_dim"], params["num_classes"], params["num_of_workers"], params["path"], params["batch"], params["val_split"])
    dm.setup()

    model = AslCnnRnnModel2(params["hidden_dim"], dm.num_classes, params["n_layers"], params["learning_rate"])

    wandb_logger = WandbLogger(project='cnn_rnn', job_type='train')
    wandb_logger.experiment.config.update({
        "model_name": "CNN_RNN",
        "input_dim": params["input_dim"],
        "model_input": params["model_input"],
        "num_classes": params["num_classes"],
        "num_of_workers": params["num_of_workers"],
        "batch": params["batch"],
        "hidden_dim": params["hidden_dim"],
        "n_layers": params["n_layers"],
        "val_split": params["val_split"],
        "learning_rate": params["learning_rate"],
        "gradient_clip": params["gradient_clip"],
        "epochs": params["epochs"],
        "precision": params["precision"],
        "accumulated gradient batches": params["accumulated gradient batches"],
        "Stochastic Weight Averaging": params["Stochastic Weight Averaging"],
    })
    checkpoint_callback = pl.callbacks.ModelCheckpoint()
    trainer = pl.Trainer(logger=wandb_logger, max_epochs=params["epochs"],
                         callbacks=[StochasticWeightAveraging(swa_lrs=params["Stochastic Weight Averaging"])], precision=params["precision"], accumulate_grad_batches=params["accumulated gradient batches"]
                         #gradient_clip_val=params["gradient_clip"]
                         )

    # tuner = Tuner(trainer)
    # tuner.lr_find(model)

    trainer.fit(model, dm)
    trainer.test(dataloaders=dm.test_dataloader())
    wandb.finish()


if __name__ == '__main__':
    main()
