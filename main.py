import torch
import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI


class MnistModel(pl.LightningModule):
    """
    .. warning::  This is meant for testing/debugging and is experimental.
    """

    def __init__(self, network: torch.nn.Module, learning_rate: float = 0.02):
        super().__init__()
        self.save_hyperparameters()
        self.net = network
        self.learning_rate = learning_rate
        self.loss = torch.nn.BCELoss()

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

    def training_step(self, batch: torch.Tensor, batch_nb: int) :
        x, y = batch
        target = torch.nn.functional.one_hot(y, num_classes=10).type(torch.float)
        pred = self(x)
        loss = self.loss(pred, target)
        self.log("loss", loss)
        return loss

    def validation_step(self,  batch: torch.Tensor, batch_nb: int):
        x, y = batch
        target = torch.nn.functional.one_hot(y, num_classes=10).type(torch.float)
        pred = self(x)
        loss = self.loss(pred, target)
        self.log("valid_loss", loss)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # lr_scheduler = {
            # 'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda t: (1 - t/10000)),
            # 'name': 'lr'
        # }
        return optimizer


def cli_main():
    cli = LightningCLI(MnistModel, run=True)


if __name__ == "__main__":
    cli_main()
