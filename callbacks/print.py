from pytorch_lightning.callbacks import Callback

class MyPrintingCallback(Callback):

    def __init__(self, p : int) -> None:
        super().__init__()
        self.p = p

    def on_train_start(self, trainer, pl_module):
        print("Training is starting")
        print(self.p)

    def on_train_end(self, trainer, pl_module):
        print("Training is ending")

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print("hello from custom callback")