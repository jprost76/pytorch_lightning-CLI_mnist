from pytorch_lightning.callbacks import Callback

class MyPrintingCallback(Callback):

    def __init__(self, p : int) -> None:
        super().__init__()
        self.p = p

    def on_train_start(self, trainer, pl_module):
        print("Training is starting")

    def on_train_end(self, trainer, pl_module):
        print("Training is ending")