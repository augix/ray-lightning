# minimum pytorch lightning example
from config import config

# --------------------------------------------
import lightning.pytorch as L
# -------------------------------------------------

from pl_data import theDataModule
from pl_model import IrisClassifier  

# -------------------------------------------------

from model import Net

def main():
    dm = theDataModule(config.fn_data, config.batch_size, config.cpu_workers, config.val_frac)
    dm.setup(stage="fit")
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    model = IrisClassifier(Net(), config.lr)
    trainer = L.Trainer(max_epochs=config.max_epochs, devices="auto", accelerator="auto")
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()