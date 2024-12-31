# minimum pytorch lightning example

fn_data = 'iris_data.csv'
max_epochs = 10
lr = 0.01
batch_size = 2
cpu_workers = 10
val_frac = 0.2

# --------------------------------------------
import lightning.pytorch as L
# -------------------------------------------------

from pl_data import theDataModule
from pl_model import IrisClassifier  

# -------------------------------------------------

from model import Net

def main():
    dm = theDataModule(fn_data, batch_size, cpu_workers, val_frac)
    dm.setup(stage="fit")
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    model = IrisClassifier(Net(), lr)
    trainer = L.Trainer(max_epochs=max_epochs, devices="auto", accelerator="auto")
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()