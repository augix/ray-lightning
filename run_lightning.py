# minimum pytorch lightning example

fn_data = 'iris_data.csv'
max_epochs = 10
lr = 0.01
batch_size = 2
cpu_workers = 10
val_frac = 0.2

# --------------------------------------------
import torch as T
import lightning.pytorch as L
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.data import DataLoader
# -------------------------------------------------

from data import IrisDataset

class theDataModule(L.LightningDataModule):
  def __init__(self, fn_data, batch_size, cpu_workers, val_frac):
    super().__init__()
    self.fn_data = fn_data
    self.batch_size = batch_size
    self.cpu_workers = cpu_workers
    self.val_frac = val_frac

  def setup(self, stage: str):
    ds_full = IrisDataset(self.fn_data)
    self.ds_train, self.ds_val = random_split(
      ds_full, [1 - self.val_frac, self.val_frac], generator=T.Generator().manual_seed(42)
    )

  def train_dataloader(self):
    return DataLoader(self.ds_train, batch_size=self.batch_size, num_workers=self.cpu_workers, shuffle=True, drop_last=True)

  def val_dataloader(self):
    return DataLoader(self.ds_val, batch_size=self.batch_size, num_workers=self.cpu_workers)

# -------------------------------------------------

from model import Net

class IrisClassifier(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
      
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        # self.log("train/loss", loss, prog_bar=True, sync_dist=False)
        print(f"train/loss: {loss}")
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        return T.optim.Adam(self.model.parameters(), lr=lr)


def main():
    dm = theDataModule(fn_data, batch_size, cpu_workers, val_frac)
    dm.setup(stage="fit")
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    model = IrisClassifier(Net())
    trainer = L.Trainer(max_epochs=max_epochs, devices="auto", accelerator="auto")
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()