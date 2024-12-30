# minimum pytorch lightning example

fn_data = 'iris_data.csv'

# --------------------------------------------
import torch as T
import pytorch_lightning as L
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.data import DataLoader
# -------------------------------------------------

from data import IrisDataset

class theDataModule(L.LightningDataModule):
  def __init__(self, fn_data):
    super().__init__()
    self.fn_data = fn_data
    self.batch_size = 2

  def setup(self, stage: str):
    ds_full = IrisDataset(self.fn_data)
    self.ds_train, self.ds_val = random_split(
      ds_full, [0.8, 0.2], generator=T.Generator().manual_seed(42)
    )

  def train_dataloader(self):
    return DataLoader(self.ds_train, batch_size=self.batch_size, num_workers=15, shuffle=True, drop_last=True)

  def val_dataloader(self):
    return DataLoader(self.ds_val, batch_size=self.batch_size, num_workers=15)

# -------------------------------------------------

from model import Net

class pl_model(L.LightningModule):
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
        return T.optim.Adam(self.model.parameters(), lr=0.001)


def main():
    dm = theDataModule(fn_data)
    dm.setup(stage="fit")
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    model = pl_model(Net())
    trainer = L.Trainer(max_epochs=3)
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()