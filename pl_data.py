import torch as T
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import lightning.pytorch as L

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

