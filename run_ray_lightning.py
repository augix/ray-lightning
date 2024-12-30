import os

import torch
import lightning.pytorch as pl

import ray.train.lightning
from ray.train.torch import TorchTrainer

# Model, Loss, Optimizer
from model import Net
class IrisClassifier(pl.LightningModule):
    def __init__(self):
        super(IrisClassifier, self).__init__()
        self.model = Net()
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.forward(x)
        loss = self.criterion(outputs, y)
        self.log("loss", loss, on_step=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.001)

wd = os.getcwd()
def train_func():
    os.chdir(wd)
    # Data
    fn_iris = 'iris_data.csv'
    from run_lightning import theDataModule
    dm = theDataModule(fn_iris)
    dm.setup(stage='fit')
    train_loader = dm.train_dataloader()

    # Training
    model = IrisClassifier()
    # [1] Configure PyTorch Lightning Trainer.
    trainer = pl.Trainer(
        max_epochs=10,
        devices="auto",
        accelerator="auto",
        strategy=ray.train.lightning.RayDDPStrategy(),
        plugins=[ray.train.lightning.RayLightningEnvironment()],
        # callbacks=[ray.train.lightning.RayTrainReportCallback()],
        # [1a] Optionally, disable the default checkpointing behavior
        # in favor of the `RayTrainReportCallback` above.
        # enable_checkpointing=False,
    )
    trainer = ray.train.lightning.prepare_trainer(trainer)
    trainer.fit(model, train_dataloaders=train_loader)

# [2] Configure scaling and resource requirements.
scaling_config = ray.train.ScalingConfig(num_workers=2, use_gpu=True)

# [3] Launch distributed training job.
trainer = TorchTrainer(
    train_func,
    scaling_config=scaling_config,
)
result: ray.train.Result = trainer.fit()
print(result)