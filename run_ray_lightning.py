# minimum ray + pytorch lightning example

fn_data = 'iris_data.csv'
max_epochs = 10
lr = 0.01
batch_size = 2
cpu_workers = 10
gpu_workers = 2
val_frac = 0.2

# -------------------------------------------------

import os
import lightning.pytorch as L
import ray.train.lightning
from ray.train.torch import TorchTrainer
# -------------------------------------------------

# Model, Loss, Optimizer
from model import Net
from run_lightning import IrisClassifier
# -------------------------------------------------

wd = os.getcwd()
def train_func():
    os.chdir(wd)
    # Data
    from run_lightning import theDataModule
    dm = theDataModule(fn_data, batch_size, cpu_workers, val_frac)
    dm.setup(stage='fit')
    train_loader = dm.train_dataloader()

    # Training
    model = IrisClassifier(Net())
    # [1] Configure PyTorch Lightning Trainer.
    trainer = L.Trainer(
        max_epochs=max_epochs,
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

# -------------------------------------------------

# [2] Configure scaling and resource requirements.
scaling_config = ray.train.ScalingConfig(num_workers=gpu_workers, use_gpu=True)

# [3] Launch distributed training job.
trainer = TorchTrainer(
    train_func,
    scaling_config=scaling_config,
)
result: ray.train.Result = trainer.fit()
print(result)