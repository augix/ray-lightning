# minimum ray + pytorch lightning example

from config import config

# -------------------------------------------------

import os
import lightning.pytorch as L
import ray.train.lightning
from ray.train.torch import TorchTrainer
# -------------------------------------------------

# Model, Loss, Optimizer
from model import Net
from pl_model import IrisClassifier
# -------------------------------------------------

wd = os.getcwd()
def train_func():
    os.chdir(wd)
    # Data
    from pl_data import theDataModule
    dm = theDataModule(config.fn_data, config.batch_size, config.cpu_workers, config.val_frac)
    dm.setup(stage='fit')
    train_loader = dm.train_dataloader()

    # Training
    model = IrisClassifier(Net(), config.lr)
    # [1] Configure PyTorch Lightning Trainer.
    trainer = L.Trainer(
        max_epochs=config.max_epochs,
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
scaling_config = ray.train.ScalingConfig(num_workers=config.gpu_workers, use_gpu=True)

# [3] Launch distributed training job.
trainer = TorchTrainer(
    train_func,
    scaling_config=scaling_config,
)
result: ray.train.Result = trainer.fit()
print(result)