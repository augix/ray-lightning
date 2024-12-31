import torch as T
import torch.nn.functional as F
import lightning.pytorch as L

class IrisClassifier(L.LightningModule):
    def __init__(self, model, lr):
        super().__init__()
        self.model = model
        self.lr = lr
      
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
        return T.optim.Adam(self.model.parameters(), lr=self.lr)