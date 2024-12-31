# minimum pytorch example
from config import config

# -------------------------------------------------
import numpy as np
import torch as T
device = T.device("cuda:0")  # use GPU
# -------------------------------------------------

from data import IrisDataset
from model import Net

# -------------------------------------------------

def main():
  # 0. get started
  print("\nBegin minimal PyTorch Iris demo ")  
  print("\nCreating IrisDataset for train data ")
  train_ds = IrisDataset(config.fn_data)
  train_ldr = T.utils.data.DataLoader(train_ds,
    batch_size=config.batch_size, shuffle=True)

# -------------------------------------------------

  # 2. create network
  print("\nCreating 4-7-3 neural network ")
  net = Net().to(device)    # or Sequential()

  # 3. train model
  loss_func = T.nn.CrossEntropyLoss()  # applies log-softmax()
  optimizer = T.optim.Adam(net.parameters(),
    lr=config.lr)

  print("\nStarting training")
  net = net.train()
  for epoch in range(0, config.max_epochs):
    loss_epoch = []
    for (batch_idx, batch) in enumerate(train_ldr):
      X = batch[0].to(device)
      Y = batch[1].to(device)
      optimizer.zero_grad()
      oupt = net(X)
      loss = loss_func(oupt, Y)  # a tensor
      loss.backward()
      optimizer.step()
      loss_epoch.append(loss.item())
    print(f"epoch: {epoch}, mean loss: {np.mean(loss_epoch)}")

    # TODO: monitor error
  print("Done training")

# -------------------------------------------------

  # 4. TODO: evaluate model accuracy

  # 5. use model to make a prediction
  net = net.eval()
  print("\nPredicting for [5.8, 2.8, 4.5, 1.3]: ")
  unk = np.array([[5.8, 2.8, 4.5, 1.3]],
    dtype=np.float32)
  unk = T.tensor(unk, dtype=T.float32).to(device) 
  with T.no_grad():
    logits = net(unk)
  probs = T.nn.functional.softmax(logits, dim=1)

  np.set_printoptions(precision=4)
  print(probs)

  # 6. TODO: save model

  print("\nEnd Iris demo")

if __name__ == "__main__":
  main()

# -------------------------------------------------