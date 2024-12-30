# minimum pytorch example
fn_data = 'iris_data.csv'

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
  train_ds = IrisDataset(fn_data)
  bat_size = 2
  train_ldr = T.utils.data.DataLoader(train_ds,
    batch_size=bat_size, shuffle=True)

# -------------------------------------------------

  # 2. create network
  print("\nCreating 4-7-3 neural network ")
  net = Net().to(device)    # or Sequential()

  # 3. train model
  max_epochs = 100
  lrn_rate = 0.04
  loss_func = T.nn.CrossEntropyLoss()  # applies log-softmax()
  optimizer = T.optim.SGD(net.parameters(),
    lr=lrn_rate)

  print("\nStarting training")
  net = net.train()
  for epoch in range(0, max_epochs):
    for (batch_idx, batch) in enumerate(train_ldr):
      X = batch[0].to(device)
      Y = batch[1].to(device)
      optimizer.zero_grad()
      oupt = net(X)
      loss_val = loss_func(oupt, Y)  # a tensor
      loss_val.backward()
      optimizer.step()

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