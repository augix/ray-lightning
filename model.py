import torch as T

class Net(T.nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.hid1 = T.nn.Linear(4, 7)  # 4-7-3
    self.oupt = T.nn.Linear(7, 3)
    # TODO: explicitly initialize weights

  def forward(self, x):
    z = T.tanh(self.hid1(x))
    z = self.oupt(z)  # see CrossEntropyLoss()
    return z
