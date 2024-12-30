import torch as T
import pandas as pd

class IrisDataset(T.utils.data.Dataset):
  def __init__(self, fn_data):
    df = pd.read_csv(fn_data)
    # convert df['species'] to int
    df['species'] = df['species'].astype('category').cat.codes
    # convert df to numpy array
    data_array = df.to_numpy()
    x_tmp = data_array[:, 0:4]
    y_tmp = data_array[:, 4]
    # Move tensors to GPU
    self.x_data = T.tensor(x_tmp, dtype=T.float32)
    self.y_data = T.tensor(y_tmp, dtype=T.int64)
    
  def __len__(self):
    return len(self.x_data)

  def __getitem__(self, idx):
    x = self.x_data[idx]
    y = self.y_data[idx] 
    sample = (x, y)
    return sample
