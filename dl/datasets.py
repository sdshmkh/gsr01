import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from .rpca import RobustPCA
from utils.cvxeda import cvxEDA

class GSRLowRankDataset(Dataset):
  """
    GSRLowRankDataset performs Standard scaling and RPCA. After the processing, it windows the data 
    by the given window size.
  """
  def __init__(self, csv_file, window):
    super().__init__()

    self.df = pd.read_csv(csv_file, skiprows=1)
    self.df = self.df[13000:]

    self.num_windows = len(self.df) // window
    self.df = self.df[:self.num_windows * window]

    signal = self.df['uS'].to_numpy()
    self.scaled_gsr_signal = StandardScaler().fit_transform((signal).reshape(-1, 1))
    rpca = RobustPCA(n_components=2)
    data_matrix = signal.reshape(-1, 51)
    rpca.fit(data_matrix)

    self.gsr_signal = torch.tensor(rpca.low_rank_.reshape((self.num_windows, window, 1)), dtype=torch.float32)

    self.labels = np.array(self.df['label']).reshape((self.num_windows, window, 1))
    self.labels = np.array([np.argmax(np.bincount(self.labels[i].flatten())) for i in range(self.num_windows)])

  def __len__(self):
    return len(self.gsr_signal)


  def __getitem__(self, idx):
    return self.gsr_signal[idx]




class GSRDataset(Dataset):
  """
    GSRDataset performs Min-Max scaling. After the processing, it windows the data 
    by the given window size.
  """
  def __init__(self, csv_file, window):
    super().__init__()

    self.df = pd.read_csv(csv_file, skiprows=1)
    self.df = self.df[13000:]

    self.num_windows = len(self.df) // window
    self.df = self.df[:self.num_windows * window]

    signal = self.df['uS'].to_numpy()
    self.scaled_gsr_signal = MinMaxScaler().fit_transform((signal).reshape(-1, 1))
    self.gsr_signal = torch.tensor(self.scaled_gsr_signal.reshape((self.num_windows, window, 1)), dtype=torch.float32)

    self.labels = np.array(self.df['label']).reshape((self.num_windows, window, 1))
    self.labels = np.array([np.argmax(np.bincount(self.labels[i].flatten())) for i in range(self.num_windows)])

    print(self.gsr_signal.shape, self.labels.shape)


  def __len__(self):
    return len(self.gsr_signal)


  def __getitem__(self, idx):
    return self.gsr_signal[idx]


class GSRTonicDataset(Dataset):
  """
    GSRTonicDataset performs Min-Max scaling and extracts Tonic component with cxvEda.
    After the processing, it windows the data by the given window size.
  """
  def __init__(self, csv_file, window):
    super().__init__()

    self.df = pd.read_csv(csv_file, skiprows=1)
    self.df = self.df[13000:]

    self.num_windows = len(self.df) // window
    self.df = self.df[:self.num_windows * window]

    self.scaled_gsr_signal = MinMaxScaler().fit_transform(np.array(self.df['uS']).reshape(-1, 1))
    self.gsr_signal = torch.tensor(self.scaled_gsr_signal.reshape((self.num_windows, window, 1)), dtype=torch.float32)

    _, _, self.tonic_signal, _, _, _, _ = cvxEDA(self.scaled_gsr_signal, 1/window)
    self.tonic_signal = self.tonic_signal.reshape((self.num_windows, window, 1))
    self.tonic_signal = torch.tensor(self.tonic_signal, dtype=torch.float32)

    self.labels = np.array(self.df['label']).reshape((self.num_windows, window, 1))
    self.labels = np.array([np.argmax(np.bincount(self.labels[i].flatten())) for i in range(self.num_windows)])


  def __len__(self):
    return len(self.tonic_signal)


  def __getitem__(self, idx):
    return self.tonic_signal[idx]


class GSRPhasicDataset(Dataset):
  """
    GSRPhasicDataset performs Min-Max scaling and extracts Phasic component with cxvEda.
    After the processing, it windows the data by the given window size.
  """
  def __init__(self, csv_file, window):
    super().__init__()

    self.df = pd.read_csv(csv_file, skiprows=1)
    self.df = self.df[13000:]

    self.num_windows = len(self.df) // window
    self.df = self.df[:self.num_windows * window]

    self.scaled_gsr_signal = MinMaxScaler().fit_transform(np.array(self.df['uS']).reshape(-1, 1))
    self.gsr_signal = torch.tensor(self.scaled_gsr_signal.reshape((self.num_windows, window, 1)), dtype=torch.float32)

    self.phasic_signal, _, t, _, _, _, _ = cvxEDA(self.scaled_gsr_signal, 1/window)
    self.phasic_signal = self.phasic_signal.reshape((self.num_windows, window, 1))
    self.phasic_signal = torch.tensor(self.phasic_signal, dtype=torch.float32)

    self.labels = np.array(self.df['label']).reshape((self.num_windows, window, 1))
    self.labels = np.array([np.argmax(np.bincount(self.labels[i].flatten())) for i in range(self.num_windows)])


  def __len__(self):
    return len(self.gsr_signal)


  def __getitem__(self, idx):
    return self.phasic_signal[idx]