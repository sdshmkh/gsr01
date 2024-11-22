import numpy as np
import torch
from torch import nn
import tqdm


def get_arch(latent_space):
  """
    Returns a pytorch Sequential model for autoencoders and decoders based
    on a specific latent_spaces. Supported latent spaces are 3, 6, 96, 128, 256, 512.

    latent_space (int): Latent space size

    returns: (nn.Sequential, nn.Sequential)
  """
  if latent_space == 3:
    enc = nn.Sequential(
        nn.Linear(51, 25),
        nn.ReLU(),
        nn.Linear(25, 12),
        nn.ReLU(),
        nn.Linear(12, 6),
        nn.ReLU(),
        nn.Linear(6, 3),
        nn.ReLU(),
    )
    dec = nn.Sequential(
        nn.Linear(3, 6),
        nn.ReLU(),
        nn.Linear(6, 12),
        nn.ReLU(),
        nn.Linear(12, 25),
        nn.ReLU(),
        nn.Linear(25, 51),
        nn.Sigmoid()
    )
    return enc, dec
  if latent_space == 6:
   enc = nn.Sequential(
        nn.Linear(51, 25),
        nn.ReLU(),
        nn.Linear(25, 12),
        nn.ReLU(),
        nn.Linear(12, 6),
        nn.ReLU(),
    )
   dec = nn.Sequential(
        nn.Linear(6, 12),
        nn.ReLU(),
        nn.Linear(12, 25),
        nn.ReLU(),
        nn.Linear(25, 51),
        nn.Sigmoid()
   )
   return enc, dec
  if latent_space == 96:
    enc = nn.Sequential(
        nn.Linear(51, 96),
        nn.ReLU(),
    )
    dec = nn.Sequential(
        nn.Linear(96, 51),
        nn.Sigmoid()
    )
    return enc, dec
  if latent_space == 128:
    enc = nn.Sequential(
        nn.Linear(51, 96),
        nn.ReLU(),
        nn.Linear(96, 128),
        nn.ReLU(),
    )

    dec = nn.Sequential(
        nn.Linear(128, 96),
        nn.ReLU(),
        nn.Linear(96, 51),
        nn.Sigmoid()
    )
    return enc, dec
  if latent_space == 256:
    enc = nn.Sequential(
        nn.Linear(51, 96),
        nn.ReLU(),
        nn.Linear(96, 128),
        nn.ReLU(),
        nn.Linear(128, 256),
        nn.ReLU(),
    )

    dec = nn.Sequential(
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 96),
        nn.ReLU(),
        nn.Linear(96, 51),
        nn.Sigmoid()
    )
    return enc, dec
  if latent_space == 512:
    enc = nn.Sequential(
        nn.Linear(51, 96),
        nn.ReLU(),
        nn.Linear(96, 128),
        nn.ReLU(),
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 512),
        nn.ReLU(),
    )

    dec = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 96),
        nn.ReLU(),
        nn.Linear(96, 51),
        nn.Sigmoid()
    )

    return enc, dec



class AutoEncoder(nn.Module):
  def __init__(self, latent_dim):
    super().__init__()

    enc, dec = get_arch(latent_dim)
    self.encoder = enc

    self.decoder = dec


  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x

  def get_latent_space(self, x):
    with torch.no_grad():
      return self.encoder(x)

def contractive_loss(model, output, target, lamda):
  """
    Contractive loss which add regularizations using the Jacobian of the latent space
  """
  batch_size = output.shape[0]
  fro_norm = 0.0

  for i in range(batch_size):
      curr_target = target[i].unsqueeze(0)  # Add batch dimension for single sample
      jac = torch.autograd.functional.jacobian(lambda x: model.get_latent_space(x), curr_target, create_graph=True)

      fro_norm += torch.sum(jac ** 2)

  mse = nn.functional.mse_loss(output, target)

  return mse + lamda * (fro_norm / batch_size)


def training_loop(model, dl, val_dl, epochs, lr, lamda):
  model.train()
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  losses = list()
  with tqdm.tqdm(range(epochs)) as t:
    t.set_description('Training')
    for i in t:
      batch_loss = 0
      for batch in dl:
        batch = batch.reshape(-1, 51)
        output = model(batch)
        loss = contractive_loss(model, output, batch, lamda)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_loss += loss.item()

      with torch.no_grad():
        val_loss = 0
        for batch in val_dl:
          batch = batch.reshape(-1, 51)
          output = model(batch)
          loss = contractive_loss(model, output, batch, lamda)
          val_loss += loss.item()
        val_loss /= len(val_dl)
      t.set_postfix(loss=batch_loss / len(dl), val_loss=val_loss)
      losses.append((batch_loss / len(dl), val_loss))
  return np.array(losses)