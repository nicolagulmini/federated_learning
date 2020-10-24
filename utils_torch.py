"""
Created on Wed Oct 14 23:43:24 2020

@author: Nicola Gulmini
@mail: nicolagulmini@gmail.com or nicola.gulmini@studenti.unipd.it

*** IMPORTANT *** : based on the code from https://github.com/DakshIdnani/pytorch-nice

"""
# tensorflow 2.0 and keras 2.3 

import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from torch.distributions import Distribution, Uniform
import numpy as np

class CouplingLayer(nn.Module):
  """
  Implementation of the additive coupling layer from section 3.2 of the NICE
  paper.
  """

  def __init__(self, data_dim, hidden_dim, mask, num_layers=4):
    super().__init__()

    assert data_dim % 2 == 0

    self.mask = mask

    modules = [nn.Linear(data_dim, hidden_dim), nn.LeakyReLU(0.2)]
    for i in range(num_layers - 2):
      modules.append(nn.Linear(hidden_dim, hidden_dim))
      modules.append(nn.LeakyReLU(0.2))
    modules.append(nn.Linear(hidden_dim, data_dim))

    self.m = nn.Sequential(*modules)

  def forward(self, x, logdet, invert=False):
    if not invert:
      x1, x2 = self.mask * x, (1. - self.mask) * x
      y1, y2 = x1, x2 + (self.m(x1) * (1. - self.mask))
      return y1 + y2, logdet

    # Inverse additive coupling layer
    y1, y2 = self.mask * x, (1. - self.mask) * x
    x1, x2 = y1, y2 - (self.m(y1) * (1. - self.mask))
    return x1 + x2, logdet


class ScalingLayer(nn.Module):
  """
  Implementation of the scaling layer from section 3.3 of the NICE paper.
  """
  def __init__(self, data_dim):
    super().__init__()
    self.log_scale_vector = nn.Parameter(torch.randn(1, data_dim, requires_grad=True))

  def forward(self, x, logdet, invert=False):
    log_det_jacobian = torch.sum(self.log_scale_vector)

    if invert:
        return torch.exp(- self.log_scale_vector) * x, logdet - log_det_jacobian

    return torch.exp(self.log_scale_vector) * x, logdet + log_det_jacobian

class LogisticDistribution(Distribution):
    def __init__(self):
        super().__init__()

    def log_prob(self, x):
        return -(F.softplus(x) + F.softplus(-x))
    
    def prob(self, x):
        return torch.exp(torch.sum(self.log_prob(x)))

    def sample(self, size):
        z = Uniform(torch.FloatTensor([0.]), torch.FloatTensor([1.])).sample(size)
        return torch.log(z) - torch.log(1. - z)

class NICE(nn.Module):
  def __init__(self, data_dim, num_coupling_layers=3):
    super().__init__()

    self.data_dim = data_dim

    # alternating mask orientations for consecutive coupling layers
    masks = [self._get_mask(data_dim, orientation=(i % 2 == 0)) for i in range(num_coupling_layers)]

    self.coupling_layers = nn.ModuleList([CouplingLayer(
        data_dim=data_dim, 
        hidden_dim=100, # ho provato con 100, ora ne metto 2000 in accordo con quanto scritto nel paper 
        mask=masks[i], 
        num_layers=6)
        for i in range(num_coupling_layers)])

    self.scaling_layer = ScalingLayer(data_dim=data_dim)
    self.prior = LogisticDistribution()

  def forward(self, x, invert=False):
    if not invert:
      z, log_det_jacobian = self.f(x)
      log_likelihood = torch.sum(self.prior.log_prob(z), dim=1) + log_det_jacobian
      return z, log_likelihood

    return self.f_inverse(x)

  def f(self, x):
    z = x
    log_det_jacobian = 0
    for i, coupling_layer in enumerate(self.coupling_layers):
      z, log_det_jacobian = coupling_layer(z, log_det_jacobian)
    z, log_det_jacobian = self.scaling_layer(z, log_det_jacobian)
    return z, log_det_jacobian

  def f_inverse(self, z):
    x = z
    x, _ = self.scaling_layer(x, 0, invert=True)
    for i, coupling_layer in reversed(list(enumerate(self.coupling_layers))):
      x, _ = coupling_layer(x, 0, invert=True)
    return x

  def sample(self, num_samples):
    z = self.prior.sample([num_samples, self.data_dim]).view(num_samples, self.data_dim)
    print(z)
    return self.f_inverse(z)

  def _get_mask(self, dim, orientation=True):
    mask = np.zeros(dim)
    mask[::2] = 1.
    if orientation:
      mask = 1. - mask # flip mask orientation
    mask = torch.tensor(mask)
    return mask.float()

def train_estimation(model, epochs, dataloader):
    model.train()
    opt = optim.Adam(model.parameters())
    for i in range(epochs):
        mean_likelihood = 0.0
        num_minibatches = 0

        for batch_id, x in enumerate(dataloader):
            x = x.view(-1, 3072) + torch.rand(3072) / 256.
            x = torch.clamp(x, 0, 1) # serve per limitare 
            z, likelihood = model(x.float()) # ho messo .float() perche mi dava qualche problema
            #print(likelihood)
            loss = -torch.mean(likelihood) # NLL
            loss.backward()
            opt.step()
            model.zero_grad()
            mean_likelihood -= loss
            num_minibatches += 1

        mean_likelihood /= num_minibatches
        print('Epoch {} completed. Log Likelihood: {}'.format(i, mean_likelihood))   
        
            
def find_min_topk(list_of_resized_tensors, k):
    maxi_tensor = torch.cat(list_of_resized_tensors)
    topk = torch.topk(maxi_tensor, k, largest=True)[0]
    min_topk = torch.min(topk).numpy()
    return min_topk

def top_k_for_tensor(tensor, val):
    tensor = tensor.numpy()
    for i in range(tensor.shape[0]):
        if tensor[i] < val:
            tensor[i] = 0.0
    return torch.from_numpy(tensor)

def topk_sparsification_torch_load(k, path):
    #load_path = './saved_models/cifar10/39_channel_last.pt'
    load = torch.load(path)
    list_of_tensors = []
    for i in iter(load):
        list_of_tensors.append(load[i])
    list_of_resized_tensors = [t.reshape(-1) for t in list_of_tensors]
    min_topk = find_min_topk(list_of_resized_tensors, k)
    spars_t = []
    for t in list_of_resized_tensors:
        spars_t.append(top_k_for_tensor(t, min_topk))
    # inserisci nel dizionario
    j = 0
    for i in iter(load):
        load[i] = spars_t[j].view(load[i].shape)
        j += 1
    return load
    # model.load_state_dict(load)

def imshow(img):
    min_x = torch.min(img)
    max_x = torch.max(img)
    img = (img-min_x)/(max_x-min_x) # no, normalizzali! DA MODIFICARE ANCHE SE POCO IMPORTANTE
    torch.clamp(img, 0, 1)
    img = img.detach()
    npimg = img.numpy()
    plt.imshow(npimg)
    plt.show()