"""The autoencoder (AE) model."""
import math
import pdb
from abc import ABC

import numpy as np
import torch
import torch.nn as nn
from termcolor import colored

EOS_VALUE = -100  # value for End-of-Sequence


#  Utils and Basic Networks
def count_params(model):
  """Return the number of parameters in the model.

  Args:
    model (torch.module): a PyTorch model.
  
  Returns:
    n_params (int): number of parameters in the given model.
  """
  model_parameters = filter(lambda p: p.requires_grad, model.parameters())
  n_params = sum([np.prod(p.size()) for p in model_parameters])
  print("There are Total Trainable %d parameters" % n_params)
  return n_params


class Encoder(nn.Module, ABC):

  def __init__(self):
    super().__init__()

  def forward(self, x):
    """
    The encoder can handle any arbitrary length of input

    Args:
      x (torch.tensor): (batch_size, seq_len, num_in_dim)

    Outputs:
      out (torch.tensor): the encoder
    """
    return x


class Decoder(nn.Module, ABC):

  def __init__(self):
    super().__init__()

  def forward(self, x):
    """
    The encoder can handle any arbitrary length of input

    Args:
      x (torch.tensor): (batch_size, seq_len, num_in_dim)

    Outputs:
      out (torch.tensor): the output
    """
    return x


class MLP(nn.Module):

  def __init__(self,
               input_dim=2,
               layers=[8, 32, 64],
               drop_p=0,
               activation="sigmoid",
               batch_norm=True):

    super().__init__()
    mlp_layers = []
    prev_dim = input_dim

    for i, curr_dim in enumerate(layers):
      mlp_layers.append(nn.Linear(prev_dim, curr_dim))
      prev_dim = curr_dim

      if i == len(layers) - 1:
        continue  # last layer, we don't need activation or batch norm

      # Activations
      if activation == "sigmoid":
        mlp_layers.append(nn.Sigmoid())
      elif activation == "relu":
        mlp_layers.append(nn.ReLU())
      elif activation == "leaky":
        mlp_layers.append(nn.LeakyReLU())
      else:
        raise ("Unknown Activation")

      if drop_p > 0:
        mlp_layers.append(nn.Dropout(drop_p))

      if batch_norm:
        mlp_layers.append(nn.BatchNorm1d(curr_dim))

    self.mlp = nn.Sequential(*mlp_layers)

  def forward(self, x):
    """
    The encoder can handle any arbitrary length of input

    Args:
      x (torch.tensor): input data with (batch_size, .., ..)

    Outputs:
      out (torch.tensor): output data with (batch_size, k), where k is the number
        of output neurons.
    """
    x = x.reshape(x.shape[0], -1)  # flatten to (batch_size, ..)
    out = self.mlp(x)
    return out


class MeanAcrossDim(torch.nn.Module):

  def __init__(self, dim):
    super().__init__()
    self.dim = dim

  def forward(self, x):
    return x.mean(dim=self.dim)


class LastSlice(torch.nn.Module):

  def __init__(self):
    super().__init__()

  def forward(self, x):
    return x[:, -1, :]


class CatDim(torch.nn.Module):

  def __init__(self, dim):
    super().__init__()
    self.dim = dim

  def forward(self, x):
    return torch.cat(x, dim=self.dim)


class Transpose(torch.nn.Module):

  def __init__(self, dim0, dim1):
    super().__init__()
    self.dim0 = dim0
    self.dim1 = dim1

  def forward(self, x):
    return x.transpose(self.dim0, self.dim1)


class ConvBlock(nn.Module):

  def __init__(self,
               prev_dim,
               curr_dim,
               activation="leaky",
               batch_norm=True,
               pool=False):
    super().__init__()

    print(curr_dim, prev_dim)
    if curr_dim - prev_dim > 1:
      self.use_residual = True
      out_dim = curr_dim - prev_dim
    else:
      self.use_residual = False
      out_dim = curr_dim

    self.conv = nn.Conv1d(prev_dim, out_dim, 7, padding=3)

    self.pool = pool
    if self.pool:
      self.pool_layer = nn.AvgPool1d(2, 2)

    self.batch_norm = batch_norm
    if self.batch_norm:
      self.bn = nn.BatchNorm1d(out_dim)

    if activation == "sigmoid":
      self.activation = nn.Sigmoid()
    elif activation == "relu":
      self.activation = nn.ReLU()
    elif activation == "leaky":
      self.activation = nn.LeakyReLU()
    else:
      raise ("Unknown Activation")

  def forward(self, x):
    y = self.activation(self.conv(x))
    if self.batch_norm:
      y = self.bn(y)

    if self.use_residual:
      out = torch.cat((x, y), dim=1)
    else:
      out = y

    if self.pool:
      out = self.pool_layer(out)

    return out


class CNNEncoder(Encoder):

  def __init__(self,
               input_dim=2,
               input_length=1000,
               latent_dim=64,
               layers=[
                   8,
                   32,
               ],
               activation="leaky",
               batch_norm=True):
    super().__init__()
    cnn_layers = []
    prev_dim = input_dim

    layers.append(latent_dim)  # Add the output layer
    print("CNN Layers:", layers)
    output_length = input_length

    for curr_dim in layers:
      cnn_layers.append(
          ConvBlock(prev_dim,
                    curr_dim,
                    activation=activation,
                    batch_norm=batch_norm,
                    pool=curr_dim != latent_dim))

      prev_dim = curr_dim

    self.cnn = nn.Sequential(*cnn_layers)
    self.output_length = int(output_length)

  def forward(self, x):
    """
    The encoder can handle any arbitrary length of input

    Args:
      x (torch.tensor): (batch_size, seq_len, num_in_dim)

    Outputs:
      out (torch.tensor): (batch_size, out_seq_len, num_out_dim)
    """
    x = x.transpose(1, 2)  # (batch_size, num_in_dim, seq_len)
    out = self.cnn(x)  # (batch_size, num_out_dim, out_seq_len)
    out = out.transpose(1, 2)
    assert x.shape[0] == out.shape[0], "batch size NOT match in CNNEncoder."
    return out


# RNNs
class RNNEncoder(Encoder):

  def __init__(self,
               input_dim=2,
               latent_dim=128,
               nlayers=3,
               backbone="gru",
               dropout=0.1,
               bidirectional=False,
               layer_norm=True):
    """RNN Encoder model.
      
    Args:
      input_dim (int): number of input channels (dimension).
      latent_dim (int): latent dimension for the RNN structure.
      nlayers (int): number of RNN layers
      backbone (str): either "gru", "lstm", or "rnn" for the backbone.
      dropout (float): the dropout ratio for all RNN layers.
      bidirectional (bool): if True, use bidirectional structure.
      layer_norm (bool): if True, apply layer norm in RNN.
    """
    super().__init__()

    self.backbone = backbone
    self.layer_norm = layer_norm
    self.latent_dim = latent_dim
    self.nlayers = nlayers

    if backbone == "gru":
      func = nn.GRU
    elif backbone == "lstm":
      func = nn.LSTM
    elif backbone == "rnn":
      func = nn.RNN
    else:
      raise ("Unknown Backbone for RNN")

    self.rnn = func(input_size=input_dim,
                    hidden_size=latent_dim,
                    num_layers=nlayers,
                    bias=True,
                    batch_first=True,
                    dropout=dropout,
                    bidirectional=bidirectional)

    if layer_norm:
      self.ln = nn.LayerNorm((nlayers, latent_dim))

  def forward(self, x):
    """
    
    Args:
      x (torch.tensor): (batch_size, seq_len, num_in_dims)
    Outputs:
      hid (torch.tensor or tuple): for RNN or GRU: (batch_size, num_layers,
         num_out_dims).For LSTM, a pair of tensor with (batch_size, num_layers,
         num_out_dims) shape
    """
    out, hid = self.rnn(x.contiguous())
    # out is a (batch_size, seq_len, num_out_dim) tensor,
    # hid is a (num_layers, batch_size, num_out_dim) tensor or tuple of the dim.
    del out

    # Apply transpose and layer norm
    if type(hid) is tuple:
      # it is a tuple with 2 elements: (num_layers, batch_size, out_dim)
      hid = (hid[0].transpose(0, 1), hid[1].transpose(0, 1))
      if self.layer_norm:
        hid = (self.ln(hid[0]), self.ln(hid[1]))
    else:
      hid = hid.transpose(0, 1)
      if self.layer_norm:
        hid = self.ln(hid)
    # pdb.set_trace()
    return hid


class RNNDecoder(Decoder):

  def __init__(self,
               seq_length=10,
               input_dim=2,
               latent_dim=128,
               out_dim=2,
               nlayers=3,
               backbone="gru",
               dropout=0.1,
               bidirectional=False,
               batch_norm=True):
    """Create a RNN Decoder.

    Only use RNN Decoder if the last layer of encoder is also RNN.

    The latent dimension and backbone should match for the encoder and decoder.
    """
    super().__init__()

    self.seq_length = seq_length
    self.backbone = backbone
    self.batch_norm = batch_norm

    assert latent_dim == input_dim, "Our design need input dim equals latent dim"
    self.input_dim = input_dim
    self.latent_dim = latent_dim

    if backbone == "gru":
      func = nn.GRU
    elif backbone == "lstm":
      func = nn.LSTM
    elif backbone == "rnn":
      func = nn.RNN
    else:
      raise ("Unknown Backbone for RNN")

    self.rnn = func(input_size=input_dim,
                    hidden_size=latent_dim,
                    num_layers=nlayers,
                    bias=True,
                    batch_first=True,
                    dropout=dropout,
                    bidirectional=bidirectional)

    if batch_norm:
      self.bn = nn.BatchNorm1d(latent_dim)
    self.out_fc = nn.Linear(latent_dim, out_dim)

  def forward(self, hiddens):
    """Run the RNNDecoder.

    Args:
      hidden (torch.tensor or tuple): hidden data from the encoder (RNN, GRU, or 
        LSTM). Its (batch_size, num_layers, num_dims) or a pair of 
        (batch_size, num_layers, num_dims).
        
    Outputs:
      outs (torch.tensor): output with (batch_size, sequence_len, num_out_dims).
    """
    # Get batch size, and Transpose hidden layers to
    # (num_layers, batch_size, num_dims).
    if type(hiddens) is tuple:
      bs = hiddens[0].shape[0]
      hiddens = (hiddens[0].transpose(0, 1).contiguous(),
                 hiddens[1].transpose(0, 1).contiguous())
    else:
      bs = hiddens.shape[0]
      hiddens = hiddens.transpose(0, 1).contiguous()

    inputs = torch.zeros(
        (bs, 1, self.latent_dim)).to(hiddens[0].device).contiguous()
    outs = []

    for _ in range(self.seq_length):
      inputs, hiddens = self.rnn(inputs, hiddens)
      # the "outputs" is the "inputs" of next round
      outs.append(inputs.clone())

    outs = torch.cat(outs, dim=1)  # a (batch_size, seq_len, num_out_dim) tensor

    # Apply FC layer to the output layer
    outs = self.out_fc(outs)

    return outs


#  Transformers
class PositionalEncoding(nn.Module):

  def __init__(self, d_model, dropout=0.1, max_len=1000):
    super(PositionalEncoding, self).__init__()
    self.dropout = nn.Dropout(p=dropout)

    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0).transpose(0, 1)
    self.register_buffer('pe', pe)

  def forward(self, x):
    x = x + self.pe[:x.size(0), :]
    return self.dropout(x)

