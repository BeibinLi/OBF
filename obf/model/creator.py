"""Functions to create deep learning models.

Parameter definitions:
  backbone_type (str): backbone RNN or Transformer type.
  use_conv (bool): if True, use Conv layers before RNN/Transformer layers.
  hidden_dim (int): hidden dimension for recurrent/Transformer layers
  nlayers (int): number of layers
  pc_seq_length (int): predictive coding output length
  input_seq_length (int): input sequence length (for batch processing).
  recon_seq_length (int): reconstruction sequence length.
  use_cuda (bool): if True, use CUDA acceleration.
"""
import glob
import os

import torch
import torch.nn as nn

from . import ae


def print_models_info(names, models):
  """Print number of parameters and model structures for each model.

  Args:
      names (list): list of strings for model names.
      models (list): list of PyTorch models.
  """
  for name, model in zip(names, models):
    print(name.upper(), "-" * 10)
    ae.count_params(model)
    print(model)


def create_models(model_setting):
  """Create the deep learning models.

  Args:
    model_setting (dict): model_setting setting.

  Returns:
    encoder (torch.module): encoder model.
    pc_decoder: predictive coding decoder.
    fi_decoder: fixation identification decoder.
    cl_decoder: contrastive learning decoder.
    rc_decoder: reconstruction decoder.
  """
  use_conv = model_setting["use conv"]
  backbone_type = model_setting["backbone type"]
  nlayers = model_setting["number of layers"]
  hidden_dim = model_setting["hidden dim"]
  use_cuda = model_setting["cuda"] and torch.cuda.is_available()

  input_seq_length = model_setting["input seq length"]
  recon_seq_length = model_setting["recon seq length"]
  pc_seq_length = model_setting["pc seq length"]

  if backbone_type in ['rnn', 'lstm', 'gru']:
    if use_conv:
      conv_dim = 32
      enc_layers = [
          ae.CNNEncoder(input_dim=2, latent_dim=conv_dim, layers=[
              16,
          ]),
          ae.RNNEncoder(input_dim=conv_dim,
                        latent_dim=hidden_dim,
                        backbone=backbone_type,
                        nlayers=nlayers,
                        layer_norm=False)
      ]
    else:
      enc_layers = [
          ae.RNNEncoder(input_dim=2,
                        latent_dim=hidden_dim,
                        backbone=backbone_type,
                        nlayers=nlayers,
                        layer_norm=False)
      ]

    encoder = nn.Sequential(*enc_layers)
    pc_decoder = ae.RNNDecoder(input_dim=hidden_dim,
                               latent_dim=hidden_dim,
                               out_dim=2,
                               seq_length=pc_seq_length,
                               backbone=backbone_type,
                               nlayers=nlayers,
                               batch_norm=False)
    fi_decoder = ae.RNNDecoder(input_dim=hidden_dim,
                               latent_dim=hidden_dim,
                               out_dim=2,
                               seq_length=input_seq_length,
                               backbone=backbone_type,
                               nlayers=nlayers,
                               batch_norm=True)

    cl_input_dim = hidden_dim * nlayers
    if backbone_type == "lstm":
      cl_input_dim *= 2

    # * 3 for rnn, * 2 for contrastive pair
    cl_decoder = ae.MLP(input_dim=cl_input_dim,
                        layers=[128, 2],
                        batch_norm=True)

    rc_decoder = ae.RNNDecoder(input_dim=hidden_dim,
                               latent_dim=hidden_dim,
                               out_dim=2,
                               seq_length=recon_seq_length,
                               backbone=backbone_type,
                               nlayers=nlayers,
                               batch_norm=True)
  else:
    raise "Unknown backbone type"

  print_models_info(["encoder", "pc", "fi", "cl", "rc"],
                    [encoder, pc_decoder, fi_decoder, cl_decoder, rc_decoder])

  if use_cuda:
    encoder = encoder.cuda()
    pc_decoder = pc_decoder.cuda()
    fi_decoder = fi_decoder.cuda()
    cl_decoder = cl_decoder.cuda()
    rc_decoder = rc_decoder.cuda()

    encoder = torch.nn.DataParallel(encoder)
    pc_decoder = torch.nn.DataParallel(pc_decoder)
    fi_decoder = torch.nn.DataParallel(fi_decoder)
    cl_decoder = torch.nn.DataParallel(cl_decoder)
    rc_decoder = torch.nn.DataParallel(rc_decoder)

  return encoder, pc_decoder, fi_decoder, cl_decoder, rc_decoder


def load_encoder(path, use_cuda=True):
  """Load encoder from given path (filename or folder name).

  Args:
      path (str): the encoder file name or its folder name.
  Raises:
      FileNotFoundError: if the provided path does not exists, raise error.

  Returns:
      encoder (torch.module): a deep learing encoder model.
  """

  if not os.path.exists(path):
    raise FileNotFoundError("%s not found." % path)

  if os.path.isdir(path) and os:
    pre_path = glob.glob(os.path.join(path, "encoder*.pt"))[0]
  else:
    pre_path = path

  print("Loading: ", pre_path)
  encoder = torch.load(pre_path, map_location="cpu")

  if use_cuda and torch.cuda.is_available():
    encoder = encoder.cuda()

  return encoder


def create_classifier_from_encoder(
    encoder,
    hidden_layers,
    n_output,
    dropout,
):

  if type(hidden_layers) is list:
    mlp_structure = hidden_layers + [n_output]
  else:
    mlp_structure = [n_output]

  if type(encoder[-1]) is ae.RNNEncoder:
    # RNN model
    rnn_type = encoder[-1].backbone
    dim = encoder[-1].rnn.hidden_size
    nlayers = encoder[-1].rnn.num_layers
    input_dim = dim * nlayers

    if rnn_type == "lstm":
      input_dim *= 2

    layers = [encoder]

    if rnn_type == "lstm":
      layers.append(ae.CatDim(dim=0))

    fc = ae.MLP(input_dim=input_dim,
                layers=mlp_structure,
                drop_p=dropout,
                activation="sigmoid",
                batch_norm=True)
    layers += [fc]
    model = nn.Sequential(*layers)
  else:
    raise "Unknown encoder type"

  return model
