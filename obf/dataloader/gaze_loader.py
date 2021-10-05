from .augmenter import train_transform, valid_transform

import matplotlib.pyplot as plt

from torchvision.transforms.functional import to_tensor

import pandas as pd
from scipy.interpolate import interp1d
import numpy as np

import torch
import torch.utils.data

import glob

import random
import os
import pdb
import re


def _helper_read_cleaned_csv(filename):
  data = np.loadtxt(filename, dtype=np.float32, delimiter=",")
  try:
    x = data[:, 1].reshape(-1, 1)
    y = data[:, 2].reshape(-1, 1)
  except Exception as e:
    print(filename, e)

  signal = np.concatenate([x, y], axis=1)
  return signal


class SignalDataset(torch.utils.data.Dataset):

  def __init__(self,
               signal_length=None,
               is_train=True,
               folder_name="",
               aug_transform=None):
    """Signal dataset that will load scanpaths from the input folder.
    
    Args:
      signal_length (int): the maximum signal length to load.
      is_train (bool): True if this dataset is for training purpose.
        We will use the first 80% files for training and the rest for
        validation.
      folder_name (str): the folder path.
      aug_transform (): if specified, the transformation will be applied
        for input signals.
    """
    if signal_length is None:
      self.signal_length = int(1e10)
    else:
      self.signal_length = signal_length
    self.folder_name = folder_name

    self.files = glob.glob(os.path.join(self.folder_name, "*.txt"))
    self.files = sorted(self.files)

    if is_train:
      self.files = self.files[:int(0.8 * len(self.files))]
    else:
      self.files = self.files[int(0.8 * len(self.files)):]

    self.aug_transform = aug_transform
    self.cached_data = {}

  def __len__(self):
    return len(self.files)

  def __getitem__(self, idx):
    filename = self.files[idx]

    if filename in self.cached_data:
      signal = self.cached_data[filename]
    else:
      data = np.loadtxt(filename, dtype=np.float32, delimiter=",")
      try:
        x = data[:, 1].reshape(-1, 1)
        y = data[:, 2].reshape(-1, 1)
      except Exception as e:
        print(filename, e)

      signal = np.concatenate([x, y], axis=1)
      self.cached_data[filename] = signal

    if signal.shape[0] > self.signal_length:  # Cut the signal
      start_idx = random.randint(0, signal.shape[0] - self.signal_length)
      signal = signal[start_idx:start_idx + self.signal_length, :]
    else:  # Pad the signal
      bg = np.zeros(shape=(self.signal_length, 2))
      bg[:signal.shape[0], :] = signal
      signal = bg

    if self.aug_transform:
      signal = self.aug_transform(signal)

    return signal


class _ConcatBatchSampler(torch.utils.data.Sampler):

  def __init__(self, concat_dataset, batch_size, drop_last=True, shuffle=True):
    """Create a sampler to concatenate input datasets into one dataset.

    In ConcatDataset, each dataset might have different dimensions.
    So, each batch should only contain samples from one dataset.
    
    Logic:
      1. Choose the dataset from a list datasets
      2. Sample a batch from the dataset

    Args:
      concat_dataset (list): a list of datasets for concatenation.
      batch_size (int): data loader batch size.
      drop_last (bool, optional): if True, drop the last batch for each dataset.
        Defaults to True.
      shuffle (bool, optional): if True, shuffle the batches. Defaults to True.
    """
    super().__init__(concat_dataset)
    assert type(concat_dataset) is torch.utils.data.dataset.ConcatDataset

    self.batch_size = batch_size
    self.drop_last = drop_last
    self.shuffle = shuffle

    self.datasets = concat_dataset.datasets
    self.ds_lengths = [len(_) for _ in self.datasets]
    self.__helper_reset_index__()

  def __helper_reset_index__(self):
    batches = []
    prev_length = 0
    for length in self.ds_lengths:
      indices = list(range(prev_length, prev_length + length))

      if self.shuffle:
        random.shuffle(indices)

      idx = 0
      while idx < length:
        batches.append(indices[idx:idx + self.batch_size])
        idx += self.batch_size

      if not self.drop_last:
        batches.append(indices[idx:]) if idx < length else None
      prev_length += length

    # Now, batches contains all
    if self.shuffle:
      random.shuffle(batches)
    self.batches = iter(batches)

  def __iter__(self):
    self.__helper_reset_index__()
    # self.batches is already an iterator. So, we can directly yield "from" it
    yield from self.batches

  def __len__(self):
    # if drop last batch, use np.floor
    func = np.floor if self.drop_last else np.ceil
    rst = np.sum([func(len(_) / self.batch_size) for _ in self.datasets])
    return int(rst)


def get_pretrain_data_loader(mode, pretrain_data_setting):
  """Get pre-training loader.

  Args:
      mode (str): either "train" or "valid".
      pretrain_data_setting (dict, optional): pretrain dataset setting.

  Returns:
      loader (torch.dataloader): a PyTorch dataloader with all input
        datasets.
  """
  is_train = mode == "train"

  use_aug = pretrain_data_setting["use augmentation"]
  batch_size = pretrain_data_setting["batch size"]

  if use_aug and is_train:
    transform = train_transform
  else:
    transform = valid_transform

  dataset_arr = []  # hold all datasets.

  for dataset_info in pretrain_data_setting["datasets"]:
    dataset_ = SignalDataset(signal_length=dataset_info["signal length"],
                             is_train=is_train,
                             folder_name=dataset_info["path"],
                             aug_transform=transform)
    dataset_arr.append(dataset_)

  dataset = torch.utils.data.ConcatDataset(dataset_arr)

  sampler = _ConcatBatchSampler(dataset,
                                batch_size=batch_size,
                                drop_last=False,
                                shuffle=is_train)
  loader = torch.utils.data.DataLoader(dataset,
                                       pin_memory=True,
                                       batch_sampler=sampler)

  assert len(loader) > 0, "empty data loader from %s" % pretrain_data_setting
  return loader
