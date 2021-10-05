"""MITDataset and MITProtoDataset for downstream application experiments."""
import torch
import numpy as np

import glob
import os
import random
import pdb
import re

# Global setting: all user IDs
MIT_USERS = [
    'tmj', 'ff', 'hp', 'kae', 'tu', 'krl', 'emb', 'po', 'CNG', 'jcw', 'jw',
    'ya', 'zb', 'ajs', 'ems'
]  # randomly shuffled


class MITDataset(torch.utils.data.Dataset):

  def __init__(self,
               folder_name,
               signal_length=180,
               mode="train",
               aug_transform=None,
               shot=10,
               way=1003):
    if signal_length is None:
      self.signal_length = int(1e10)
    else:
      self.signal_length = signal_length

    self.folder_name = folder_name
    self.files = glob.glob(os.path.join(self.folder_name, "*.txt"))
    self.files = sorted(self.files)
    self.shot = shot
    self.way = way

    # Step 1: find the stim names
    STIM_NAME_FILE = "cache_mit_stims.txt"
    STIM_NAME_LIST = "cache_mit_stims_all.txt"
    if os.path.exists(STIM_NAME_FILE) and os.path.exists(STIM_NAME_LIST):
      self.stim_names = open(STIM_NAME_FILE, "r").read().split()
      self.stim_names_list = open(STIM_NAME_LIST, "r").read().split()
    else:
      self.stim_names_list = [
          re.findall("(.*)_user_", os.path.basename(_))[0] for _ in self.files
      ]
      self.stim_names = sorted(list(set(self.stim_names_list)))
      open(STIM_NAME_FILE, "w").write("\n".join(self.stim_names))
      open(STIM_NAME_LIST, "w").write("\n".join(self.stim_names_list))

    # Step 2: Use the first k stimuli (k-way)
    self.stim_names = sorted(self.stim_names)
    self.stim_names = self.stim_names[:self.way]
    idx = [
        i for i, _ in enumerate(self.stim_names_list) if _ in self.stim_names
    ]
    self.files = np.array(self.files)[idx].tolist()

    # Step 3: Find the users (split train/test/val)
    # Note: empty list could be casted to False
    def is_valid(fname):  # The last 2 users are always for validation
      return [True for _ in MIT_USERS[-2:] if fname.endswith("_%s.txt" % _)]

    def is_train(fname):  # The first k-users are always for training
      return [True for _ in MIT_USERS[:shot] if fname.endswith("_%s.txt" % _)]

    def is_test(fname):  # The middle users are used for testing
      return not is_valid(fname) and not is_train(fname)

    if mode == "train":
      self.files = [_ for _ in self.files if is_train(_)]
    elif mode == "valid":
      self.files = [_ for _ in self.files if is_valid(_)]
    elif mode == "test":
      self.files = [_ for _ in self.files if is_test(_)]
    else:
      raise "Unknown Mode"

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
        data = data.reshape(-1, 3)
        x = data[:, 1].reshape(-1, 1)
        y = data[:, 2].reshape(-1, 1)
      except Exception as e:
        print(filename, e)

      signal = np.concatenate([x, y], axis=1)
      self.cached_data[filename] = signal

    if signal.shape[0] > self.signal_length:
      # Cut the signal
      start_idx = random.randint(0, signal.shape[0] - self.signal_length)
      signal = signal[start_idx:start_idx + self.signal_length, :]
    else:
      # Pad the signal
      bg = np.zeros(shape=(self.signal_length, 2))
      bg[:signal.shape[0], :] = signal
      signal = bg

    if self.aug_transform:
      signal = self.aug_transform(signal)

    stim_name = re.findall("(.*)_user_", os.path.basename(filename))[0]

    try:
      label = self.stim_names.index(stim_name)
    except Exception as e:
      print(e)
      print("I cannot find the stim name")
      pdb.set_trace()

    return signal, np.array(label)


class MITProtoDataset(torch.utils.data.Dataset):
  """MIT ProtoNet Dataset: a meta-learning dataset and loader.
  
  In this implementation, the batch size and k (for k-way classification)
  are different from vanilla PyTorch implementation.
  
  For training:
    - We record the batch size here, which control the actual "batch-size" with
    number of support (shot) and queries for training.
    - The so called "batch_size" in dataloader is the "k-way" classification.
  
  For testing:
    - We record the batch size here, which is number of support (shot) + number of testing
    - The so called "batch_size" in dataloader is the "k-way" classification.      
  """

  def __init__(self,
               folder_name,
               signal_length=180,
               mode="train",
               aug_transform=None,
               shot=10,
               way=1003):
    super().__init__()

    if signal_length is None:
      self.signal_length = int(1e10)
    else:
      self.signal_length = signal_length

    self.mode = mode
    self.folder_name = folder_name
    self.files = glob.glob(os.path.join(self.folder_name, "*.txt"))
    self.files = sorted(self.files)
    self.shot = shot
    self.way = way

    # Step 1: find the stim names
    STIM_NAME_FILE = "cache_mit_stims.txt"
    STIM_NAME_LIST = "cache_mit_stims_all.txt"
    if os.path.exists(STIM_NAME_FILE) and os.path.exists(STIM_NAME_LIST):
      self.stim_names = open(STIM_NAME_FILE, "r").read().split()
      self.stim_names_list = open(STIM_NAME_LIST, "r").read().split()
    else:
      self.stim_names_list = [
          re.findall("(.*)_user_", os.path.basename(_))[0] for _ in self.files
      ]
      self.stim_names = sorted(list(set(self.stim_names_list)))
      open(STIM_NAME_FILE, "w").write("\n".join(self.stim_names))
      open(STIM_NAME_LIST, "w").write("\n".join(self.stim_names_list))

    # Step 2: find the stimuli
    self.stim_names = sorted(self.stim_names)

    N_CLS_IN_TEST = 200

    if self.mode == "test":
      # Use the first k stimuli (k-way) for meta-testing
      self.stim_names = self.stim_names[:
                                        N_CLS_IN_TEST]  # use the first 300 stimuli for meta-testing
    else:
      # Use the rest (not the first k) for meta-training
      self.stim_names = self.stim_names[
          N_CLS_IN_TEST:]  # use the rest (703) stimuli for meta-training

    idx = [
        i for i, _ in enumerate(self.stim_names_list) if _ in self.stim_names
    ]
    self.files = np.array(self.files)[idx].tolist()

    self.files = sorted(self.files)

    stim_names_for_files = [
        re.findall("(.*)_user_", os.path.basename(_))[0] for _ in self.files
    ]
    self.labels = [self.stim_names.index(_) for _ in stim_names_for_files]

    # Don't worry about the users for this meta-learning setting;
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
        data = data.reshape(-1, 3)
        x = data[:, 1].reshape(-1, 1)
        y = data[:, 2].reshape(-1, 1)
      except Exception as e:
        print(filename, e)

        pdb.set_trace()
      signal = np.concatenate([x, y], axis=1)
      self.cached_data[filename] = signal

    if signal.shape[0] > self.signal_length:
      # Cut the signal
      start_idx = random.randint(0, signal.shape[0] - self.signal_length)
      signal = signal[start_idx:start_idx + self.signal_length, :]
    else:
      # Pad the signal
      bg = np.zeros(shape=(self.signal_length, 2))
      bg[:signal.shape[0], :] = signal
      signal = bg

    if self.aug_transform:
      signal = self.aug_transform(signal)

    stim_name = re.findall("(.*)_user_", os.path.basename(filename))[0]

    try:
      label = self.stim_names.index(stim_name)
    except Exception as e:
      print(e)
      print("I cannot find the stim name")
      pdb.set_trace()

    return signal, np.array(label)
