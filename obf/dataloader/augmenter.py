"""Data augmenter for Gaze data."""
from torchvision import transforms, utils
import torch

import pandas as pd
import numpy as np

import random
import pdb

DEBUG = False


class RandomAffineSignal(object):

  def __init__(self, translate=3, scale=0.2, rotate=0.314, shear=0.1):
    """
    Args:
      translate (float): random translate distance (in signal's input units)
      scale (float): random scale ratio
      rotate (float): random rotate degrees (in radians; e.g. 3.14, )
      shear (float): random ratio for shear (w.r.t. x, y)
    """

    self.translate = translate
    self.scale = scale
    self.rotate = rotate
    self.shear = shear

  def __call__(self, data):
    """Run the affine transformation.
    
    Args:
      data (np.array): 2D arrays (signal-length, 2)
      
    Output:
      augmented_data (np.array): 2D arrays that matches input data
    """
    sl, _ = data.shape

    tx = random.uniform(0, self.translate)  # translate x distance
    ty = random.uniform(0, self.translate)  # translate y distance
    translate_matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])

    sx = random.uniform(1 - self.scale, 1 + self.scale)  # scale for x
    sy = random.uniform(1 - self.scale, 1 + self.scale)  # scale for y
    scale_matrix = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])

    rd = random.uniform(0, self.rotate)
    cr = np.cos(rd)  # cos(theta)
    sr = np.sin(rd)  # sin(theta)
    rotate_matrix = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])

    cx = random.uniform(0, self.shear)  # shear x
    cy = random.uniform(0, self.shear)  # shear y

    shear_matrix = np.array([[1, cx, 0], [cy, 1, 0], [0, 0, 1]])

    # Combine to an affine matrix
    affine_Matrix = np.dot(
        np.dot(np.dot(translate_matrix, scale_matrix), rotate_matrix),
        shear_matrix)

    # Reshape signal to (3, batch-size, signal-length), where 3 is from (x, y, 1)
    data_reshape = data.swapaxes(0, 1)
    data_reshape = np.concatenate(
        (data_reshape, np.ones((1, sl), dtype=data_reshape.dtype)), axis=0)

    aug_data = np.dot(affine_Matrix, data_reshape)

    aug_data = aug_data.swapaxes(0, 1)
    aug_data = aug_data[:, 0:2]

    return aug_data  # remove the final "1"


class RandomPointNoise(object):

  def __init__(self, max_std=0.5):
    self.max_std = max_std

  def __call__(self, data):
    """Add random point noise to signal.
    
    Args:
      data (np.array): 2D arrays (batch-size, signal-length, 2)
      
    Output:
      augmented_data (np.array): 2D arrays that matches input data
    """
    rand_std = np.random.uniform(low=0, high=self.max_std, size=data.shape)
    noise = np.random.normal(loc=0, scale=rand_std, size=data.shape)

    aug_data = data + noise

    return aug_data


class ToFloatTensor(object):

  def __call__(self, data):
    """Convert to float tensor.

    Args:
      data (np.array): 2D arrays (batch-size, signal-length, 2)
      
    Output:
      data (torch.tensor): 2D arrays that matches input data
    """
    return torch.tensor(data).float()


train_transform = transforms.Compose([
    RandomAffineSignal(translate=1, scale=0.05, rotate=3.14 / 50, shear=0.01),
    RandomPointNoise(max_std=0.2),
    ToFloatTensor()
])

valid_transform = transforms.Compose([ToFloatTensor()])
