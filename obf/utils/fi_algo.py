"""Fixation Identification Algorithm."""
import numpy as np
import torch

DEBUG = False
COMPUTE_FEATURE_MEAN_STD = False

# Manually define a gaussian filter
GAUSSIAN_FILTER = torch.nn.Conv1d(1, 1, 3, bias=False, padding=1)
GAUSSIAN_FILTER.weight = torch.nn.Parameter(
    torch.tensor((0.27901, 0.44198, 0.27901)).reshape(1, 1, 3))


def euclidean_distance(s1, s2, reduction="mean"):
  """Calculate Euclidean distance between S1 and S2.

  Args:
    s1 (torch.tensor or np.array): The shape is (..., 2). It can have any kinds of dimension, but the last dimension should contain the x, y data.
    s2 (torch.tensor or np.array)): Similar to x
    reduction (str): reduction technique. If "mean", then return a single number. If "none", then return the same shape (without the last dimension).

  Returns:
    dist (torch.tensor or np.array)

  Raises:
    ValueError: if the reduction technique is unknown, raise value error.
  """
  assert s1.shape == s2.shape, "The two signals should have the same shape while calculating Euclidean distance."

  s1_x = s1[..., 0]
  s1_y = s1[..., 1]
  s2_x = s2[..., 0]
  s2_y = s2[..., 1]

  dist_square = (s1_x - s2_x)**2 + (s1_y - s2_y)**2

  if type(s1) is np.ndarray:
    dist = np.sqrt(dist_square)
  else:
    dist = dist_square.sqrt()

  if reduction == "none":
    return dist
  if reduction == "mean":
    return dist.mean()
  raise ValueError("Unknown reduction technique")


def ivt_fixations(data, threshold, min_len):
  """Run I-VT for fixation identification.

  Args:
    data (torch.tensor): (batch_size, seq_len, 2)
    threshold (float): maximum speed for a fixation. The unit is degree/interval
    min_len (float): the minimum length for a fixation. The unit is number of points
    
  Output:
    is_fix (torch.tensor): (batch_size, seq_len, S) boolean matrix. The value is 1 for gaze points that
      are identified as fixations.  
  """
  # Apply Gaussian filter for smoothing
  data = data.cpu().transpose(1, 2)  # (batch_size, 2, seq_len)
  data[:, 0:1, :] = GAUSSIAN_FILTER(data[:, 0:1, :])
  data[:, 1:2, :] = GAUSSIAN_FILTER(data[:, 1:2, :])
  data = data.transpose(1, 2).detach()  # (batch_size, seq_len, 2)

  # Find the velocity for each point
  data_l = data[:, :-1, :]  # (batch_size, seq_len-1, 2)
  data_r = data[:, 1:, :]

  # Cartesian velocity
  vel = torch.sqrt(
      (data_l[:, :, 0] - data_r[:, :, 0])**2 +
      (data_l[:, :, 1] - data_r[:, :, 1])**2)  # (batch_size, seq_len-1)

  # Reject high velocity points
  is_fix = vel <= threshold

  # Reject short fixations (usually noise)
  for b in range(is_fix.shape[0]):
    i = 0
    num_fixations = 0
    while i < is_fix.shape[1]:
      j = i
      while i < is_fix.shape[1] and is_fix[b, i]:
        i += 1

      if i - j < min_len:
        # This is not a fixation because it is too short
        is_fix[b, i:j + 1] = 0
      i += 1

  return is_fix.long()
