import numpy as np


def top_k_accuracy(probs, labels, k=5):
  """Calculate the Top-k Accuarcy.
  
  Args:
    probs (np.array): [batch_size, num_class] matrix.
    labels (np.array): [batch_size] matrix.

  Returns:
    accuracy (float): top k accuracy.
  """
  correct = 0
  for i in range(probs.shape[0]):
    p_ = probs[i, :]
    top_k = np.argsort(p_)[-k:]

    correct += labels[i] in top_k

  return correct / probs.shape[0]
