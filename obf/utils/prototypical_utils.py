# coding=utf-8
# https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch/blob/master/src/prototypical_loss.py
import torch
import numpy as np
from torch.nn import functional as F
from torch.nn.modules import Module
import sklearn.metrics
import pdb


class PrototypicalBatchSampler(object):
  '''
  Source: https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch/blob/master/src/prototypical_batch_sampler.py
    
    
  PrototypicalBatchSampler: yield a batch of indexes at each iteration.
  Indexes are calculated by keeping in account 'classes_per_it' and 'num_samples',
  In fact at every iteration the batch indexes will refer to  'num_support' + 'num_query' samples
  for 'classes_per_it' random classes.
  __len__ returns the number of episodes per epoch (same as 'self.iterations').
  '''

  def __init__(self, labels, classes_per_it, num_samples, iterations):
    '''
    Initialize the PrototypicalBatchSampler object
    Args:
    - labels: an iterable containing all the labels for the current dataset
    samples indexes will be infered from this iterable.
    - classes_per_it: number of random classes for each iteration
    - num_samples: number of samples for each iteration for each class (support + query)
    - iterations: number of iterations (episodes) per epoch
    '''
    super(PrototypicalBatchSampler, self).__init__()
    self.labels = labels
    self.classes_per_it = classes_per_it
    self.sample_per_class = num_samples
    self.iterations = iterations

    self.classes, self.counts = np.unique(self.labels, return_counts=True)
    self.classes = torch.LongTensor(self.classes)

    # create a matrix, indexes, of dim: classes X max(elements per class)
    # fill it with nans
    # for every class c, fill the relative row with the indices samples belonging to c
    # in numel_per_class we store the number of samples for each class/row
    self.idxs = range(len(self.labels))
    self.indexes = np.empty(
        (len(self.classes), max(self.counts)), dtype=int) * np.nan
    self.indexes = torch.Tensor(self.indexes)
    self.numel_per_class = torch.zeros_like(self.classes)
    for idx, label in enumerate(self.labels):
      label_idx = np.argwhere(self.classes == label).item()
      self.indexes[label_idx,
                   np.where(np.isnan(self.indexes[label_idx]))[0][0]] = idx
      self.numel_per_class[label_idx] += 1

  def __iter__(self):
    '''
    yield a batch of indexes
    '''
    spc = self.sample_per_class
    cpi = self.classes_per_it

    for it in range(self.iterations):
      batch_size = spc * cpi
      batch = torch.LongTensor(batch_size)
      c_idxs = torch.randperm(len(self.classes))[:cpi]

      # pdb.set_trace()
      for i, c in enumerate(self.classes[c_idxs]):
        s = slice(i * spc, (i + 1) * spc)
        # FIXME when torch.argwhere will exists
        label_idx = torch.arange(len(
            self.classes)).long()[self.classes == c].item()
        sample_idxs = torch.randperm(self.numel_per_class[label_idx])[:spc]
        batch[s] = self.indexes[label_idx][sample_idxs]
      batch = batch[torch.randperm(len(batch))]

      # print("Batch (sampler):", batch.detach().cpu().numpy().tolist())
      yield batch


class PrototypicalLoss(Module):
  '''
  Loss class deriving from Module for the prototypical loss function defined below
  '''

  def __init__(self, n_support):
    super(PrototypicalLoss, self).__init__()
    self.n_support = n_support

  def forward(self, inputs, targets):
    return prototypical_loss(inputs, targets, self.n_support)


def euclidean_dist(x, y):
  '''
  Compute euclidean distance between two tensors
  '''
  # x: N x D
  # y: M x D
  n = x.size(0)
  m = y.size(0)
  d = x.size(1)
  if d != y.size(1):
    raise Exception

  # assert d == 64

  x = x.unsqueeze(1).expand(n, m, d)
  y = y.unsqueeze(0).expand(n, m, d)

  return torch.pow(x - y, 2).sum(2)


def prototypical_loss(inputs, targets, n_support):
  '''
  Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py
  Compute the barycentres by averaging the features of n_support
  samples for each class in targets, computes then the distances from each
  samples' features to each one of the barycentres, computes the
  log_probability for each n_query samples for each one of the current
  classes, of appartaining to a class c, loss and accuracy are then computed
  and returned
  Args:
  - inputs: the model output for a batch of samples
  - targets: ground truth for the above batch of samples
  - n_support: number of samples to keep in account when computing
    barycentres, for each one of the current classes
  '''

  def supp_idxs(c):
    return targets.eq(c).nonzero()[:n_support].squeeze(1)

  classes = torch.unique(targets)
  n_classes = len(classes)
  n_query = targets.eq(classes[0].item()).sum().item() - n_support

  support_idxs = list(map(supp_idxs, classes))

  prototypes = torch.stack(
      [inputs[idx_list].mean(0) for idx_list in support_idxs])
  query_idxs = torch.stack(
      list(map(lambda c: targets.eq(c).nonzero()[n_support:],
               classes))).view(-1)

  query_samples = inputs[query_idxs]
  dists = euclidean_dist(query_samples, prototypes)

  log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)
  # probs = F.softmax(-dists, dim=1).view(n_classes, n_query, -1)

  targets_inds = torch.arange(0, n_classes).cuda()
  targets_inds = targets_inds.view(n_classes, 1, 1)
  targets_inds = targets_inds.expand(n_classes, n_query, 1).long()

  loss_val = -log_p_y.gather(2, targets_inds).squeeze().view(-1).mean()
  # loss_val = -probs.gather(2, targets_inds).squeeze().view(-1).mean()
  # # TODO: remove. Here add the distance to the correct class
  # loss_val += dists.view(n_classes, n_query, -1).gather(2, targets_inds).mean()

  _, y_hat = log_p_y.max(2)
  # acc_val = y_hat.squeeze().eq(targets_inds.squeeze()).float().mean()
  # acc_val = y_hat.eq(targets_inds.squeeze()).float().mean()
  correct = targets_inds.reshape(-1) == y_hat.reshape(-1)
  acc_val = torch.mean(correct.float())

  return loss_val, acc_val


def inference_loss(supports, queries, support_labels, targets):
  """

  Beibin: 08/29/2020
  It is similar to protypical loss, and this function is more flexible because it allows 
  different number of supports/queries for each class.

  k: number of classes
  d: embedding size
  n: total number of support samples
  m: total number of query samples

  Args:
    supports (torch.tensor): [n, d] embedding from the support sets
    queries (torch.tensor): [m, d] embedding from the support sets
    support_labels (torch.tensor): [m] GT from the support sets
    targets (torch.tensor): [m] GT from the query (aka test) sets

  Outputs:
    # loss (torch.tensor): proto loss
    acc (torch.tensor): testing accuracy
  """

  classes = np.unique(support_labels.cpu().numpy()).tolist()

  # Transform the labels s.t. they are from 0 - k
  support_labels = [
      classes.index(_) for _ in support_labels.cpu().numpy().tolist()
  ]
  targets = [classes.index(_) for _ in targets.cpu().numpy().tolist()]
  classes = np.unique(support_labels)
  support_labels = torch.tensor(support_labels).long().cuda()
  targets = torch.tensor(targets).long().cuda()

  centers = [None] * len(
      classes)  # key: class id. value: embedding center for the class
  for cls in classes:
    ctr = torch.mean(supports[support_labels == cls], dim=0).reshape(1, -1)
    centers[cls] = ctr

  centers_tensor = torch.cat(centers, dim=0).cuda()
  dists = euclidean_dist(queries, centers_tensor)

  # m, d = queries.shape
  # log_p_y = F.log_softmax(-dists, dim=1).reshape(len(classes), m, -1)

  _, y_hat = dists.max(1)

  correct = targets.reshape(-1) == y_hat.reshape(-1)
  acc_val = torch.mean(correct.float())

  # pdb.set_trace()
  f1 = sklearn.metrics.f1_score(targets.cpu().numpy(),
                                y_hat.detach().cpu().numpy())

  # pdb.set_trace()

  auc = sklearn.metrics.roc_auc_score(
      targets.cpu().numpy(),
      torch.softmax(dists, dim=1)[:, 0].detach().cpu().numpy())

  print(
      sklearn.metrics.confusion_matrix(targets.cpu().numpy(),
                                       y_hat.detach().cpu().numpy()))
  print("AUC:", auc)
  return acc_val, f1, auc
