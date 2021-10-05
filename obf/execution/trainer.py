import os
import pdb
import random
import time
from functools import reduce

import numpy as np
import sklearn.metrics as metrics
import termcolor
import torch
import torch.nn as nn
import tqdm
from tensorboardX import SummaryWriter

from ..utils import fi_algo

_CE_CRITERION = nn.CrossEntropyLoss()


def _create_cl_data(data):
  """Create data batches for contrastive learning.

  Args:
    data (torch.tensor): (batch_size, sequence_length, 2)
    
  Outputs:
    x1 (torch.tensor): (batch_size, sequence_length_A, 2)
    x2 (torch.tensor): (batch_size, sequence_length_B, 2)
    y (torch.tensor): (batch_size, ) binary labels s.t. if x1 and x2 are from the 
      same document (sequence), it is 1; otherwise, it's 0
  """
  n, s, _ = data.shape
  assert n >= 2

  # Random input sequence length
  s1 = random.randrange(int(s * 0.2), int(s * 0.4))
  s2 = random.randrange(int(s * 0.2), int(s * 0.4))

  x1 = torch.zeros((n, s1, 2)).to(data.device)
  x2 = torch.zeros((n, s2, 2)).to(data.device)
  y = torch.zeros(n).to(data.device)

  for i in range(n):
    # get x1
    try:
      x1_start = random.randrange(0, s - s1)
    except:
      x1_start = 0
    x1[i, :, :] = data[i, x1_start:x1_start + s1, :]

    if random.random() > 0.5:
      # Get x2 from the same sequence
      j = i
      y[i] = 1
    else:
      # Get x2 from different sequence
      j = i
      y[i] = 0
      while j == i:
        j = random.randrange(0, n)

    try:
      x2_start = random.randrange(0, s - s2)
    except:
      x2_start = 0

    x2[i, :, :] = data[j, x2_start:x2_start + s2, :]

  return x1.float(), x2.float(), y.float()


class Trainer:

  def __init__(self, experiment, fi_config):
    self.experiment = experiment
    self.fi_config = fi_config

    self._parse_setting()

    self._setup_criterion()

  def _parse_setting(self):
    self.num_epochs = self.experiment["epochs"]
    self.input_seq_length = self.experiment["input seq length"]
    self.recon_seq_length = self.experiment["recon seq length"]
    self.pc_seq_length = self.experiment["pc seq length"]
    self.grad_max = self.experiment["grad norm"]  # maximum gradient for SGD
    self.init_lr = self.experiment["learning rate"]
    self.tasks = self.experiment["tasks"]
    self.use_cuda = self.experiment["cuda"] and torch.cuda.is_available()

    self.hidden_dim = self.experiment["hidden dim"]
    self.use_conv = self.experiment["use conv"]
    self.backbone_type = self.experiment["backbone type"]
    self.nlayers = self.experiment["number of layers"]
    self.epochs = self.experiment["epochs"]
    self.save_epochs = self.experiment["save epochs"]
    self.outdir = self.experiment["outdir"]

    # Setup output folder and files
    outdir = self.outdir
    os.makedirs(outdir, exist_ok=True)

    self.logname = "%d_%s" % (time.time(), self.backbone_type)
    self.logdir = os.path.join(outdir, self.logname)
    if os.path.exists(self.logdir):
      print("The self.logdir already existed! Ignore!")
    self.summary_writer = SummaryWriter(log_dir=self.logdir, flush_secs=10)

  def _setup_criterion(self):

    fi_weights = torch.tensor(
        [4.53, 0.18]).float()  # the actual inverse of ratio of these data

    if self.use_cuda:
      fi_weights = fi_weights.cuda()
    self.fi_criterion = nn.CrossEntropyLoss(fi_weights / fi_weights.sum())

    self.criterion = nn.MSELoss()

  def _setup_model_optimizer(self, models):
    self.models = models
    self.encoder, self.pc_decoder, self.fi_decoder, \
      self.cl_decoder, self.rc_decoder = self.models

    all_params = [list(_.parameters()) for _ in self.models]
    all_params = reduce(lambda x, y: x + y, all_params)

    for p in all_params:
      if p.dim() > 1:
        nn.init.xavier_uniform_(p)

    self.optimizer = torch.optim.Adam(all_params,
                                      lr=self.init_lr,
                                      weight_decay=0)
    self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                     step_size=max(
                                                         1,
                                                         int(self.epochs / 5)),
                                                     gamma=0.5)

  def train(self, models, train_loader, valid_loader):

    self._setup_model_optimizer(models)

    for epoch_id in range(self.epochs):
      self.run(train_loader, "train", epoch_id)

      with torch.no_grad():
        self.run(valid_loader, "valid", epoch_id)

      if epoch_id % self.save_epochs == 0 or epoch_id == self.epochs - 1:
        torch.save(self.encoder.module,
                   os.path.join(self.logdir, "encoder_%s.pt" % self.logname))
        torch.save(self.pc_decoder.module,
                   os.path.join(self.logdir, "pc_%s.pt" % self.logname))
        torch.save(self.fi_decoder.module,
                   os.path.join(self.logdir, "fi_%s.pt" % self.logname))
        torch.save(self.cl_decoder.module,
                   os.path.join(self.logdir, "cl_%s.pt" % self.logname))
        torch.save(self.rc_decoder.module,
                   os.path.join(self.logdir, "rc_%s.pt" % self.logname))
        # TODO: call run for training and validation
    return

  def _optimize_models(self, loss, models):
    assert torch.sum(torch.isnan(loss)) == 0
    assert torch.sum(torch.isinf(loss)) == 0

    loss.backward()
    for m in models:
      torch.nn.utils.clip_grad_norm_(m.parameters(), self.grad_max)
    self.optimizer.step()
    self.optimizer.zero_grad()

  def run(self, loader, mode="valid", epoch_id=0):
    pc_losses = []
    fi_losses = []
    cl_losses = []
    rc_losses = []

    cl_accs = []

    pc_dists = []
    rc_dists = []

    fi_all_labels = []
    fi_all_preds = []
    fi_all_probs = []

    encoder, pc_decoder, fi_decoder, cl_decoder, rc_decoder = self.models

    if mode == "train":
      (_.train() for _ in self.models)
      self.summary_writer.add_scalar("train/learning_rate",
                                     self.scheduler.get_last_lr()[-1], epoch_id)
    else:
      (_.eval() for _ in self.models)

    for x in tqdm.tqdm(loader):
      bs, s, _ = x.shape
      if self.use_cuda:
        x = x.cuda()

      # Cut the signal
      isl = random.randrange(
          int(self.input_seq_length / 2),
          self.input_seq_length)  # random input sequence length
      try:
        start_point = random.randrange(0, s - isl - self.pc_seq_length)
      except:
        start_point = 0
      inputs = x[:, start_point:start_point + isl, :]
      outcomes = x[:,
                   start_point + isl:start_point + isl + self.pc_seq_length, :]

      # Task 1: Predictive Coding
      if "pc" in self.tasks:
        embed = encoder(inputs)
        pc = pc_decoder(embed).reshape(outcomes.shape)
        pc_loss = self.criterion(outcomes, pc)
        pc_losses.append(pc_loss.item())

        pc_dists.append(
            fi_algo.euclidean_distance(outcomes, pc.detach()).cpu().numpy())

        if mode == "train":
          self._optimize_models(pc_loss, [encoder, pc_decoder])

      # Task 2: Fixation Identification
      if "fi" in self.tasks:
        fixations = fi_algo.ivt_fixations(
            inputs,
            threshold=self.fi_config["min_num_points_per_fixation"],
            min_len=self.fi_config["threshold_dist_per_interval"])

        fixations = fixations.to(inputs.device)
        embed = encoder(inputs)
        fi0 = fi_decoder(embed)
        fi = fi0.reshape(fi0.shape[0], -1, 2)[:, :fixations.shape[1], :]
        # (batch_size, seq_len, 2) shape

        fi_pred_np = fi.argmax(dim=2).detach().cpu().numpy().reshape(-1)
        fi_gt_np = fixations.cpu().numpy().reshape(-1)

        fi_all_labels += fi_gt_np.tolist()
        fi_all_preds += fi_pred_np.tolist()
        fi_all_probs += fi.softmax(
            dim=2)[:, :, 1].detach().cpu().numpy().reshape(-1).tolist()

        fi2 = fi.reshape(-1,
                         2)  # [batch_size * seq_len, 2] shape for prediction
        fixations2 = fixations.reshape(-1)  # Ground truth as a single vector

        sac_idx = fixations2 == 0  # saccade index

        if torch.sum(sac_idx) > 0 and torch.sum(sac_idx) < fixations2.shape[0]:
          # We have some saccades and fixations. Train the FI decoder.
          sac_loss = _CE_CRITERION(fi2[sac_idx, :], fixations2[sac_idx])
          fix_idx = torch.where(fixations2 == 1)[0]
          fix_idx_rand = np.random.choice(fix_idx.cpu().numpy(),
                                          size=len(sac_idx),
                                          replace=True)
          fix_loss = _CE_CRITERION(fi2[fix_idx_rand, :],
                                   fixations2[fix_idx_rand])

          # Avoid no saccade / no fixation bug
          sac_loss = sac_loss if not torch.isnan(sac_loss) else 0
          fix_loss = fix_loss if not torch.isnan(fix_loss) else 0

          fi_loss = sac_loss + fix_loss
          fi_losses.append(fi_loss.item())

          if mode == "train":
            self._optimize_models(fi_loss, [encoder, fi_decoder])

      # Task 3: Contrastive Learning: Judge from same document or not
      if "cl" in self.tasks and x.shape[0] > 2:
        x1, x2, cl_y = _create_cl_data(x)
        e1 = encoder(x1)
        e2 = encoder(x2)

        if self.backbone_type in ["rnn", "gru"]:
          # Transpose for RNNs
          pass
        elif self.backbone_type == "lstm":
          # Concat to (batch_size, num_layers * 2, num_channels)
          e1 = torch.cat(e1, dim=1)
          e2 = torch.cat(e2, dim=1)
        else:
          raise ValueError("Unknown backbone type: %s" % self.backbone_type)

        embed = torch.abs(e1 - e2)
        y_pred = cl_decoder(embed)
        cl_loss = _CE_CRITERION(y_pred, cl_y.long())
        cl_losses.append(cl_loss.item())

        acc = (y_pred.argmax(dim=1) == cl_y).float().mean()
        cl_accs.append(acc.item())

        if mode == "train":
          self._optimize_models(cl_loss, [encoder, cl_decoder])

      # Task 4: Reconstruction (AE)
      if "rc" in self.tasks:
        isl2 = random.randrange(
            int(60), self.recon_seq_length)  # random input sequence length
        try:
          start_point = random.randrange(0, s - isl2 - self.pc_seq_length)
        except:
          start_point = 0
        inputs_short = x[:, start_point:start_point + isl2, :]

        embed = encoder(inputs_short)
        rc = rc_decoder(embed)  # (batch_size, seq_len, 2)
        rc = rc.reshape(rc.shape[0], -1, 2)
        rc_loss = self.criterion(inputs_short, rc[:, :isl2, :])
        rc_losses.append(rc_loss.item())
        rc_dists.append(
            fi_algo.euclidean_distance(inputs_short,
                                       rc[:, :isl2, :].detach()).cpu().numpy())

        if mode == "train":
          self._optimize_models(rc_loss, [encoder, rc_decoder])

    fi_auc = metrics.roc_auc_score(fi_all_labels, fi_all_probs)
    fi_f1 = metrics.f1_score(fi_all_labels, fi_all_preds, average="weighted")

    msg = "(%s)\tEpoch %d\tpc-Loss %.2f\tfi-Loss: %.2f\tcl-loss %.2f\trc-loss %.2f" % (
        mode, epoch_id, np.mean(pc_losses), np.mean(fi_losses),
        np.mean(cl_losses), np.mean(rc_losses))

    msg += "\tcl-acc: %.2f\tfi-f1: %.2f\tfi-auc: %.2f" % (np.mean(cl_accs),
                                                          fi_f1, fi_auc)
    msg += "\tpc-dist: %.2f\trc-dist: %.2f" % (np.mean(pc_dists),
                                               np.mean(rc_dists))

    color = "yellow" if mode == "train" else "green"
    print(termcolor.colored(msg, color=color))

    self.summary_writer.add_scalar(mode + "/loss_pc", np.mean(pc_losses),
                                   epoch_id)
    self.summary_writer.add_scalar(mode + "/loss_fi", np.mean(fi_losses),
                                   epoch_id)
    self.summary_writer.add_scalar(mode + "/loss_cl", np.mean(cl_losses),
                                   epoch_id)
    self.summary_writer.add_scalar(mode + "/loss_rc", np.mean(rc_losses),
                                   epoch_id)

    self.summary_writer.add_scalar(mode + "/cl_acc", np.mean(cl_accs), epoch_id)
    self.summary_writer.add_scalar(mode + "/fi_f1", fi_f1, epoch_id)
    self.summary_writer.add_scalar(mode + "/fi_auc", fi_auc, epoch_id)
    self.summary_writer.add_scalar(mode + "/pc_dist", np.mean(pc_dists),
                                   epoch_id)
    self.summary_writer.add_scalar(mode + "/rc_dist", np.mean(rc_dists),
                                   epoch_id)

    self.summary_writer.file_writer.flush()

    if "cl" in self.tasks:
      print("Confusion Matrix: Contrastive Learning")
      print(
          metrics.confusion_matrix(cl_y.detach().cpu().numpy(),
                                   y_pred.argmax(dim=1).detach().cpu().numpy()))

    if "fi" in self.tasks:
      print("Confusion Matrix: Fixation Identification")
      print(metrics.confusion_matrix(fi_gt_np, fi_pred_np, normalize="true")
           )  # F1-score, regarding saccade as Positive (rare case)

    if mode == "train":
      self.scheduler.step()
