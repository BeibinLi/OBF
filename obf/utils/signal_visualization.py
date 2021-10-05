import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

import pdb


def show_signal(signal,
                screen_width=1920 / 15,
                screen_height=1080 / 15,
                resolution_magnifier=10,
                title=""):
  """
  
  
  Args:
    signal: 2D signal
    screen_width: unit is visual degrees
    screen_height: unit is visual degrees
    resolution_magnifier (int): magnify visualization 
    
  """
  m_ = resolution_magnifier  # rename

  w_ = 5  # plot width (scatter size)

  # background image: * 2 for off-screen gazes
  img = np.zeros((int(screen_height * 2 * m_), int(screen_width * 2 * m_), 3),
                 dtype=np.float32)

  for x, y in signal:
    xi = int(x * m_ + screen_width * m_)
    yi = int(y * m_ + screen_height * m_)

    try:
      img[yi - w_:yi + w_, xi - w_:xi + w_, :] += 0.1
    except Exception as e:
      print(e)

  img = np.clip(img, 0, 1)
  plt.imshow(img)

  plt.title(title)
  plt.show()


def xy_t_signal(signal, title=""):
  sl = signal.shape[0]
  plt.plot(list(range(sl)), signal[:, 0], label="x")
  plt.plot(list(range(sl)), signal[:, 1], label="y")
  plt.legend()
  plt.title(title)
  plt.show()

  return


def xy_t_fix_signal(signal, is_fix, title=""):
  sl = signal.shape[0]
  plt.plot(list(range(sl)), signal[:, 0], label="x")
  plt.plot(list(range(sl)), signal[:, 1], label="y")

  plt.plot(list(range(sl - 1)), is_fix, label="fixation")

  plt.legend()
  plt.title(title)
  plt.show()

  return


def viz_pred_code(inputs, outcomes, preds, title):
  """
  
  Args:
    inputs (torch.tensor): the input data s.t. sequence with shape 
      (batch_size, sequence_length, 2)
    outcomes (torch.tensor): the actual following sequence with shape
      (batch_size, sequence_length, 2)
    preds (torch.tensor): predicted following sequence with shape
      (batch_size, sequence_length, 2)
    title (str): visualization title
    
  """

  n1 = inputs.shape[1]
  n2 = outcomes.shape[1]

  # The input sequence
  plt.plot(list(range(n1)),
           inputs[0, :, 0].cpu().numpy(),
           label="x",
           color="orange")
  plt.plot(list(range(n1)),
           inputs[0, :, 1].cpu().numpy(),
           label="y",
           color="blue")

  # The acutal sequence
  plt.scatter(list(range(n1, n1 + n2)),
              outcomes[0, :, 0].cpu().numpy(),
              marker=".",
              alpha=0.5,
              label="x next",
              color="orange")
  plt.scatter(list(range(n1, n1 + n2)),
              outcomes[0, :, 1].cpu().numpy(),
              marker=".",
              alpha=0.5,
              label="y next",
              color="blue")

  # The pred sequence
  plt.scatter(list(range(n1, n1 + n2)),
              preds[0, :, 0].detach().cpu().numpy(),
              marker="x",
              alpha=0.5,
              label="x pred",
              color="red")
  plt.scatter(list(range(n1, n1 + n2)),
              preds[0, :, 1].detach().cpu().numpy(),
              marker="x",
              alpha=0.5,
              label="y pred",
              color="green")

  plt.title(title)
  plt.legend()
  plt.show()


def viz_recon(inputs, recons, title):
  """
  
  Args:
    inputs (torch.tensor): the input data s.t. sequence with shape 
      (batch_size, sequence_length, 2)
    preds (torch.tensor): reconstrution with the same shape
      (batch_size, sequence_length, 2)
    title (str): visualization title
  """

  n1 = inputs.shape[1]

  # The input sequence
  plt.plot(list(range(n1)),
           inputs[0, :, 0].cpu().numpy(),
           label="x",
           color="orange",
           alpha=0.8)
  plt.plot(list(range(n1)),
           inputs[0, :, 1].cpu().numpy(),
           label="y",
           color="blue",
           alpha=0.8)

  # The re-construction sequence
  plt.plot(list(range(n1)),
           recons[0, :n1, 0].cpu().numpy(),
           '--',
           label="x'",
           color="orange",
           alpha=0.8)
  plt.plot(list(range(n1)),
           recons[0, :n1, 1].cpu().numpy(),
           '--',
           label="y'",
           color="blue",
           alpha=0.8)

  plt.title(title)
  plt.legend()
  plt.show()


def viz_fi_signal(inputs, fixations, preds, title=""):
  """
  Suppose there are k features
  
  Args:
    fi_labels (torch.tensor): real labels
    fi_preds (torch.tensor): predictions
  """

  n1 = inputs.shape[1]
  max_val = inputs.abs().max().item()

  # The input sequence
  plt.plot(list(range(n1)),
           inputs[0, :, 0].cpu().numpy(),
           label="x",
           color="orange",
           alpha=0.8)
  plt.plot(list(range(n1)),
           inputs[0, :, 1].cpu().numpy(),
           label="y",
           color="blue",
           alpha=0.8)

  # The Ground truth sequence
  saccade_gt_time = np.where(
      fixations[0, :n1].cpu().numpy().reshape(-1) == 0)[0]
  plt.scatter(saccade_gt_time,
              saccade_gt_time * 0,
              label="GT Sac",
              color="green",
              alpha=0.2)

  # The Predictions
  saccade_pred_time = np.where(preds[0, :n1, :].argmax(
      dim=1).detach().cpu().numpy().reshape(-1) == 0)[0]
  plt.scatter(saccade_pred_time,
              saccade_pred_time * 0 + max_val * 0.1,
              label="Pred Sac",
              color="red",
              alpha=0.2)  # offset the y value for plotting

  plt.title(title)
  plt.legend()
  plt.show()


def viz_fi(fi_labels, fi_preds, title=""):
  """
  Suppose there are k features
  
  Args:
    fi_labels (torch.tensor): real labels
    fi_preds (torch.tensor): predictions
  """

  k = fi_labels.shape[1]  # number of features

  r = int(np.sqrt(k))  # number of subplot rows
  c = int(np.ceil(k / r))  # number of subplot columns

  assert r * c >= k

  fig = plt.figure()
  for i in range(k):
    plt.subplot(r, c, i + 1)
    plt.scatter(fi_labels[:, i].cpu().numpy(),
                fi_preds[:, i].detach().cpu().numpy(),
                alpha=0.8)
    plt.xlabel("real")
    plt.ylabel("pred")

    corr, _ = pearsonr(fi_labels[:, i].cpu().numpy(),
                       fi_preds[:, i].detach().cpu().numpy())

    plt.title("%d: r=%.2f" % (i, corr))
    plt.gca().set_aspect('equal', 'box')

  fig.tight_layout()
  fig.suptitle(title)
  plt.show()
