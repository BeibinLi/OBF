import pandas as pd
import numpy as np
import scipy
import scipy.io
import scipy.interpolate
import matplotlib.pyplot as plt  # for DEBUG

import os
import glob
import pdb
import copy

INPUT_FILE = "./coutrot/coutrot_database2.mat"
OUTPUT_DIR = "./coutrot/clean_data_2/"
FREQUENCY = 25  # input frequence (Hz)

TARGET_FREQUENCY = 60  # 60 Hz

# 22 inch screen (47.6 x 26.8 cm),
# participant is 56 cm away
# Screen resolution (1280, 1024)
# So, 1 visual degree is about 0.97748 cm on screen
# So, visual degree is 26 pixels per degree
VISUAL_DEGREES = 26  #  pixels per visual angle

# Note: 26 pixels per degree for coutrot dataset2

SCREEN_CENTER = (1280 // 2, 1024 // 2)

OFF_SCREEN_MARGIN = 10  # number of degrees allowed to be off-screen

VIZ_DEBUG = False

if not os.path.exists(OUTPUT_DIR):
  os.mkdir(OUTPUT_DIR)

NA_FLAG = np.nan

mat_data = scipy.io.loadmat(INPUT_FILE)
mat_data = mat_data["Coutrot_Database2"]

# X is in a complex nested array. We need DFS to extract the information.
# We know that actual data is in 3 dimension

fronts = [mat_data]

# There are 15 videos, and each is in 2 auditorial conditions
# So, there are total 15 * 2 = 30 stimuli
actual_data = []

count = 0
while len(fronts):

  f = fronts.pop(0)

  if type(f) is not np.ndarray and type(f) is not np.void:
    continue

  # If it is 3D, it is the actual data
  if len(f.shape) == 3:  # 3d
    actual_data.append(copy.deepcopy(f))
    continue

  # Else, we need every elements in the array "f"
  [fronts.append(_) for _ in f]

  print("Count", count)
  [print(_.shape) for _ in f]

  count += 1

for i, data in enumerate(actual_data):
  print("begin process", i)

  data = data.astype(float)
  # Screen Center to Zero
  data[0, :, :] = data[0, :, :] - SCREEN_CENTER[0]
  data[1, :, :] = data[1, :, :] - SCREEN_CENTER[1]

  # Pixel to Visual Angle
  data /= VISUAL_DEGREES

  stimulus_time = np.round(data.shape[1] / FREQUENCY * 1000)  # milisecond

  # Extract Data, and Re-sample (interpolate) to 60 Hz
  input_time_signal = np.arange(0, stimulus_time, step=1000 / FREQUENCY)
  target_time_signal = np.arange(0,
                                 input_time_signal.max(),
                                 step=1000 / TARGET_FREQUENCY)

  for pid in range(data.shape[2]):
    x = scipy.interpolate.interp1d(x=input_time_signal, y=data[0, :, pid])
    y = scipy.interpolate.interp1d(x=input_time_signal, y=data[1, :, pid])
    x_signal = x(target_time_signal)
    y_signal = y(target_time_signal)

    # Create Output Data Frame
    df_data = {"time": target_time_signal, "x": x_signal, "y": y_signal}

    df = pd.DataFrame(df_data)

    # Off-Screen to NaN
    x_lim = SCREEN_CENTER[
        0] / VISUAL_DEGREES + OFF_SCREEN_MARGIN * VISUAL_DEGREES
    df.loc[df.x < -x_lim, "x"] = NA_FLAG
    df.loc[df.x > x_lim, "x"] = NA_FLAG

    y_lim = SCREEN_CENTER[
        1] / VISUAL_DEGREES + OFF_SCREEN_MARGIN * VISUAL_DEGREES
    df.loc[df.y < -y_lim, "y"] = NA_FLAG
    df.loc[df.y > y_lim, "y"] = NA_FLAG

    df = df.interpolate()  # Interpret missing data except heading and trailing
    df = df[~(df.x.isna() |
              df.y.isna())]  # remove the heading and trailing with NAs

    filename = "stimulus_%03d_pid_%02d.txt" % (i, pid)
    if VIZ_DEBUG:
      plt.plot(df["time"], df["x"], label="x")
      plt.plot(df["time"], df["y"], label="y")
      plt.legend()
      plt.title(filename)
      plt.show()

    ofname = os.path.join(OUTPUT_DIR, os.path.basename(filename))
    df.astype(np.float32).to_csv(ofname, header=None, index=None)

    x = np.loadtxt(ofname, dtype=np.float32, delimiter=",")
