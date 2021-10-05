import pandas as pd
import numpy as np
import scipy
import scipy.interpolate
import matplotlib.pyplot as plt  # for DEBUG

import os
import glob
import pdb

INPUT_DIR = "./moscow/filtered_gaze_data/"
OUTPUT_DIR = "./moscow/clean_data/"

FREQUENCY = 500  # 500 Hz
TARGET_FREQUENCY = 60  # 60 Hz

VISUAL_DEGREES = 15  # 15 pixels per visual angle

SCREEN_CENTER = (1920 // 2, 1080 // 2)

RAW_TIME_UNIT = 1e-3  # 1e-3 ms in the original format

OFF_SCREEN_MARGIN = 10  # number of degrees allowed to be off-screen

VIZ_DEBUG = True
NA_FLAG = -180

if not os.path.exists(OUTPUT_DIR):
  os.mkdir(OUTPUT_DIR)

for filename in glob.glob(os.path.join(INPUT_DIR, "*.txt")):
  print("begin process", filename)

  data = pd.read_csv(filename, names=["t", "lx", "ly", "rx", "ry"], sep="\t")

  # Average for binocular
  data["x"] = (data["lx"] + data["rx"]) / 2
  data["y"] = (data["ly"] + data["ry"]) / 2

  # Screen Center to Zero
  data["x"] = data["x"] - SCREEN_CENTER[0]
  data["y"] = data["y"] - SCREEN_CENTER[1]

  # Pixel to Visual Angle
  data["x"] /= VISUAL_DEGREES
  data["y"] /= VISUAL_DEGREES

  # Handle Time
  data["time"] = (data["t"] - data["t"][0]) * RAW_TIME_UNIT

  # Extract Data, and Re-sample (interpolate) to 60 Hz
  data = data[["time", "x", "y"]]
  num = data["time"].tolist()[-1] / 1000 * TARGET_FREQUENCY
  num = int(np.ceil(num))

  x = scipy.interpolate.interp1d(x=data["time"], y=data["x"])
  y = scipy.interpolate.interp1d(x=data["time"], y=data["y"])
  time_signal = np.arange(0,
                          data["time"].tolist()[-1],
                          step=1000 / TARGET_FREQUENCY)
  x_signal = x(time_signal)
  y_signal = y(time_signal)

  # Create Output Data Frame
  df_data = {"time": time_signal, "x": x_signal, "y": y_signal}

  df = pd.DataFrame(df_data)

  # Off-Screen to NaN
  x_lim = SCREEN_CENTER[0] / VISUAL_DEGREES + OFF_SCREEN_MARGIN * VISUAL_DEGREES
  df.loc[df.x < -x_lim, "x"] = NA_FLAG
  df.loc[df.x > x_lim, "x"] = NA_FLAG

  y_lim = SCREEN_CENTER[1] / VISUAL_DEGREES + OFF_SCREEN_MARGIN * VISUAL_DEGREES
  df.loc[df.y < -y_lim, "y"] = NA_FLAG
  df.loc[df.y > y_lim, "y"] = NA_FLAG

  df.loc[df.x.isna(), "x"] = NA_FLAG
  df.loc[df.y.isna(), "y"] = NA_FLAG

  if VIZ_DEBUG:
    plt.plot(df["time"], df["x"], label="x")
    plt.plot(df["time"], df["y"], label="y")
    plt.legend()
    plt.title(filename)
    plt.show()

  ofname = os.path.join(OUTPUT_DIR, os.path.basename(filename))
  df.to_csv(ofname, header=None, index=None)
