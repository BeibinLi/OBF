import pandas as pd
import numpy as np
import scipy
import scipy.interpolate
import matplotlib.pyplot as plt  # for DEBUG

import os
import glob
import pdb

INPUT_DIR = "./FixaTons/MIT1003/SCANPATHS"
OUTPUT_DIR = "./FixaTons/MIT1003/clean_data/"

TARGET_FREQUENCY = 60  # 60 Hz

# 19 inch screen (with 1.25 aspect ratio) is 14.84 x 11.87 inch
# participant is 2 feet (i.e. 24 inch) away
# Screen resolution (1280, 1024)
# So, 1 visual degree is about 0.42 inch on screen (note: tan(1) = ? / 24 = 0.017455)
# 0.42 inch / 11.87 inch = ? pixels / 1024 pixels
# So, visual degree is ? pixels per degree
VISUAL_DEGREES = 36.23  #  pixels per visual angle

# Note: previously I assume the visual degree is 15 pixels

SCREEN_CENTER = (1280 // 2, 1024 // 2)

# RAW_TIME_UNIT = 1e-3 # 1e-3 ms in the original format

OFF_SCREEN_MARGIN = 10  # number of degrees allowed to be off-screen

VIZ_DEBUG = False
NA_FLAG = -180

if not os.path.exists(OUTPUT_DIR):
  os.mkdir(OUTPUT_DIR)

users = set()
for stim_dir in glob.glob(os.path.join(INPUT_DIR, "*")):
  if not os.path.isdir(stim_dir):
    continue

  for filename in glob.glob(os.path.join(stim_dir, "*")):
    if not os.path.isfile(filename):
      continue

    users.add(os.path.basename(filename))
    print("begin process", filename)

    data = pd.read_csv(filename,
                       names=["x", "y", "t_start", "t_end", ""],
                       sep=" ")

    # Screen Center to Zero
    data["x"] = data["x"] - SCREEN_CENTER[0]
    data["y"] = data["y"] - SCREEN_CENTER[1]

    # Pixel to Visual Angle
    data["x"] /= VISUAL_DEGREES
    data["y"] /= VISUAL_DEGREES

    # Create an array to store the informations
    raw_x = [SCREEN_CENTER[0] / VISUAL_DEGREES]
    raw_y = [SCREEN_CENTER[1] / VISUAL_DEGREES]
    raw_t = [0]  # Assume the start point is always the center
    for i in range(data.shape[0]):
      # Each row will be used twice (Start and End) of time
      raw_x.append(data.loc[i, "x"])
      raw_x.append(data.loc[i, "x"])
      raw_y.append(data.loc[i, "y"])
      raw_y.append(data.loc[i, "y"])
      raw_t.append(data.loc[i, "t_start"] * 1000)  # "sec" to "ms"
      raw_t.append(data.loc[i, "t_end"] * 1000)

    x = scipy.interpolate.interp1d(x=raw_t, y=raw_x)
    y = scipy.interpolate.interp1d(x=raw_t, y=raw_y)
    time_signal = np.arange(0,
                            data["t_end"].max() * 1000,
                            step=1000 / TARGET_FREQUENCY)
    x_signal = x(time_signal)
    y_signal = y(time_signal)

    # pdb.set_trace()
    # Create Output Data Frame
    df_data = {"time": time_signal, "x": x_signal, "y": y_signal}

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

    df.loc[df.x.isna(), "x"] = NA_FLAG
    df.loc[df.y.isna(), "y"] = NA_FLAG

    if VIZ_DEBUG:
      plt.plot(df["time"], df["x"], label="x")
      plt.plot(df["time"], df["y"], label="y")
      plt.legend()
      plt.title(filename)
      plt.show()

    txtname = "%s_user_%s.txt" % (os.path.basename(stim_dir),
                                  os.path.basename(filename))

    # if is_train(txtname):
    #     dirname = "train"
    # elif is_valid(txtname):
    #     dirname = "valid"s
    # else:
    #     dirname = "test"

    ofname = os.path.join(OUTPUT_DIR, txtname)
    df.to_csv(ofname, header=None, index=None)

users = sorted(list(users))
print("Users:", users)
