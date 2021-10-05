"""Configuration utils for loading and checking config json files."""
import json
import os
import math


def _check_config(config):
  """Check the configuration is robust.

  Args:
      config (dict): the configuration in dict.
    
  Raises:
    AssertionError: if the configuration violates some of rules.
  """

  # Check Dataset definition
  if "pretrain data setting" in config:
    data_loader = config["pretrain data setting"]
    assert data_loader["batch size"] > 0

    for dataset_info in data_loader["datasets"]:
      assert os.path.exists(dataset_info["path"])
      assert dataset_info["signal length"] > 0

  # Check Experiment setting
  if "experiment" in config:
    experiment = config["experiment"]

    assert experiment["learning rate"] > 0
    # TODO: add other checks


def _process_fixation_identification_setting(setting):

  if "fixation identification setting" not in setting:
    return setting

  fi = setting["fixation identification setting"]

  velocity_threshold = fi["velocity threshold (d/s)"]  # degree per second
  min_fixation_length = fi["min fixation length (ms)"]  # milisecond

  if "frequency" in fi:
    frequency = fi["frequency"]  # hz
  else:
    frequency = 60  # hz

  fi["min_num_points_per_fixation"] = int(
      math.ceil(min_fixation_length / (1000 / frequency)))
  fi["threshold_dist_per_interval"] = velocity_threshold / frequency

  setting["fixation identification setting"] = fi
  return setting


def load_config(config_filename="configs/setting.json"):
  """Load Config file and check the robustness.

  Args:
    config_filename (str): the filename location.

  Returns:
    config (dict): configuration in a dictionary
  """
  import glob
  print(glob.glob("config/*"))
  # Load config
  f = open(config_filename, "r")
  config = json.load(f)
  f.close()

  # Check config
  _check_config(config)

  # Post-process config
  config = _process_fixation_identification_setting(config)

  return config
