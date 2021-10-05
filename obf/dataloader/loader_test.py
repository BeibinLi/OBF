"""Test for all dataloaders, including gaze loader, mit loader."""
import json
import unittest

from ..utils.config import load_config
from .gaze_loader import get_pretrain_data_loader
from .mit_loader import MITProtoDataset


class TestConcatLoader(unittest.TestCase):

  def test_concat_loader(self):
    setting = load_config()
    loader = get_pretrain_data_loader("valid", setting["pretrain data setting"])
    self.assertGreater(len(loader), 0)

    # Try to load 5 batches of signals
    count = 0
    for x in loader:
      print(x.shape)
      self.assertNotEqual(x.shape, None)
      count += 1
      if count == 5:
        break


class TestMITProtoLoader(unittest.TestCase):

  def test_mit_proto(self):
    ds = MITProtoDataset(folder_name="sample_data/FixaTons/MIT1003/clean_data/",
                         mode="test",
                         shot=3,
                         way=10)
    self.assertNotEqual(ds, None)


if __name__ == '__main__':
  unittest.main()
