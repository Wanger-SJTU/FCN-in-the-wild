import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)




def add_full_path():
  model_path = osp.join(this_dir, 'FCN')
  add_path(model_path)

  data_path = osp.join(this_dir, 'data')
  add_path(data_path)

  util_path = osp.join(this_dir, 'utils')
  add_path(util_path)