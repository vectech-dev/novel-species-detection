'''Holds configuration information'''
import os.path
import json

class ConfigLoader:
  """ Loads key-value pairs from given json file as attributes """
  def __init__(self, fpath):
    self.fpath = fpath
    with open(fpath, 'r') as f:
      d = json.load(f)
    for k,v in d.items():
      setattr(self, k, v)

class UpdateableConfigLoader(ConfigLoader):
  """ Adds ability to update keys in the config file """
  def update(self, k, v, append=False):
    with open(self.fpath, 'r') as f:
      d = json.load(f)
    if append:
      if isinstance(d.get(k, None), list):
        d[k].append(v)
      else:
        raise ValueError(f"Unable to append to config at {k} because it is not iterable ({type(d.get(k, None))})")
      arr = getattr(self, k)
      arr.append(v)
      setattr(self, k, arr)
    else:
      d[k] = v
      setattr(self, k, v)
    with open(self.fpath, 'w') as f:
      json.dump(d, f, indent=2)

class UnknownConfigLoader(UpdateableConfigLoader):
  """ Overload constructor to use default path of config.json in the tier2 project """
  CONFIG_PATH = os.path.join(
      os.path.dirname(
      os.path.realpath(
        __file__)), 'config.json')

  def __init__(self):
    super().__init__(UnknownConfigLoader.CONFIG_PATH)
  
  
    