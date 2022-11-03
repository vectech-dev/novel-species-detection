import os
from typing import Dict, Any
import hashlib
import json
import importlib.util
from unknown.utils.paths import ROOT_DIR
from unknown.config.config import UnknownConfigLoader
cfg = UnknownConfigLoader()

def read_config_path_from_env(env_key: str):
  fpath = os.environ.get(env_key, None)
  if fpath is None:
    raise ValueError(f'Could not find environment variable "{env_key}". Refer to README for more details.')
  if not os.path.exists(fpath) and not os.path.isabs(fpath):
    fpath = os.path.join(ROOT_DIR, fpath)
  if not os.path.exists(fpath):
    raise ValueError(f"Could not load module for ({env_key}={fpath}).\nAre you sure that file exists?")
  return fpath

def load_single_config_given_path(key, fpath):
  """ Load config file from fpath """
  try:
    spec = importlib.util.spec_from_file_location(f"config.{key}", fpath)
    cfg_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_module)
  except Exception as err:
    raise ValueError(f"Could not load module for ({key}={fpath}).\nAre you sure that's a python file?")
  # check that cfg_module is actually a config module
  fold = cfg_module.__dict__.get("FOLD", None)
  cfg_dict = cfg_module.__dict__.get("config", None)
  if cfg_dict is not None and fold is not None and fold in cfg.VALID_FOLDS:
    cfg_dict.FOLD = fold
    if all([k in cfg_dict for k in ['exp_name', 'known_only', "DATA_CSV_PATH"]]):
      return cfg_dict
  raise ValueError(f"Invalid config file ({key}={fpath}).\nAre you sure that's a config file? ")

def validate_configs(closed_config, open_config):
  if open_config.exp_name == closed_config.exp_name:
    raise ValueError("exp_names are the same for open and closed. One may have overwritten the other.")
  if open_config.FOLD != closed_config.FOLD:
    raise ValueError("open and closed are using different folds")
  if open_config.DATA_CSV_PATH != closed_config.DATA_CSV_PATH:
    raise ValueError("open and closed used a different data split")
  if open_config.known_only:
    raise ValueError("Supplied open_config is not really open. Did you check that config.known_only is set to False?")
  if not closed_config.known_only:
    raise ValueError("Supplied closed_config is not really closed. Did you check that config.known_only is set to True?")

def load_configs_given_paths(closed_fpath, open_fpath):
  """load config files (given path) and check that they are valid"""
  closed_config = load_single_config_given_path('CLOSED_CONFIG', closed_fpath)
  open_config = load_single_config_given_path('OPEN_CONFIG', open_fpath)
  validate_configs(closed_config, open_config)
  return closed_config, open_config

def load_configs_from_env():
  """load config files and check that they are valid"""
  closed_fpath = read_config_path_from_env("CLOSED_CONFIG")
  open_fpath = read_config_path_from_env("OPEN_CONFIG")
  return load_configs_given_paths(closed_fpath, open_fpath)

def dict_hash(dictionary: Dict[str, Any]) -> str:
  """MD5 hash of a dictionary."""
  dhash = hashlib.md5()
  # sort arguments so {'a': 1, 'b': 2} is the same as {'b': 2, 'a': 1}
  encoded = json.dumps(dictionary, sort_keys=True).encode()
  dhash.update(encoded)
  return dhash.hexdigest()
