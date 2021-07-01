import os.path
import json
from unknown.utils.paths import CONFIG_DIR
from unknown.config.config import UnknownConfigLoader
config = UnknownConfigLoader()

_config_json_path = os.path.join(CONFIG_DIR, 'state.json')

def _get_config_json():
  with open(_config_json_path, 'r') as f:
    config_d = json.load(f)
  return config_d

def _write_config_json(d):
  with open(_config_json_path, 'w') as f:
    f.write(json.dumps(d))
  return d

def get_state(name):
  ''' Get state variable by name'''
  return _get_config_json()[name]

def update_state(name, value=None):
  ''' Update state variable '''
  d = _get_config_json()
  if value is None:
    value = d[name] + 1
  d[name] = value
  return _write_config_json(d)

def get_run(alg):
  ''' Get current run num '''
  return get_state(f'run-{alg}')

def update_run(alg):
  ''' Increment run num '''
  return update_state(f'run-{alg}')