""" Load models, config, and grid_search params """
import json
from sklearn.base import clone as clone_pipeline
from unknown.utils.paths import get_params_path
from unknown.models.t2 import rfor, svm, wnn
from unknown.models.t3 import garb

def get_gs_params(alg):
  assert alg in ['rfor', 'svm', 'wnn', 'garb'], f"Unsupported alg {alg}"
  if alg == 'rfor':
    return rfor.gs_params
  elif alg == 'svm':
    return svm.gs_params
  elif alg == 'wnn':
    return wnn.gs_params
  elif alg == "garb":
    return garb.gs_params

def get_best_params(alg, is_closed=None, params=None):
  """ 
  Load params from file and add defaults as necessary 
      Note: If you get params from a grid search, 
        you should still use this function. Just put
        your grid_searched params into the optional params arg
  """
  assert alg in ['rfor', 'svm', 'wnn', 'garb'], f"Unsupported alg {alg}"
  if is_closed is None:
    assert alg == 'garb', f"Only garb doesn't specify closed vs open set (Found {alg})"
  if params is None:
    fname = get_params_path(alg, is_closed=is_closed)
    with open(fname) as f:
      params = json.load(f)
  if alg == 'wnn':
    best_params = {
      **params,
      **wnn.default_params(is_closed)           
    }
  else:
    best_params = params
  print(f"Best Params: {best_params}")
  return best_params

def get_model(alg, is_closed=None, best_params=None):
  """ Get model set with best_params """
  assert alg in ['rfor', 'svm', 'wnn', 'garb'], f"Unsupported alg {alg}"
  if alg == 'wnn':
    if best_params is None:
      model = wnn.build_wnn_factory(alg, is_closed)
    else:
      model = wnn.build_wnn(**best_params)
  else:
    if alg == 'rfor':
      pipeline = rfor.pipeline  
    elif alg == 'svm':
      pipeline = svm.pipeline
    elif alg == 'garb':
      pipeline = garb.pipeline
    model = clone_pipeline(pipeline)
    if best_params is not None:
      model.set_params(**best_params)
  return model