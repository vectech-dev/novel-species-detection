'''Validates and retrieves path information'''
import os.path
from unknown.config.config import UnknownConfigLoader
cfg = UnknownConfigLoader()

SRC_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
ROOT_DIR = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(ROOT_DIR, 'tier3_output')
CONFIG_DIR = os.path.join(SRC_DIR, 'config')
DATA_DIR = os.path.join(SRC_DIR, "data")
CACHE_DIR = os.path.join(DATA_DIR, 'cache')


def get_t3_save_folder(run_num, job_type='full'):
  assert job_type in ["full", "t2_16xcept", "svote", "extras/openset", "extras/closedsoftmax"]
  return os.path.join(
    ROOT_DIR, "tierIII_output",
    run_num, job_type)

def get_t3_config_folder(run_num, is_closed, fold):
  return os.path.join(
    os.path.dirname(
      get_t3_save_folder(run_num)), 
    'config', 'closed' if is_closed else 'open', 
    f"fold{fold}")

def get_t3_config_path(run_num, is_closed, fold):
  return os.path.join(
    get_t3_config_folder(run_num, is_closed, fold), 
    'config.py')

def get_t3_save_location(run_num, job_type='full', fold=1, species_seed=0):
  fdir = get_t3_save_folder(run_num, job_type=job_type)
  job_type_str = job_type.replace('/', '_')
  return os.path.join(fdir, 
    f"{job_type_str}_fold{fold}_seed{species_seed}.csv")

def get_t3_extra_logfile_location(run_num, job_type='full'):
  return os.path.join(
    ROOT_DIR, "tierIII_output",
    run_num, f"{job_type}.txt")

def _get_garb_params_path():
  """ Same as get_params_path but garb doesn't have concept of close/openset"""
  alg = 'garb'
  params_dir = os.path.join(DATA_DIR, "tier3", "params", alg)
  param_files = [param_file for param_file in os.listdir(params_dir) if param_file.endswith(".json")]
  if len(param_files) > 0:
    if len(param_files) == 1:
      return os.path.join(params_dir,  param_files[0])
    else:
      raise ValueError(f"Found too many valid param files for {alg}")
  return os.path.join(params_dir, "finale", f"fold1_seed0__params.json")

def get_params_path(alg, is_closed=None):
  """ Get path to hyperparameters for this alg """
  assert alg in ['rfor', 'svm', 'wnn', 'garb'], f"Unsupported alg {alg}"
  num_classes = 16 if is_closed else 21
  if alg == "garb":
    return _get_garb_params_path()
  params_dir = os.path.join(DATA_DIR, "tier2", "params", alg)
  param_files = [param_file for param_file in os.listdir(params_dir) if param_file.endswith(".json") and str(num_classes) in param_file]
  if len(param_files) > 0:
    if len(param_files) == 1:
      return os.path.join(params_dir,  param_files[0])
    else:
      raise ValueError(f"Found too many valid param files for {alg}")
  return os.path.join(params_dir, "finale", f"fold1_seed0__xception_{num_classes}_feats.json")

def get_split_fold(data_csv_path: str):
  ''' Get location of split_fold csv (given data_csv_path from config)'''
  fpath = data_csv_path
  if os.path.isabs(fpath):
    return fpath
  return os.path.join(ROOT_DIR, fpath)

def get_unknown_datasplits(fold):
  ''' Get location of unknown_datasplits xlsx '''
  return os.path.join(DATA_DIR, 'data_splits', f"T2T3datasplits_fold{fold}.xlsx")

def get_tier1_csv_path(exp_name:str, layer_size='feats'):
  ''' returns path to the relevant tier 1 features from the data dir (given exp_name from config) '''
  return os.path.join (ROOT_DIR,
    "tierI_output",
    exp_name,
    "features.csv" if layer_size == "feats" else "probabilities.csv")