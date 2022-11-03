import os
from unknown.loaders.cache_loader import T2Iterator
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report
from unknown.loaders.test_split import SpeciesBalancedTestTrainSplitter
from unknown.models.other.extras import SoftmaxClassifier
from unknown.utils.paths import get_t3_extra_logfile_location, get_t3_save_location, get_t3_save_folder
from unknown.utils.logger import Logger
from unknown.config.config import UnknownConfigLoader
from unknown.utils.helpers import load_configs_given_paths
from unknown.utils.paths import get_t3_config_path

cfg = UnknownConfigLoader()

def try_make_job_dir(run_num, job):
  job_dir = get_t3_save_folder(run_num, job_type=job)
  if os.path.exists(job_dir):
    raise FileExistsError(f"{job.capitalize()} already exists. Please delete/rename previous {job} folder and rerun")
  else:
    os.makedirs(job_dir)
  return job_dir                       

def do_closedsoftmax_extras(run_num, trial=None, folds = range(1, 5+1), seeds = range(5)):
  '''
   Get unknown from t1 closed set results directly using a naive softmax thresholding approach
    trial is included for compatability (not used)
  '''
  job = 'extras/closedsoftmax'
  try_make_job_dir(run_num, job)
  job_file = get_t3_extra_logfile_location(run_num, job_type=job)
  logger = Logger(job_file)
  for fold in tqdm(folds, desc=f'{job} Fold'):
    logger.log(f"---------------- Fold {fold} ----------------\n")
    closed_cfg_fpath = get_t3_config_path(run_num, True, fold)
    open_cfg_fpath = get_t3_config_path(run_num, False, fold)
    closed_config, open_config = load_configs_given_paths(closed_cfg_fpath, open_cfg_fpath)
    splitter = SpeciesBalancedTestTrainSplitter(closed_config, open_config)
    for species_seed in tqdm(seeds, desc=f'{job} Seeds ({fold})'):
      logger.log(f"---------------- Seed {species_seed} ----------------\n")
      data = splitter.get_t1_data('xception', 16, 'probs', species_seed=species_seed)
      x_train, y_train, id_train = data.load('train')
      clf = SoftmaxClassifier(n_trials=500)
      clf.fit(x_train, y_train)
      dfs = []
      for split, x_test, y_test, id_test in data.iter():
        pred = clf.predict(x_test)
        if split == 'T':
          logger.log(f"---------------- Split {split} ----------------\n")
          logger.log(classification_report(y_test, pred, target_names=['known', 'unknown'], labels=[0,1]))
        if split in ['A', 'B', 'C', 'D']:
          dfs.append(
            pd.DataFrame({
              'Id': id_test,
              'y': y_test,
              'pred': pred,
              'split': [split] * len(y_test)
            })
          )
      df = pd.concat(dfs, axis=0)
      df.to_csv(get_t3_save_location(run_num, job_type=job, fold=fold, species_seed=species_seed), index=None)

def do_openset_extras(run_num, trial=None, folds = range(1, 5+1), seeds = range(5)):
  '''
   Get unknown from t1 open set results directly using a naive grouping approach
    trial is included for compatability (not used)
  '''
  job = 'extras/openset'
  try_make_job_dir(run_num, job)
  job_file = get_t3_extra_logfile_location(run_num, job_type=job)
  logger = Logger(job_file)
  for fold in tqdm(folds, desc=f'{job} Fold'):
    logger.log(f"---------------- Fold {fold} ----------------\n")
    closed_cfg_fpath = get_t3_config_path(run_num, True, fold)
    open_cfg_fpath = get_t3_config_path(run_num, False, fold)
    closed_config, open_config = load_configs_given_paths(closed_cfg_fpath, open_cfg_fpath)
    splitter = SpeciesBalancedTestTrainSplitter(closed_config, open_config)
    for species_seed in tqdm(seeds, desc=f'{job} Seeds ({fold})'):
      logger.log(f"---------------- Seed {species_seed} ----------------\n")
      data = splitter.get_t1_data('xception', 21, 'probs', species_seed=species_seed)
      dfs = []
      for split, x_test, y_test, id_test in data.iter():
        pred = 1*(np.apply_along_axis(lambda row: np.argmax(row), axis=1, arr=x_test) >= 16)
        if split == 'T':
          logger.log(f"---------------- Split {split} ----------------\n")
          logger.log(classification_report(y_test, pred, target_names=['known', 'unknown'], labels=[0,1]))
        if split in ['A', 'B', 'C', 'D']:
          dfs.append(
            pd.DataFrame({
              'Id': id_test,
              'y': y_test,
              'pred': pred,
              'split': [split] * len(y_test)
            })
          )
      df = pd.concat(dfs, axis=0)
      fpath = get_t3_save_location(run_num, job_type=job, fold=fold, species_seed=species_seed)
      df.to_csv(fpath, index=None)

def do_t2_16xcept_extras(run_num, trial, folds = range(1, 5+1), seeds = range(5)):
  '''
   Get t2 results from the cache into a human readable format
  '''
  job = 't2_16xcept'
  try_make_job_dir(run_num, job)
  job_file = get_t3_extra_logfile_location(run_num, job_type=job)
  logger = Logger(job_file)
  for fold in tqdm(folds, desc='Fold'):
    logger.log(f'---------------- FOLD {fold} ----------------\n')
    closed_cfg_fpath = get_t3_config_path(run_num, True, fold)
    open_cfg_fpath = get_t3_config_path(run_num, False, fold)
    closed_config, open_config = load_configs_given_paths(closed_cfg_fpath, open_cfg_fpath)
    splitter = SpeciesBalancedTestTrainSplitter(closed_config, open_config)
    for species_seed in tqdm(seeds, desc=f'Fold {fold} Seeds'):
      logger.log(f'---------------- SEED {species_seed} ----------------\n')
      t2_iter = T2Iterator(trial, splitter, fold=fold, species_seed=species_seed)
      combo = 'xception_16_feats'
      logger.log(f'---------------- {" ".join(map(lambda x: x.capitalize(), combo.split("_")))} ----------------\n')
      dfs = []
      for split in ['T', 'A', 'B', 'C', 'D']:
        df = pd.DataFrame(t2_iter.data[combo].item()[split])
        for alg in ['rfor', 'svm', 'wnn']:
          df[f'{alg} probas'] = df[alg]
          df[f'{alg} preds'] = df[alg].apply(lambda x: int(x > 0.5))
          if split == 'T':
            logger.log(f'---------------- Split T ----------------\n')
            logger.log(f'---------------- {alg} ----------------\n')
            logger.log(classification_report(df.y, df[f'{alg} preds'], target_names=['known', 'unknown'], labels=[0,1]))
            logger.log('\n\n')
        df['split'] = [split] * len(df)
        if split != 'T':
          alg_cols = [[f'{alg} preds', f'{alg} probas'] for alg in ['rfor', 'svm', 'wnn']]
          dfs.append(df[['Id', 'y', *np.array(alg_cols).flatten(), 'split']])
      pd.concat(dfs, axis=0).to_csv(
        get_t3_save_location(run_num, job_type=job, fold=fold, species_seed=species_seed), index=None)

def do_svote_extras(run_num, trial, folds = range(1, 5+1), seeds = range(5)):
  '''
   Get unknown results using a raw soft voting classifer rather than the Arbitration CLS
  '''
  job = 'svote'
  try_make_job_dir(run_num, job)
  job_file = get_t3_extra_logfile_location(run_num, job_type=job)
  logger = Logger(job_file)
  for fold in tqdm(folds, desc='Fold'):
    logger.log(f'---------------- FOLD {fold} ----------------\n')
    closed_cfg_fpath = get_t3_config_path(run_num, True, fold)
    open_cfg_fpath = get_t3_config_path(run_num, False, fold)
    closed_config, open_config = load_configs_given_paths(closed_cfg_fpath, open_cfg_fpath)
    splitter = SpeciesBalancedTestTrainSplitter(closed_config, open_config)
    for species_seed in tqdm(seeds, desc=f'Fold {fold} Seeds'):
      logger.log(f'---------------- SEED {species_seed} ----------------\n')
      t2_iter = T2Iterator(trial, splitter, fold=fold, species_seed=species_seed)
      dfs = []
      for split, x_test, y_test, test_names in t2_iter.iter():
        if split in ['T', 'A', 'B', 'C', 'D']:
          probas = np.apply_along_axis(lambda row: (sum(row)/len(row)), axis=1, arr=x_test)
          preds = np.apply_along_axis(lambda row: int((sum(row)/len(row)) > 0.5), axis=1, arr=x_test)
          if split == 'T':
            logger.log(f'---------------- Split T ----------------\n')
            logger.log(classification_report(y_test, preds, target_names=['known', 'unknown'], labels=[0,1]))
            logger.log('\n\n')
          else:
            dfs.append(pd.DataFrame({
              'Id': test_names,
              'y': y_test,
              'probas': probas,
              'preds': preds,
              'split': [split] * len(test_names)
            }))
      pd.concat(dfs, axis=0).to_csv(
        get_t3_save_location(run_num, job_type=job, fold=fold, species_seed=species_seed), index=None)