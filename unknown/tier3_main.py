''' Train/Test/Validate SVM '''
import os
import sys
sys.path.append(
  os.path.dirname(
  os.path.dirname(
  os.path.abspath(__file__))))
import time
from shutil import copyfile
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report
from unknown.loaders.test_split import SpeciesBalancedTestTrainSplitter
from unknown.loaders.cache_loader import T2Iterator
from unknown.loaders.cross_val import SpeciesStratifiedShuffleSplit
from unknown.models.zoo import get_model, get_best_params
from unknown.config.trials import trials
from unknown.search.gs import t3_grid_search
from unknown.utils.paths import DATA_DIR, get_t3_save_location, get_t3_save_folder, get_t3_config_folder, get_t3_config_path
from unknown.utils.helpers import load_configs_given_paths, read_config_path_from_env
from unknown.config.config import UnknownConfigLoader
from unknown.utils.logger import Logger
cfg = UnknownConfigLoader()
np.random.seed(cfg.SEED) # for reproducibility
import tensorflow as tf
tf.random.set_seed(cfg.SEED) # for reproducibility
from tensorflow import keras

def main(
  alg, run, trial, t1_closed_config,
  t1_open_config, gs=False, num_seeds=5,
  should_test=False, closed_config_fpath=None,
  open_config_fpath=None):
  ''' Grid Search or Cross Validate individual tier3 methods '''
  try:
    logger = Logger(
      os.path.join(
        os.path.dirname(
          get_t3_save_folder(run)),
        f"{alg}_fold{t1_closed_config.FOLD}.txt"
    ))
    logger.log(f"----------------- STARTING {time.time()} -----------------")
    splitter = SpeciesBalancedTestTrainSplitter(t1_closed_config, t1_open_config, random_seed=cfg.SEED)
    if gs:
      t2_iter = T2Iterator(trial, splitter, fold=t1_closed_config.FOLD, species_seed=0, use_val=True)
      # closed
      logger.log(f"Running grid search on closed set data ({alg})")
      x_train, y_train, id_train = t2_iter.load('train')
      x_valid, y_valid, id_valid = t2_iter.load('valid') # because we used use_val, we have a valid (see cache_loader for details)
      params = t3_grid_search(
        x_train, y_train, id_train,
        x_valid, y_valid, id_valid)
      logger.log("---------------------------------------------")
      logger.log(f"Best Params for ({alg}): {params}")
      with open(os.path.join(DATA_DIR, "tier3", "params", alg, "params.json"), 'w') as f:
        json.dump(params, f)
    
    # validate
    for seed in range(num_seeds):
      logger.log(f'--------------- SEED {seed} ---------------') 
      t2_iter = T2Iterator(trial, splitter, fold=t1_closed_config.FOLD, species_seed=seed)
      x_train, y_train, id_train = t2_iter.load('train')

      # fit model
      best_params = get_best_params(alg)
      clf = get_model(alg, best_params=best_params)
      logger.log('Params:', clf.get_params(deep=True))   
      logger.log('--------------- Cross Validation ---------------')
      cv = SpeciesStratifiedShuffleSplit(t1_closed_config.DATA_CSV_PATH, n_splits=3, random_state=cfg.SEED, shuffle=True)
      scores = cross_validate(clf, x_train, y_train, groups=id_train, scoring=['precision', 'recall', 'f1'], cv=cv)
      for k,v in scores.items():
        logger.log(k, ': ', v)
      # Test
      if should_test:
        dfs = []
        logger.log('\n\n--------------- Test ---------------\n\n')
        clf = get_model(alg, best_params=best_params)
        clf.fit(x_train, y_train)
        # clf.save("garb.joblib")
        for split, x_test, y_test, test_names in t2_iter.iter():
          if 'T' in split:
            continue
          preds = clf.predict(x_test)
          logger.log(f'--------------- Set {split} ---------------')
          logger.log(classification_report(y_test, preds, target_names=['known', 'unknown'], labels=[0,1]))
          df_curr = pd.DataFrame(test_names, columns=['Id'])
          df_curr['y'] = y_test
          df_curr['preds'] = preds
          df_curr['split'] = [split] * len(df_curr)
          dfs.append(df_curr)
        df = pd.concat(dfs, axis=0)
        fpath = get_t3_save_location(run, fold=t1_closed_config.FOLD, species_seed=seed)
        df.to_csv(fpath, index=None)
  except Exception as err:
    logger.log(f"Error: {err}")
  else:
    if should_test:
      if num_seeds == 5:
        # copy configs into the config dir
        closed_cfg_dir = get_t3_config_folder(run, True, t1_closed_config.FOLD)
        if not os.path.exists(closed_cfg_dir):
          os.makedirs(closed_cfg_dir)
        open_cfg_dir = get_t3_config_folder(run, False, t1_open_config.FOLD)
        if not os.path.exists(open_cfg_dir):
          os.makedirs(open_cfg_dir)
        copyfile(closed_config_fpath, 
          get_t3_config_path(run, True, t1_closed_config.FOLD))
        copyfile(open_config_fpath, 
          get_t3_config_path(run, False, t1_open_config.FOLD))
      # print results
      logger.log(f'--------- Totals for Fold {t1_closed_config.FOLD} ----------')
      dfs = [pd.read_csv(get_t3_save_location(run, fold=t1_closed_config.FOLD, species_seed=seed)) for seed in range(num_seeds)]
      df = pd.concat(dfs, axis=0, ignore_index=True)
      
      logger.log(f'--------------- Set T ---------------')
      logger.log(classification_report(df.y, df.preds, target_names=['known', 'unknown'], labels=[0,1]))

      logger.log(f'--------------- Set T1 ---------------')
      df_T1 = df[(df.split == 'A') | (df.split == 'B')]
      logger.log(classification_report(df_T1.y, df_T1.preds, target_names=['known', 'unknown'], labels=[0,1]))

      logger.log(f'--------------- Set T2 ---------------')
      df_T2 = df[(df.split == 'C') | (df.split == 'D')]
      logger.log(classification_report(df_T2.y, df_T2.preds, target_names=['known', 'unknown'], labels=[0,1]))

      logger.log(f'--------------- Set A ---------------')
      df_A = df[(df.split == 'A')]
      logger.log(classification_report(df_A.y, df_A.preds, target_names=['known', 'unknown'], labels=[0,1]))

      logger.log(f'--------------- Set B ---------------')
      df_B = df[(df.split == 'B')]
      logger.log(classification_report(df_B.y, df_B.preds, target_names=['known', 'unknown'], labels=[0,1]))

      logger.log(f'--------------- Set C ---------------')
      df_C = df[(df.split == 'C')]
      logger.log(classification_report(df_C.y, df_C.preds, target_names=['known', 'unknown'], labels=[0,1]))

      logger.log(f'--------------- Set D ---------------')
      df_D = df[(df.split == 'D')]
      logger.log(classification_report(df_D.y, df_D.preds, target_names=['known', 'unknown'], labels=[0,1]))

    
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Optimize/Validate/Test Tier3 Methods')
  parser.add_argument('run', type=str, help='Which run to')
  parser.add_argument('--grid-search', '-gs', action='store_true', help='Run a grid search first')
  parser.add_argument('--num-seeds', '-ns', type=int, default=5, help='Number of seeds to use')
  parser.add_argument('--test', action='store_true', help='Run on test data')
  parser.add_argument('--trial', '-t', type=str, default="T1", help='Which data to use (Default: T1 as used in paper)')

  args = parser.parse_args()
  
  # validate
  try:
    # load config files from the env variables CLOSED_CONFIG and OPEN_CONFIG    
    closed_config_fpath =  read_config_path_from_env("CLOSED_CONFIG")
    open_config_fpath =  read_config_path_from_env("OPEN_CONFIG")
    closed_config, open_config = load_configs_given_paths(closed_config_fpath, open_config_fpath)
    # Trial
    trial = trials.get(args.trial, None)
    if trial is None:
      raise ValueError(f"Invalid trial name {args.trial}")
    if args.num_seeds > 5 or args.num_seeds < 0:
      raise ValueError(f"{args.num_seeds} is an invalid number of seeds  (0 <= n < 5)")
    save_dir = get_t3_save_folder(args.run, job_type='full')
    if os.path.exists(save_dir):
      if any([os.path.exists(get_t3_save_location(args.run, fold=closed_config.FOLD, species_seed = seed)) for seed in range(5)]):
        raise ValueError(f"Found previous results for fold {closed_config.FOLD}.\n    \
                              Check {save_dir} for more info")
    else:
      os.makedirs(save_dir)
  except ValueError as err:
    print(err)
  else:     
    main(
      "garb", args.run, trial,
      closed_config, open_config,
      gs=args.grid_search, num_seeds=args.num_seeds,
      should_test=args.test, closed_config_fpath=closed_config_fpath,
      open_config_fpath=open_config_fpath)
