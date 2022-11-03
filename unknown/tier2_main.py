''' Train/Test/Validate SVM '''
import os
import sys
sys.path.append(
  os.path.dirname(
  os.path.dirname(
  os.path.abspath(__file__))))
import json
import joblib
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report
from unknown.loaders.test_split import T1Iterator, SpeciesBalancedTestTrainSplitter
from unknown.loaders.cross_val import SpeciesStratifiedShuffleSplit, speciesBalancedShuffleSplit
from unknown.config.state import update_run, get_run
from unknown.models.zoo import get_model, get_best_params
from unknown.config.trials import trials
from unknown.search.gs import t2_grid_search
from unknown.utils.paths import DATA_DIR
from unknown.utils.helpers import load_configs_from_env
from unknown.config.config import UnknownConfigLoader
cfg = UnknownConfigLoader()
np.random.seed(cfg.SEED) # for reproducibility
import tensorflow as tf
tf.random.set_seed(cfg.SEED) # for reproducibility
from tensorflow import keras

def main(alg, trial, t1_closed_config, t1_open_config, gs=False, num_seeds=5, should_test=False):
  ''' Grid Search or Cross Validate individual tier2 methods '''
  assert alg in cfg.SUPPORTED_TIER2_ALGS, f"Unsupported Tier2 Algorithm {alg}"
  run_num = get_run(alg)
  update_run(alg)
  try:
    splitter = SpeciesBalancedTestTrainSplitter(t1_closed_config, t1_open_config, random_seed=cfg.SEED)
    if gs:
      t1_iter = T1Iterator(trial, splitter, species_seed=0)
      # closed
      print(f"Running grid search on closed set data ({alg})")
      x_train, y_train, id_train = t1_iter.load('train', num_classes=16, wnn=(alg=='wnn'))
      closed_params = t2_grid_search(
        alg, run_num, t1_closed_config, True, 
        x_train, y_train, id_train,
        random_state=cfg.SEED)
      print("---------------------------------------------")
      print(f"Best Params for Closed Set ({alg}): {closed_params}")
      with open(os.path.join(DATA_DIR, "tier2", "params", alg, "params_16.json"), 'w') as f:
        json.dump(closed_params, f)

      # open
      print(f"Running grid search on open set data ({alg})")
      x_train, y_train, id_train = t1_iter.load('train', num_classes=21, wnn=(alg=='wnn'))
      open_params = t2_grid_search(
        alg, run_num, t1_open_config, False, 
        x_train, y_train, id_train,
        random_state=cfg.SEED)
      print("---------------------------------------------")
      print(f"Best Params for Open Set ({alg}): {open_params}")
      with open(os.path.join(DATA_DIR, "tier2", "params", alg, "params_21.json"), 'w') as f:
        json.dump(open_params, f)
    
    # validate
    for is_closed in [True, False]:    
      print(f'--------------- {"Closed" if is_closed else "Open"} Set ---------------')
      t1_config = t1_closed_config if is_closed else t1_open_config
      for seed in range(num_seeds):
        print(f'--------------- SEED {seed} ---------------') 
        t1_iter = T1Iterator(trial, splitter, species_seed=seed)
        x_train, y_train, id_train = t1_iter.load('train', num_classes=(16 if is_closed else 21), wnn=(alg=='wnn'))

        # fit model
        best_params = get_best_params(alg, is_closed=is_closed)
        clf = get_model(alg, is_closed=is_closed, best_params=best_params)
        if alg != "wnn":
          print('Params:', clf.get_params(deep=True))   
        else:
          print('Params: ', best_params)   
        print('--------------- Cross Validation ---------------')
        cv = SpeciesStratifiedShuffleSplit(t1_config.DATA_CSV_PATH, n_splits=3, random_state=cfg.SEED, shuffle=True)
        if alg in ['rfor', "svm"]:
          clf.fit(x_train, y_train)
          # joblib.dump(f'{alg}.joblib')
          scores = cross_validate(clf, x_train, y_train, groups=id_train, scoring=['precision', 'recall', 'f1'], cv=cv)
          for k,v in scores.items():
            print(k, ': ', v)
        else: # alg == 'wnn'
          if should_test:
            train_index, valid_index = list(cv.split(x_train[0], y_train, id_train))[0]
            x_vtrain = (x_train[0][train_index], x_train[1][train_index])
            y_vtrain = y_train[train_index]
            id_vtrain = id_train[train_index]
            x_valid = (x_train[0][valid_index], x_train[1][valid_index])
            y_valid = y_train[valid_index]
            id_valid = id_train[valid_index]
            clf.fit(x_vtrain, y_vtrain, epochs=cfg.KERAS_EPOCHS, workers=-1,
                        validation_data=(x_valid, y_valid), batch_size=cfg.KERAS_BATCH_SIZE,
                        callbacks=[keras.callbacks.EarlyStopping(patience=10)])
          else:
            clf.fit(x_train, y_train, epochs=20, workers=-1, batch_size=cfg.KERAS_BATCH_SIZE) 
          # clf.save(alg)
        # Test
        if should_test:
          print('\n\n--------------- Test ---------------\n\n')
          test_iter = t1_iter.iter(num_classes=16 if is_closed else 21, wnn=(alg=='wnn'))
          for split, x_test, y_test, test_names in test_iter:
            preds = clf.predict(x_test)
            if alg == 'wnn':
              preds = np.where(preds > 0.5, 1, 0)
            print(f'--------------- Set {split} ---------------')
            print(classification_report(y_test, preds, target_names=['known', 'unknown'], labels=[0,1]))

  except Exception as err:
    print(f"Error: {err}")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Optimize/Validate/Test Tier2 Methods')
  parser.add_argument('alg', type=str, help='Which tier2 alg (rfor, svm, wnn)')
  parser.add_argument('--grid-search', '-gs', action='store_true', help='Run a grid search first')
  parser.add_argument('--num-seeds', '-ns', type=int, default=5, help='Number of seeds to use')
  parser.add_argument('--test', action='store_true', help='Run on test data')
  parser.add_argument('--trial', '-t', type=str, default="T1", help='Which data to use (Default: T1 as used in paper)')

  args = parser.parse_args()
  
  # validate
  try:
    # load config files from the env variables CLOSED_CONFIG and OPEN_CONFIG
    closed_config, open_config = load_configs_from_env()
    # Trial
    trial = trials.get(args.trial, None)
    if trial is None:
      raise ValueError(f"Invalid trial name {args.trial}")
    if args.num_seeds > 5 or args.num_seeds < 0:
      raise ValueError(f"{args.num_seeds} is an invalid number of seeds  (0 <= n < 5)")
  except ValueError as err:
    print(err)
  else:     
    main(args.alg, trial, closed_config, open_config, gs=args.grid_search, num_seeds=args.num_seeds, should_test=args.test)
