import os
import numpy as np
import pandas as pd
from unknown.config.config import UnknownConfigLoader
cfg = UnknownConfigLoader()
np.random.seed(cfg.SEED) # for reproducibility
import tensorflow as tf
tf.random.set_seed(cfg.SEED) # for reproducibility
from tensorflow import keras
from unknown.utils.helpers import dict_hash
from unknown.loaders.test_split import T1Iterator
from unknown.loaders.cross_val import speciesBalancedShuffleSplit
from unknown.models.zoo import get_model, get_best_params
from unknown.utils.paths import CACHE_DIR

def cache_t2_data(trial, splitter, fold=1, species_seed=1, use_val=False): 
  """ 
  Iterate through all t2 methods required for this trial and cache the necessary data
    * use_val: splits the usual train set into a train and validation set, then puts the validation set into "T".
        This is used for feature optimization in tier3 methods
  """
  info = trial['t1']
  t1_iter = T1Iterator(trial, splitter, species_seed=species_seed)
  
  if t1_iter.combined:
    combos = ['combined']
  else:
    combos = []
    for num_class in info['num_classes']:
      for layer_size in info['layer_size']:
        for t1_alg in info['algs']:
          combos.append(f'{t1_alg}_{num_class}_{layer_size}')
  all_data = {}
  for combo in combos:
    all_data[combo] = {}
    for alg in t1_iter.trial['t2']: 
      # load training data
      if use_val:
        x_train_all, y_train_all, id_train_all = t1_iter.load_given_combo('train', combo, wnn=(alg == 'wnn'))
        train_index, valid_index = speciesBalancedShuffleSplit(id_train_all, fold=fold, species_seed=species_seed, goal_bounds = (-5,2), random_seed=cfg.SEED)
        if alg == 'wnn':
          x_train = (x_train_all[0][train_index], x_train_all[1][train_index])
          x_valid = (x_train_all[0][valid_index], x_train_all[1][valid_index])
        else:
          x_train = x_train_all[train_index]
          x_valid = x_train_all[valid_index]
        y_train = y_train_all[train_index]
        y_valid = y_train_all[valid_index]
        train_names = id_train_all[train_index]                
        valid_names = id_train_all[valid_index]
      else:
        x_train, y_train, train_names = t1_iter.load_given_combo('train', combo, wnn=(alg == 'wnn'))
      is_closed = ('16' in combo)
      best_params = get_best_params(alg, is_closed=is_closed)
      model = get_model(alg, is_closed=is_closed, best_params=best_params)
      # train
      print("Training model...")
      if alg == "wnn":
        model.fit(x_train, y_train, epochs=20, workers=-1, batch_size=cfg.KERAS_BATCH_SIZE)
      else:
        model.fit(x_train, y_train)      
      # test
      print("Gathering test data...")  
      for split, x_test, y_test, test_names in t1_iter.iter_given_combo(combo, wnn=(alg == 'wnn'), skip_train=False):
        if use_val: 
          if split == 'T':
            x_test = x_valid
            y_test = y_valid
            test_names = valid_names
        if alg == 'wnn':
          preds = np.squeeze(model.predict(x_test))
        else:
          preds = model.predict_proba(x_test)[:,1]
        if split not in all_data[combo]:
          all_data[combo][split] = {
            'Id': test_names,
            'y': y_test,
            alg: preds,
          }
        else:
          all_data[combo][split][alg] = preds
  return all_data
  
def load_t2_data(trial, splitter, fold=1, species_seed=1, use_val=False):
  """ Try to load t2 data from cache """
  info = trial['t1']
  if use_val:
    trial_hash = dict_hash({'fold': fold, 'species_seed': species_seed, 't2': trial['t2'], 'use_val': True, **info})
  else:
    trial_hash = dict_hash({'fold': fold, 'species_seed': species_seed, 't2': trial['t2'], **info})
  cache_path = os.path.join(CACHE_DIR, 't2', f'data_{trial_hash}.npz')
  if not os.path.exists(cache_path):
    print("T2 cache miss")
    all_data = cache_t2_data(trial, splitter, fold=fold, species_seed=species_seed, use_val=use_val)    
    print(f"Saving to cache {cache_path}...")            
    np.savez(cache_path, **all_data)
  # else:
  #   print("T2 cache hit")    
  # print(f"Loading from cache {cache_path}...")      
  return np.load(cache_path, allow_pickle=True)

class T2Iterator():
  """
   Iterate over tier2 results (aka tier3 input)
    Note: See cache_t2_data() for more info on the use_val argument  
  """
  def __init__(self, trial, splitter, fold=1, species_seed=1, use_val=False):
    self.trial = trial
    self.fold = fold
    self.species_seed = species_seed
    self.use_val = use_val
    self.splitter = splitter
    self.splitter.reshuffle_data(species_seed)
    self.data = load_t2_data(trial, splitter, fold=fold, species_seed=species_seed, use_val=use_val)
    
  def load(self, split):
    """ Load a specific set from the T2 data """
    # Make sure split is correct given the status of use_val
    if split == 'valid':
      if self.use_val:
        split = 'T'
      else:
        raise ValueError("There is no valid in the set, since you did not use the 'use_val' flag")
    elif split == 'T':
      if self.use_val:
        raise ValueError("There is no T in the set, since you used the 'use_val' flag")
    x_all = []
    y_all = []
    id_all = []
    for combo in self.data.keys():
      df = pd.DataFrame(self.data[combo].item()[split])
      df.sort_values(by='Id', inplace=True)
      id_all.append(df.pop('Id').to_numpy())
      y_all.append(df.pop('y').to_numpy())
      x_all.append(df.to_numpy())
    return np.concatenate(x_all, axis=1), y_all[0], id_all[0]
  
  def iter(self, skip_train=True):
    """ Iterate over all splits of t2 data """
    splits = cfg.T2_TEST_SPLITS if skip_train else ['train', *cfg.T2_TEST_SPLITS]    
    for split in splits:
      if split == 'T' and self.use_val:
        split = 'valid'
      yield split, *self.load(split)
