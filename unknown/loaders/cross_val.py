import pandas as pd
import numpy as np
from itertools import chain
import re
import random

from unknown.utils.paths import get_split_fold, get_unknown_datasplits
from unknown.loaders.test_split import SpeciesBalancedTestTrainSplitter
from unknown.config.config import UnknownConfigLoader
cfg = UnknownConfigLoader()

genus_pattern = r'\/opt\/ImageBase\/AWS_sync\/cropped([a-zA-Z\/]*\/[a-zA-Z_\-\d]+)\/[a-zA-Z\/_\-\d]*JHU-\d+_\d+m\.jpg'
species_pattern = r'\/opt\/ImageBase\/AWS_sync\/cropped[a-zA-Z\/]*\/([a-zA-Z_\-\d]+)\/[a-zA-Z\/_\-\d]*JHU-\d+_\d+m\.jpg'
specimen_pattern = r'\/opt\/ImageBase\/AWS_sync\/cropped[a-zA-Z\/]*\/[a-zA-Z_\-\d]+\/[a-zA-Z\/_\-\d]*(JHU-\d+)_\d+m\.jpg'

class SpeciesStratifiedShuffleSplit():
  """
    cross-validation generator which shuffles so that...
      1. items with the same specimen are in the same cv fold
      2. Known species should be evenly spread amongst all splits
      2. each cv fold has some species that are ExtendedTest and do not appear in any other cv fold
  """
  # class_legend = {'aegypti': 0, 'albopictus': 1, 'dorsalis': 2, 'japonicus': 3, 'sollicitans': 4, 'vexans': 5, 'coustani': 6, 'crucians_sl': 7, 'freeborni': 8, 'funestus_sl': 9, 'gambiae_sl': 10, 'erraticus': 11, 'pipiens_sl': 12, 'salinarius': 13, 'columbiae': 14, 'ferox': 15}
  def __init__(self, data_csv_path, n_splits=3, random_state=cfg.SEED, shuffle=True):
    self.n_splits = n_splits
    self.random_state = random_state
    self.shuffle = shuffle
    df_split = pd.read_csv(get_split_fold(data_csv_path))
    df_split['species_name'] = df_split["Id"].str.extract(species_pattern, expand=True)
    self.class_legend = df_split[['species_name','Species']].drop_duplicates(subset='species_name').set_index('species_name').to_dict()['Species']

  def get_n_splits(self, x, y, groups=None):
    ''' Get number of splits '''
    return self.n_splits

  def split(self, x, y, groups):
    ''' Split dataset for cross-validation (x is x, y is y, groups is names) '''
    # create df
    df = pd.DataFrame(x)
    df['Id'] = groups
    df['y'] = y
    df['species_name'] = df["Id"].str.extract(species_pattern, expand=True)
    df['specimen'] = df["Id"].str.extract(specimen_pattern, expand=True)
    df['species'] = df.species_name.apply(lambda s: self.class_legend.get(s, 17))
    
    # init groups
    groups = []
    for n_split in range(self.n_splits):
      groups.append([])
    i = 0

    # add known species evenly
    counts = dict(df[df.species < 16][['Id', 'species']].groupby('species').count()['Id'])
    split_perc = 1/len(groups)
    goals = [round(split_perc * count) for spec, count in counts.items()]
    curr_all = [[0] * len(goals) for n_split in range(self.n_splits)]

    NUMERIC_THRESHOLD = 2
    PERC_THRESHOLD = 5
    for n_split in range(self.n_splits):
      curr = curr_all[n_split]
      is_done = lambda idx: (goals[idx] - curr[idx] <= NUMERIC_THRESHOLD) or (((goals[idx] - curr[idx])/goals[idx]) * 100 < PERC_THRESHOLD)
      while(True):
        i+=1
        diff = [(goals[i] - curr[i]) if curr[i] < goals[i] else 0 for i in range(len(goals))]
        max_spec = np.argmax(diff) # find item furthest from goal
        try:
          sample = df[df['species'] == max_spec].sample(random_state=self.random_state + i)
        except ValueError:
          break
        indices = df[df.specimen == sample.specimen.iloc[0]].index
        curr[max_spec] += len(indices) # keep curr current
        groups[n_split] += list(indices)
        df = df.drop(indices) # drop from train_df

        doneness = [is_done(i) for i in range(len(goals))]
        if all(doneness):
          break

    # add a few unique species per split
    for j in range(2):
      for n_split in range(self.n_splits):
        sample = df[df.species >= 16].sample(random_state=self.random_state + i)
        rows = df[df.species_name == sample.species_name.iloc[0]].index
        df = df.drop(rows)
        groups[n_split] += list(rows)
        i += 1

    # add the rest randomly
    while(True):
      i+=1
      n_split = np.argmin([len(group) for group in groups])
      try:
        sample = df.sample(random_state=self.random_state + i)
      except ValueError:
        break
      rows = df[df.specimen == sample.specimen.iloc[0]].index
      df = df.drop(rows)
      groups[n_split] += list(rows)

    if self.shuffle:
      rstate = random.getstate()
      random.seed(self.random_state)
      for group in groups:
        random.shuffle(group)
      random.setstate(rstate)

    self._groups = groups

    flatten = lambda t: [item for sublist in t for item in sublist]
    for i in range(self.n_splits):
      train = flatten([group for j, group in enumerate(groups) if j != i])
      test = groups[i]
      yield (train, test)


def validate_SpeciesStratifiedShuffleSplit(closed_config, open_config):
  splitter = SpeciesBalancedTestTrainSplitter(closed_config, open_config, random_seed=cfg.SEED)
  x, y, groups = splitter.load('train')
  cv = SpeciesStratifiedShuffleSplit(closed_config.DATA_CSV_PATH, n_splits=3, random_state=cfg.SEED, shuffle=True)
  for train, test in cv.split(x, y, groups):
    print(len(train), len(test))
    break
  groups = cv._groups

  df = splitter.load('train', as_df=True)
  df_groups = [df.loc[group] for group in groups]

  for i, df_group in enumerate(df_groups):
    for specimen in df_group.specimen:
      for j, other_group in enumerate(df_groups):
        if j == i:
          continue
        if any(other_group.specimen == specimen):
          raise AssertionError(f"Group {specimen} is in both {i} and {j}")

def speciesBalancedShuffleSplit(id_train, fold=1, species_seed=0, goal_bounds = (-5,2), random_seed=42):
  LOWER_BOUND, UPPER_BOUND = goal_bounds
  data = pd.DataFrame(id_train, columns=['Id'])
  data['Species'] = data["Id"].str.extract(genus_pattern, expand=False).str.split('/').str.join(' ')
  data['specimen'] = data["Id"].str.extract(specimen_pattern, expand=True)
  splits = pd.read_excel(get_unknown_datasplits(fold), sheet_name=species_seed, engine='openpyxl')
  VALID_SPLITS = ['K', 'U1', 'U2']
  train = dict(zip(VALID_SPLITS, [[] for _ in range(len(VALID_SPLITS))]))
  test = dict(zip(VALID_SPLITS, [[] for _ in range(len(VALID_SPLITS))]))
  for split in VALID_SPLITS:
    curr_splits = splits[splits.Set == split][['Species', 'Train', 'Test']]
    for _, row in curr_splits.iterrows():
      curr_data = data[data.Species == row['Species']].copy(deep=True)
      train_goal = round((row['Train'] / (row['Train'] + row['Test'])) * len(curr_data))
      while train_goal > UPPER_BOUND:
        if len(curr_data) == 0:
          break
        group_counts = curr_data[['Id','specimen']].groupby('specimen').count()
        specimen = curr_data.sample(n=1, random_state = random_seed).specimen.iloc[0]
        specimen_group = curr_data[curr_data.specimen == specimen]
        if train_goal - len(specimen_group) < LOWER_BOUND:
          # print(f'train_goal - {train_goal} and {len(specimen_group)} is too big')
          group_counts = curr_data[['Id','specimen']].groupby('specimen').count()
          perfect_group_counts = group_counts[group_counts.Id <= train_goal]
          if len(perfect_group_counts) == 0:
            # print(f"    no perfect group choosing min of {len(group_counts)} choices")
            group_idx = np.argmin(group_counts)
          else:
            # print("    found perfect group")
            group_counts = perfect_group_counts
            group_idx = np.argmax(group_counts)
          specimen = group_counts.reset_index().iloc[group_idx].specimen
          # finish adding min specimen group then break
          specimen_group = curr_data[curr_data.specimen == specimen]
          # print(f"    using group of size {len(specimen_group)}")
        train[split].extend(specimen_group.index)
        train_goal -= len(specimen_group)
        curr_data.drop(specimen_group.index, inplace=True)
      test[split].extend(curr_data.index)  

  train_index = list(chain(*train.values()))
  valid_index = list(chain(*test.values()))
  return train_index, valid_index