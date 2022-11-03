import re
import os
import random
import operator
import json
import pandas as pd
import numpy as np
from unknown.utils.paths import DATA_DIR, get_split_fold, get_tier1_csv_path, get_unknown_datasplits
from unknown.config.config import UnknownConfigLoader
cfg = UnknownConfigLoader()

class T1Iterator():
    def __init__(self, trial, splitter, species_seed=1):
        self.trial = trial
        self.fold = splitter.fold
        self.seed = species_seed
        splitter.reshuffle_data(self.seed)
        info = self.trial['t1']
        self.combined = info['combined']
        assert not self.combined, "Combined isn't currently supported"
        assert 'proto' not in info['algs'], 'Only xception is currently supported'
        self.combos = {}
        self.wide = {}
        for alg in info['algs']:
            for num_classes in info['num_classes']:
                for layer_size in info['layer_size']:
                    combo = f'{alg}_{num_classes}_{layer_size}'
                    combo_data = splitter.get_t1_data(alg, num_classes, layer_size)
                    wide_data = splitter.get_t1_data(alg, num_classes, 'probs')
                    self.combos[combo] = combo_data
                    self.wide[combo] = wide_data


    def load_given_combo(self, split, combo, wnn=False):
        x_split, y_split, id_split = self.combos[combo].load(split)
        if wnn:
            x_wide, _, _ = self.wide[combo].load(split)
            x_split = (x_split, x_wide)
        return x_split, y_split, id_split

    def load(self, split, alg='xception', num_classes=None, layer_size='feats', wnn=False):
        combo = f'{alg}_{num_classes}_{layer_size}'
        return self.load_given_combo(split, combo, wnn=wnn)
    
    def iter(self, alg='xception', num_classes=None, layer_size='feats', skip_train=True, wnn=False):
        combo = f'{alg}_{num_classes}_{layer_size}'
        return self.iter_given_combo(combo, skip_train=skip_train, wnn=wnn)
    
    def iter_given_combo(self, combo, skip_train=True, wnn=False):
        splits = cfg.T2_TEST_SPLITS if skip_train else ['train', *cfg.T2_TEST_SPLITS]    
        for split in splits:
            yield split, *self.load_given_combo(split, combo, wnn=wnn)

class Tier1Data():
    def __init__(self, dfs, fold=1, species_seed=0):
        self.dfs = dfs
        self.fold = fold
        self.species_seed = species_seed
        
    def load(self, split):
        df = self.dfs[split]
        y_split = df.y.to_numpy()
        id_split = df.Id.to_numpy()
        x_split = df.drop(columns=['y','Id']).to_numpy()
        return x_split, y_split, id_split

    def iter(self, skip_train=True):
        splits = cfg.T2_TEST_SPLITS if skip_train else ['train', *cfg.T2_TEST_SPLITS]    
        for split in splits:
            yield split, *self.load(split)

class SpeciesBalancedTestTrainSplitter():
    VALID_SPLITS = ['K', 'U1', 'U2', 'N']
    genus_pattern = r'\/opt\/ImageBase\/AWS_sync\/cropped([a-zA-Z\/]*\/[a-zA-Z_\-\d]+)\/[a-zA-Z\/_\-\d]*JHU-\d+_\d+m\.jpg'
    specimen_pattern = r'\/opt\/ImageBase\/AWS_sync\/cropped[a-zA-Z\/]*\/[a-zA-Z_\-\d]+\/[a-zA-Z\/_\-\d]*(JHU-\d+)_\d+m\.jpg'
    
    def __init__(self, closed_config, open_config, initial_seed=0, random_seed=cfg.SEED):
        self.closed_config = closed_config
        self.open_config = open_config
        self.fold = closed_config.FOLD
        self.species_seed = None
        self.train = None
        self.test = None
        self.random_seed=random_seed
        self.data = SpeciesBalancedTestTrainSplitter._get_dataset_from_splits(self.closed_config.DATA_CSV_PATH)
        self.reshuffle_data(initial_seed)

    def reshuffle_data(self, species_seed):
        if self.species_seed != species_seed:
            self.species_seed = species_seed
            splits = pd.read_excel(get_unknown_datasplits(self.fold), sheet_name=species_seed, engine='openpyxl')
            self.train, self.test = self._test_train_split(splits)

    @staticmethod
    def _get_dataset_from_splits(data_csv_path: str):
        df_split = pd.read_csv(get_split_fold(data_csv_path))[['Id', 'Split']]
        df_split.sort_values(by='Id', inplace=True, ignore_index=True)
        df_split['Species'] = df_split["Id"].str.extract(SpeciesBalancedTestTrainSplitter.genus_pattern, expand=False).str.split('/').str.join(' ')
        df_split['specimen'] = df_split["Id"].str.extract(SpeciesBalancedTestTrainSplitter.specimen_pattern, expand=True)
        df = df_split[(df_split.Split == 'Test') | (df_split.Split == 'ExtendedTest')].drop(columns=['Split'])
        return df

    def _test_train_split(self, splits, goal_bounds = (-5,2), MAX_SKIPS = 5):
        LOWER_BOUND, UPPER_BOUND = goal_bounds
        # load data
        
        train = dict(zip(SpeciesBalancedTestTrainSplitter.VALID_SPLITS, [[] for _ in range(len(SpeciesBalancedTestTrainSplitter.VALID_SPLITS))]))
        test = dict(zip(SpeciesBalancedTestTrainSplitter.VALID_SPLITS, [[] for _ in range(len(SpeciesBalancedTestTrainSplitter.VALID_SPLITS))]))

        for split in SpeciesBalancedTestTrainSplitter.VALID_SPLITS:
            curr_splits = splits[splits.Set == split][['Species', 'Train', 'Test']]
            for _, row in curr_splits.iterrows():
                curr_data = self.data[self.data.Species == row['Species']].copy(deep=True)
                train_goal = round((row['Train'] / (row['Train'] + row['Test'])) * len(curr_data))
                num_skip = 0
                while train_goal > UPPER_BOUND:
                    if len(curr_data) == 0:
                        break
                    group_counts = curr_data[['Id','specimen']].groupby('specimen').count()
                    specimen = curr_data.sample(n=1, random_state = self.random_seed).specimen.iloc[0]
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
        return train, test

    def get_t1_data(self, alg, num_classes, layer_size, species_seed=None):
        ''' Get single T1 data '''

        def _get_partial_by_split(df, split_df, split):
            ''' helper fn for splitting part of train/test '''
            partial = df.loc[split_df[split]].copy(deep=True).reset_index(drop=True)
            partial['y'] = pd.Series([int(split != 'K')] * len(partial), dtype=int)
            return partial
        # reshuffle data if necessary
        if species_seed is not None:
            self.reshuffle_data(species_seed)
        # load x data
        exp_name = self.closed_config.exp_name if num_classes == 16 else self.open_config.exp_name
        fpath = get_tier1_csv_path(exp_name, layer_size=layer_size)
        df = pd.read_csv(fpath).sort_values(by='Id', ignore_index=True)

        # build dfs
        dfs = {
            'train': pd.concat([_get_partial_by_split(df, self.train, split) for split in SpeciesBalancedTestTrainSplitter.VALID_SPLITS], axis=0),
            'A': _get_partial_by_split(df, self.test, 'K'),
            'B': _get_partial_by_split(df, self.test, 'U1'),
            'C': _get_partial_by_split(df, self.test, 'U2'),
            'D': _get_partial_by_split(df, self.test, 'N')
        }
        dfs['T1'] = pd.concat([dfs['A'], dfs['B']], axis=0)
        dfs['T2'] = pd.concat([dfs['C'], dfs['D']], axis=0)
        dfs['T'] = pd.concat([dfs['T1'], dfs['T2']], axis=0)

        # return as Tier1Data object
        return Tier1Data(dfs, fold=self.fold, species_seed=self.species_seed)

def validate_splitter(fold=1, species_seed=0):
    splitter = SpeciesBalancedTestTrainSplitter(fold=fold, initial_seed=species_seed)
    data = splitter.get_t1_data('xception', 16, 'feats')
    dfs = data.dfs

    # no Ids are duplicated
    assert (all([
        len(dfs['train']) + len(dfs['T']) == len(splitter.data),
        len(dfs['T1']) + len(dfs['T2']) == len(dfs['T']),
        len(dfs['A']) + len(dfs['B']) == len(dfs['T1']),
        len(dfs['C']) + len(dfs['D']) == len(dfs['T2']) 
        ])), "Ids are duplicated"  

    # no specimen is in multiple sets
    genus_pattern = r'\/opt\/ImageBase\/AWS_sync\/cropped([a-zA-Z\/]*\/[a-zA-Z_\-\d]+)\/[a-zA-Z\/_\-\d]*JHU-\d+_\d+m\.jpg'
    specimen_pattern = r'\/opt\/ImageBase\/AWS_sync\/cropped[a-zA-Z\/]*\/[a-zA-Z_\-\d]+\/[a-zA-Z\/_\-\d]*(JHU-\d+)_\d+m\.jpg'

    def dress(df):
        df['Species'] = df["Id"].str.extract(genus_pattern, expand=False).str.split('/').str.join(' ')
        df['specimen'] = df["Id"].str.extract(specimen_pattern, expand=True)
        return df

    A = dress(dfs['A'])
    B = dress(dfs['B'])
    C = dress(dfs['C'])
    D = dress(dfs['D'])
    train = dress(dfs['train'])

    to_check = {'A': A, 'B': B, 'C': C, 'D': D, 'train': train}
    for split_1, df_1 in to_check.items():
        for split_2, df_2 in to_check.items():
            if split_1 == split_2:
                continue
            else:
                df2_specimens = list(df_2.specimen.unique())
                for specimen in df_1.specimen.unique():
                    assert specimen not in df2_specimens, f"Specimen {specimen} was found in {split_1} and {split_2}"

    # train has 0 species from N
    # set D has all species from N
    splits = pd.read_excel(get_unknown_datasplits(fold), sheet_name=species_seed, engine='openpyxl')
    N_species = list(splits[splits.Set == 'N'].Species.unique())
    setD_species = list(D.Species.unique())
    train_species = list(train.Species.unique())
    assert N_species == setD_species, "setD != setN"
    for species in N_species:
        assert species not in train_species, f"Set N species {species} found in Train"

    # train is within 5 of target on each species
    # no species is in multiple test sets
    T = dress(dfs['T'])
    sets_dict = {'K':A, 'U1':B, 'U2':C}
    train_extra = 0
    test_extra = 0
    for split, test in sets_dict.items():
        curr_splits = splits[splits.Set == split][['Species', 'Train', 'Test']]
        for _, row in curr_splits.iterrows():
            species = row['Species']
            curr_data = splitter.data[splitter.data.Species == species].copy(deep=True)
            train_goal = round((row['Train'] / (row['Train'] + row['Test'])) * len(curr_data))
            test_goal = int((row['Test'] / (row['Train'] + row['Test'])) * len(curr_data))
            # print(f"For species {species}:")
            train_curr = train[train.Species == species]
            test_curr = test[test.Species == species]
            T_curr = T[T.Species == species]
            # print(f'    {len(train_curr) - train_goal} extra samples in Train')
            train_extra += len(train_curr) - train_goal
            # print(f'    {len(test_curr) - test_goal} extra samples in Test')
            test_extra += len(test_curr) - test_goal
            assert len(T_curr) == len(test_curr), f'species {species} ({split}) was found in multiple test sets'
            assert abs(len(train_curr) - train_goal) < 5, f"There was {len(train_curr)} of {species} in training, compared to a target of {train_goal} (Test: {len(test_curr)}/{test_goal})"
            assert abs(len(test_curr) - test_goal) < 5, f"There was {len(test_curr)} of {species} in the test set ({split}), compared to a target of {test_goal} (Train: {len(train_curr)}/{train_goal})"
    print(f"        Train Extra: {train_extra}")
    print(f"        Test Extra: {test_extra}")
