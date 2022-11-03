import os.path
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, PredefinedSplit
# load cfg for seed (must be set before import)
from unknown.config.config import UnknownConfigLoader
cfg = UnknownConfigLoader()
np.random.seed(cfg.SEED) # for reproducibility
import tensorflow as tf
tf.random.set_seed(cfg.SEED) # for reproducibility
from tensorflow import keras
from tensorflow.keras import layers, metrics, optimizers
import kerastuner.tuners as kt
from tensorflow.keras import backend as K

from unknown.loaders.cross_val import SpeciesStratifiedShuffleSplit
from unknown.models.zoo import get_gs_params, get_model
from unknown.search.custom_rs import RandomSearchCV
from unknown.utils.paths import DATA_DIR

def t2_grid_search(
    alg, run_num, config, is_closed, 
    x_train, y_train, id_train,
    random_state=cfg.SEED):
    """ Perform grid search for sklearn model or a random search for keras models (tier2 method) """
    model = get_model(alg, is_closed=is_closed)
    cv = SpeciesStratifiedShuffleSplit(config.DATA_CSV_PATH, n_splits=3, random_state=random_state, shuffle=True)

    if alg in ['rfor', 'svm']:
        search = GridSearchCV(model, get_gs_params(alg), n_jobs=-1, cv=cv)
        search.fit(x_train, y_train, groups=id_train)
        print("Best parameter (CV score=%0.3f):" % search.best_score_)
        print("Best Params: ", search.best_params_)
        best_params = search.best_params_
    else:
        tuner = RandomSearchCV(
                        model,
                        objective='val_accuracy',
                        max_trials=70,
                        cv_iter= cv.split(x_train[0], y_train, id_train),
                        directory=os.path.join(DATA_DIR, 'tier2', 'search', 'wnn'),
                        project_name=f'run{run_num}')
        tuner.search(x_train, y_train, epochs=cfg.KERAS_EPOCHS, workers=-1,
                        batch_size=cfg.KERAS_BATCH_SIZE, callbacks=[keras.callbacks.EarlyStopping(patience=10)])
        print(f"------------ Grid Search Results ------------")
        tuner.results_summary() 
        best_params = tuner.get_best_hyperparameters()[0].get_config()['values']
    return best_params

def t3_grid_search(
    x_train, y_train, id_train,
    x_valid, y_valid, id_valid):
    """ Perform grid search for sklearn model or a random search for keras models (tier2 method) """
    df_train = pd.DataFrame(x_train)
    df_train['y'] = y_train
    df_train['Id'] = id_train
    df_train['split'] = [-1] * len(df_train)
    
    df_valid = pd.DataFrame(x_valid)
    df_valid['y'] = y_valid
    df_valid['Id'] = id_valid
    df_valid['split'] = [0] * len(df_valid)
    
    df = pd.concat([df_train, df_valid])

    y_all = df.pop('y').to_numpy()
    id_all = df.pop("Id").to_numpy()
    test_folds = df.pop('split').to_numpy()
    x_all = df.to_numpy()

    cv = PredefinedSplit(test_folds)
    
    alg = "garb"
    model = get_model(alg)
    search = GridSearchCV(model, get_gs_params(alg), n_jobs=-1, cv=cv)
    search.fit(x_all, y_all, groups=id_all)
    print("Best parameter (CV score=%0.3f):" % search.best_score_)
    print("Best Params: ", search.best_params_)
    best_params = search.best_params_
    return best_params