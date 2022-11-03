from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import kerastuner.tuners as kt

import collections
import copy
import numpy as np
import os
from tensorboard.plugins.hparams import api as hparams_api
from tensorflow import keras
from kerastuner.engine import tuner_utils
from kerastuner.engine import tuner as tuner_module

class RandomSearchCV(kt.RandomSearch):
    def __init__(self,
                 hypermodel,
                 objective,
                 max_trials,
                 seed=None,
                 hyperparameters=None,
                 tune_new_entries=True,
                 allow_new_entries=True,
                 cv_iter=None,
                 **kwargs):
        self.cv_indexes = list(cv_iter)
        kwargs['executions_per_trial'] = len(self.cv_indexes)
        super(RandomSearchCV, self).__init__(
            hypermodel,
            objective,
            max_trials,
            seed=seed,
            hyperparameters=hyperparameters,
            tune_new_entries=tune_new_entries,
            allow_new_entries=allow_new_entries,
            **kwargs)

    def _convert_fit_args(self, fit_args, indices):
        x, y = fit_args
        if isinstance(x, tuple):
            x_deep, x_wide = x
            x_idx = (x_deep[indices], x_wide[indices])
        else:
            x_idx = x[indices]
        y_idx = y[indices]
        return x_idx, y_idx      

    def run_trial(self, trial, *fit_args, **fit_kwargs):
        model_checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=self._get_checkpoint_fname(
                trial.trial_id, self._reported_step),
            monitor=self.oracle.objective.name,
            mode=self.oracle.objective.direction,
            save_best_only=True,
            save_weights_only=True)
        original_callbacks = fit_kwargs.pop('callbacks', [])

        # Run the training process multiple times.
        metrics = collections.defaultdict(list)
        for execution in range(len(self.cv_indexes)):
            train_indices, valid_indices = self.cv_indexes[execution]
            train_args = self._convert_fit_args(fit_args, train_indices)
            valid_args = self._convert_fit_args(fit_args, valid_indices)
            copied_fit_kwargs = copy.copy(fit_kwargs)
            callbacks = self._deepcopy_callbacks(original_callbacks)
            self._configure_tensorboard_dir(callbacks, trial, execution)
            callbacks.append(tuner_utils.TunerCallback(self, trial))
            # Only checkpoint the best epoch across all executions.
            callbacks.append(model_checkpoint)
            copied_fit_kwargs['callbacks'] = callbacks
            copied_fit_kwargs['validation_data'] = valid_args
            history = self._build_and_fit_model(trial, train_args, copied_fit_kwargs)
            for metric, epoch_values in history.history.items():
                if self.oracle.objective.direction == 'min':
                    best_value = np.min(epoch_values)
                else:
                    best_value = np.max(epoch_values)
                metrics[metric].append(best_value)
        # Average the results across executions and send to the Oracle.
        averaged_metrics = {}
        for metric, execution_values in metrics.items():
            averaged_metrics[metric] = np.mean(execution_values)
        self.oracle.update_trial(
            trial.trial_id, metrics=averaged_metrics, step=self._reported_step)