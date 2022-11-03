import numpy as np
# load cfg for seed (must be set before import)
from unknown.config.config import UnknownConfigLoader
cfg = UnknownConfigLoader()
np.random.seed(cfg.SEED) # for reproducibility
import tensorflow as tf
tf.random.set_seed(cfg.SEED) # for reproducibility
from tensorflow import keras
from tensorflow.keras import layers, metrics, optimizers

gs_params = {
    'n_neurons': {'min_value': 600, 'max_value': 2000, 'step': 100, 'default': 1900},
    'hidden_layers': {'min_value': 3, 'max_value': 6, 'step': 1, 'default': 4},
    'dropout': [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8],
    'first_hidden': {'min_value': 1000, 'max_value': 3000, 'step': 500, 'default': 2500},
    'lr': [1e-3, 3e-3, 1e-2],
    'decay_steps': {'min_value': 244, 'max_value': 732, 'step': 1, 'default': 488}, # decay between 5 and 15 epochs
}

default_params = lambda is_closed: {
      "deep_input_shape": [2048],
      "wide_input_shape": [16 if is_closed else 21],
      "activation_fn": "elu",
      "kernel_distribution": "normal",
      "output_activation_fn": "sigmoid",
      "loss_fn": "binary_crossentropy",
      "optimizer_fn": "adam",
      "batch_norm": True,
      "metrics": ["accuracy"],
      "adam__beta_1": 0.9,
      "adam__beta_2": 0.999,
      "adam__epsilon": 1e-07                
    }

def get_initializer(activation_fn, kernel_distribution, random_seed=cfg.SEED):
    """ Get default initializer given kernel distribution and activation function"""
    if activation_fn in ['elu', 'relu']:
        if kernel_distribution == 'normal':
            return keras.initializers.he_normal(seed=random_seed)
        elif kernel_distribution == 'uniform':
            return keras.initializers.he_uniform(seed=random_seed)
    elif activation_fn in ['tanh', 'logistic', 'softmax', 'softplus', 'softsign']:
        if kernel_distribution == 'normal':
            return keras.initializers.glorot_normal(seed=random_seed)
        elif kernel_distribution == 'uniform':
            return keras.initializers.glorot_uniform(seed=random_seed)
    elif activation_fn == 'selu':
        if kernel_distribution == 'normal':
            return keras.initializers.lecun_normal(seed=random_seed)
        elif kernel_distribution == 'uniform':
            return keras.initializers.lecun_uniform(seed=random_seed)
    raise NotImplementedError(f"Activation fn {activation_fn} ({kernel_distribution}) has no valid initializer candidates")

def build_wnn(
    hidden_layers=3, n_neurons=300, first_hidden=None,
    lr=3e-3, decay_steps=10000, dropout=0.2, activation_fn='elu', kernel_distribution='normal',
    deep_input_shape=[4096], wide_input_shape=[16],
    output_activation_fn='sigmoid', loss_fn='binary_crossentropy', 
    optimizer_fn='adam', metrics=['accuracy'], batch_norm=False, **kwargs
    ):
    """ Build wnn given params"""
    if first_hidden is None:
        first_hidden = n_neurons

    # interpret activation fn args
    initializer = get_initializer(activation_fn, kernel_distribution, random_seed=cfg.SEED)
    batch_norm_kwargs ={ k[len('batch_norm__'):] : v for k, v in kwargs.items() if k.startswith('batch_norm__')}
    bias = not batch_norm

    # deep input layer
    deep_input = layers.Input(shape=deep_input_shape, name="deep_input")
    prev = deep_input
    if batch_norm:
        prev = layers.BatchNormalization(**batch_norm_kwargs, name='deep_input_bn')(prev)
    prev = layers.Dropout(rate=dropout, name='deep_input_dropout')(prev)

    # deep hidden layers 
    for i in range(hidden_layers):
        n = n_neurons if i > 0 else first_hidden
        prev = layers.Dense(n, kernel_initializer=initializer, use_bias=bias, name=f'hidden{i}')(prev)
        if batch_norm:
            prev = layers.BatchNormalization(**batch_norm_kwargs, name=f'hidden{i}_bn')(prev)
        prev = layers.Activation(activation_fn, name=f'hidden{i}_activation')(prev)
        prev = layers.Dropout(rate=dropout, name=f'hidden{i}_dropout')(prev)

    # wide input layer
    wide_input = layers.Input(shape=wide_input_shape, name="wide_input") 
    concat = layers.concatenate([wide_input, prev])

    # output layer
    output = layers.Dense(1, activation=output_activation_fn)(concat)
    model = keras.Model(inputs=[deep_input, wide_input], outputs=[output])

    # optimizer
    opt_prefix = f'{optimizer_fn}__'
    opt_kwargs ={ k[len(opt_prefix):] : v for k, v in kwargs.items() if k.startswith(opt_prefix)}
            
    if optimizer_fn == 'adam':
        learning_rate = keras.optimizers.schedules.ExponentialDecay(lr, decay_steps, 0.1)
        optimizer = optimizers.Adam(learning_rate=learning_rate, **opt_kwargs)
    else:
        raise NotImplementedError(f"Optimizer fn {optimizer_fn} is not supported at this time.")
    
    # compile model
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=metrics)
    return model

def build_wnn_factory(alg, is_closed):
    """ build a build_model function for random search """
    def build_model(hp):
        params = default_params(is_closed).copy()
        for k,v in gs_params.items():
            if isinstance(v, list):
                params[k] = hp.Choice(k, values=v)
            elif isinstance(v, dict):
                assert all([isinstance(v.get(i, None), int) for i in ['min_value', 'max_value', 'step']]), "malformed gs params"
                params[k] = hp.Int(k, **v)
        return build_wnn(**params)
    return build_model 