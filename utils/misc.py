# Owned by Johns Hopkins University, created prior to 5/28/2020
import argparse

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from albumentations.core.transforms_interface import DualTransform
from scipy.special import softmax

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def label_gen_tensor(labelstr):
    label = torch.zeros(28)
    labelstr = labelstr.split()
    for l in labelstr:
        label[int(l)]=1
    return label

def label_gen_np(labelstr):
    label = np.zeros(28, dtype='uint8')
    labelstr = labelstr.split()
    for l in labelstr:
        label[int(l)]=1
    return label

def get_class_map(df):
    class_map = {}
    for i in range(len(df)):
        class_map[df.loc[i, 'Species']] = df.loc[i, 'Species_Name']
    return class_map

def save_pred(config, pred, image_df, SUBM_OUT='./subm/submission.csv', atg=False, save=True):
    if type(pred) == tuple:
        image_df['GenusPred'] = pred[0].argmax(1)
        image_df['SpeciesPred'] = pred[1].argmax(1)
        image_df['GenusPredConfidence'] = softmax(pred[0], axis=1).max(1)
        image_df['SpeciesPredConfidence'] = softmax(pred[1], axis=1).max(1)
        if atg:
            g_to_sp_map = {}
            start = 0
            for i in range(len(config['num_species'])):
                g_to_sp_map[i] = slice(start, start+config['num_species'][i]), start
                start += config['num_species'][i]

            for i in range(len(image_df)):
                sp_range, start = g_to_sp_map[image_df.loc[i, 'GenusPred']]
                image_df.loc[i, 'SpeciesPred'] = pred[1][i, sp_range].argmax() + start
                image_df.loc[i, 'SpeciesPredConfidence'] = softmax(pred[1][i, sp_range]).max()

        image_df['GenusPredConfidence'] = image_df['GenusPredConfidence'].map(lambda x: '{0:.4f}'.format(x))
        cols = ['Id', 'Genus', 'GenusPred', 'GenusPredConfidence', 'Species',
                 'SpeciesPred', 'SpeciesPredConfidence']

    else:
        image_df['SpeciesPred'] = pred.argmax(1)
        image_df['SpeciesPredConfidence'] = softmax(pred, axis=1).max(1)
        if 'Genus' in image_df.columns:
            cols = ['Id', 'Genus', 'Species', 'SpeciesPred', 'SpeciesPredConfidence']
        else:
            cols = ['Id', 'Species', 'SpeciesPred', 'SpeciesPredConfidence']

    image_df['SpeciesPredConfidence'] = image_df['SpeciesPredConfidence'].map(lambda x: '{0:.4f}'.format(x))
    image_df = image_df[cols]
    if 'GenusOneHot' in image_df.columns:
        image_df.drop('GenusOneHot', axis=1, inplace=True)
    if 'SpeciesOneHot' in image_df.columns:
        image_df.drop('SpeciesOneHot', axis=1, inplace=True)

    if save:
        image_df.to_csv(SUBM_OUT, header=True, index=False)
        print('Saved to ', SUBM_OUT)
    return image_df

def get_top_preds(preds, top_n=3):
    outputs = []
    conf = torch.nn.functional.softmax(preds, dim=1)
    for i in range(top_n):
        ci = conf.max(dim=1)
        outputs.append(ci)
        for i in range(conf.shape[0]):
            conf[i, ci[1][i]]=0
    return outputs
    
def log_metrics(train_losses, valid_losses, valid_f1s, lr_hist, e, model_ckpt, config):
    _, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes[0, 0].plot(train_losses)
    axes[0, 1].plot(valid_losses)
    axes[1, 0].plot(valid_f1s)
    axes[1, 1].plot(lr_hist)
    axes[0, 0].set_title('Train Loss')
    axes[0, 1].set_title('Val Loss')
    axes[1, 0].set_title('Val F1')
    axes[1, 1].set_title('LR History')
    plt.suptitle("At Epoch {}, desc: {}".format(e+1, config.desc), fontsize=16)
    plt.savefig(model_ckpt.replace('model_weights', 'logs').replace('.pth', '.png'))
    plt.close('all')

def cosine_annealing_lr(min_lr, max_lr, cycle_size, epochs, cycle_size_inc = 0):
    new_epochs = cycle_size
    n_cycles = 1
    temp_cs = cycle_size
    while (new_epochs <= epochs-temp_cs):
        temp_cs += cycle_size_inc
        new_epochs += temp_cs
        n_cycles += 1
    print("Performing {} epochs for {} cycles".format(new_epochs, n_cycles))
    
    cycle_e = 0
    lr = []
    cycle_ends = [0]
    for e in range(new_epochs):
        lr.append(min_lr + 0.5*(max_lr - min_lr)*(1 + np.cos(cycle_e*np.pi/cycle_size)))
        cycle_e += 1
        if cycle_e == cycle_size:
            cycle_ends.append(cycle_e + cycle_ends[-1])
            cycle_e = 0
            cycle_size += cycle_size_inc
    cycle_ends = np.array(cycle_ends[1:]) - 1
    return lr, cycle_ends

class SaltAndPepper(DualTransform):
    """Flip the input either horizontally, vertically or both horizontally and vertically.
    Args:
        p (float): probability of applying the transform. Default: 0.5.
    Targets:
        image, mask, bboxes, keypoints
    Image types:
        uint8, float32
    """
    def __init__(self, snp_limit=0.001, always_apply=False, p=.5):
        super(DualTransform, self).__init__(always_apply, p)
        self.snp_limit = snp_limit
    
    def salt_and_pepper(self, image):
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = self.snp_limit
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
        out[tuple(coords)] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
        out[tuple(coords)] = 0
        return out
#         return np.ascontiguousarray(img)

    def apply(self, img, p=0.1, **params):
        """Args:
        d (int): code that specifies how to flip the input. 0 for vertical flipping, 1 for horizontal flipping,
                -1 for both vertical and horizontal flipping (which is also could be seen as rotating the input by
                180 degrees).
        """
        return self.salt_and_pepper(img)

    def get_params(self):
        # Random number in the range [0, 1]
        return {'p': np.random.random()/50}

    def apply_to_bbox(self, bbox, **params):
        return bbox

    def apply_to_keypoint(self, keypoint, **params):
        return keypoint

class Presize(DualTransform):
    """Randomly zoom into a part of the input.
    Args:
        zoom_limit (int): the maximum zoom that can be applied
        p (float): probability of applying the transform. Default: 1.
    Targets:
        image, mask, bboxes, keypoints
    Image types:
        uint8, float32
    """

    def __init__(self, zoom_limit, always_apply=True, p=1):
        super(Presize, self).__init__(always_apply, p)
        assert zoom_limit >= 1, 'Zoom limit should be greater than 1'
        self.zoom_limit = zoom_limit
    
    def get_params(self):
        new_size = np.random.uniform(1./self.zoom_limit, 1.)
        top = np.random.uniform(0, 1 - new_size)
        left = np.random.uniform(0, 1 - new_size)
        return {"new_size": new_size, "top": top, "left": left}

    def apply(self, img, new_size, top, left, **params):
        new_size_px = np.round(img.shape[0]*new_size).astype('int')
        top_px = np.round(img.shape[0]*top).astype('int')
        left_px = np.round(img.shape[1]*left).astype('int')
        return img[top_px:top_px+new_size_px, left_px:left_px+new_size_px]

    def apply_to_bbox(self, bbox, **params):
        # Bounding box coordinates are scale invariant
        raise

    def apply_to_keypoint(self, keypoint, **params):
        raise

    def get_transform_init_args_names(self):
        return ("zoom_limit")

def f1_loss_keras(y_true, y_pred):
    from keras import backend as K
    import tensorflow as tf
    
    #y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), THRESHOLD), K.floatx())
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1-K.mean(f1)

def f1_keras(y_true, y_pred, THRESHOLD = 0.05):
    from keras import backend as K
    import tensorflow as tf

    #y_pred = K.round(y_pred)
    y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), THRESHOLD), K.floatx())
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)