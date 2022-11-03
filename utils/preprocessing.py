# Owned by Johns Hopkins University, created prior to 5/28/2020
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, 
    Compose, RandomGamma, ElasticTransform, ChannelShuffle,RGBShift, Rotate, Normalize,
    Resize, InvertImg
)
from .misc import label_gen_np, SaltAndPepper, Presize

from fastai.vision import (RandTransform, TfmCrop, TfmAffine, TfmLighting, TfmCoord, crop_pad,
    dihedral_affine, symmetric_warp, rotate, zoom, brightness, contrast)

# https://pytorch.org/docs/master/torchvision/models.html
torchvision_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])

def train_transformer(imsize = 256):
    train_tf = transforms.Compose([
        transforms.ToPILImage(),
        # resize the image to 64x64 (remove if images are already 64x64)
        transforms.Resize((imsize, imsize)),
        transforms.RandomRotation(40.0),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        torchvision_normalize
        ]) 
    return train_tf

def test_transformer(imsize = 256):
    test_tf = transforms.Compose([
        transforms.ToPILImage(),
        # resize the image to 64x64 (remove if images are already 64x64)
        transforms.Resize((imsize, imsize)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        torchvision_normalize
        ]) 
    return test_tf

def alb_transform_train(imsize = 256, p=0.2):
    albumentations_transform = Compose([
    Presize(zoom_limit=2.),
    Resize(imsize, imsize), 
    RandomRotate90(),
    Flip(),
    Transpose(),
    # SaltAndPepper(snp_limit=0.05, p=p),
    # InvertImg(),
    # ChannelShuffle(),
    # OneOf([
    #         IAAAdditiveGaussianNoise(loc=0, scale=(2.55, 12.75), per_channel=False, always_apply=False, p=p),
    #         GaussNoise(var_limit=(10., 50.), always_apply=False, p=p),
    #     ], p=0.75*p),
    # OneOf([
    #         MotionBlur(p=0.75*p),
    #         MedianBlur(blur_limit=15, p=0.5*p),
    #         Blur(blur_limit=15, p=0.5*p),
    #     ], p=p),
    ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
    # OneOf([
    #     OpticalDistortion(p=p),
    #     GridDistortion(p=0.5*p),
    #     IAAPiecewiseAffine(p=p),
    #     ], p=0.75*p),
    # OneOf([
    #     CLAHE(clip_limit=2),
    #     IAASharpen(),
    #     IAAEmboss(),
    #     RandomContrast(),
    #     RandomBrightness(),
    #     ], p=p),
    # Normalize(
    #     mean=[0.485, 0.456, 0.406],
    #     std=[0.229, 0.224, 0.225]
    #     )
    Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
        )
    ], p=1)
    return albumentations_transform

# def alb_transform_test(imsize = 256, p=0.2):
#     albumentations_transform = Compose([
#     Resize(imsize, imsize), 
#     RandomRotate90(),
#     Flip(),
#     Transpose(),
#     SaltAndPepper(snp_limit=0.05),
#     # InvertImg(),
#     # ChannelShuffle(),
#     OneOf([
#             IAAAdditiveGaussianNoise(loc=0, scale=(2.55, 12.75), per_channel=False, always_apply=False, p=p),
#             GaussNoise(var_limit=(10., 50.), always_apply=False, p=p),
#         ], p=0.75*p),
#     OneOf([
#             MotionBlur(p=0.75*p),
#             MedianBlur(blur_limit=15, p=0.5*p),
#             Blur(blur_limit=15, p=0.5*p),
#         ], p=p),
#     ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=p),
#     OneOf([
#         OpticalDistortion(p=p),
#         GridDistortion(p=0.5*p),
#         IAAPiecewiseAffine(p=p),
#         ], p=0.75*p),
#     OneOf([
#         CLAHE(clip_limit=2),
#         IAASharpen(),
#         IAAEmboss(),
#         RandomContrast(),
#         RandomBrightness(),
#         ], p=p),
#     # Normalize(
#     #     mean=[0.485, 0.456, 0.406],
#     #     std=[0.229, 0.224, 0.225]
#     #     )
#     Normalize(
#         mean=[0.5, 0.5, 0.5],
#         std=[0.5, 0.5, 0.5]
#         )
#     ], p=1)
#     return albumentations_transform

def alb_transform_test(imsize = 256, p=1):
    albumentations_transform = Compose([
    Resize(imsize, imsize), 
    # RandomRotate90(),
    # Flip(),
    # Normalize(
    #     mean=[0.485, 0.456, 0.406],
    #     std=[0.229, 0.224, 0.225]
    #     )
    Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
        )
    ], p=1)
    return albumentations_transform


def fastai_transform_train():
    train_tf = [
        RandTransform(tfm=TfmCrop (crop_pad), kwargs={'row_pct': (0, 1), 'col_pct': (0, 1), 'padding_mode': 'reflection'}, p=1.0, resolved={}, do_run=True, is_random=True, use_on_y=True),
        RandTransform(tfm=TfmAffine (dihedral_affine), kwargs={}, p=1.0, resolved={}, do_run=True, is_random=True, use_on_y=True),
        RandTransform(tfm=TfmCoord (symmetric_warp), kwargs={'magnitude': (-0.2, 0.2)}, p=0.75, resolved={}, do_run=True, is_random=True, use_on_y=True),
        RandTransform(tfm=TfmAffine (rotate), kwargs={'degrees': (-10.0, 10.0)}, p=0.75, resolved={}, do_run=True, is_random=True, use_on_y=True),
        RandTransform(tfm=TfmAffine (zoom), kwargs={'scale': (1.0, 1.1), 'row_pct': (0, 1), 'col_pct': (0, 1)}, p=0.75, resolved={}, do_run=True, is_random=True, use_on_y=True),
        RandTransform(tfm=TfmLighting (brightness), kwargs={'change': (0.4, 0.6)}, p=0.75, resolved={}, do_run=True, is_random=True, use_on_y=True),
        RandTransform(tfm=TfmLighting (contrast), kwargs={'scale': (0.8, 1.25)}, p=0.75, resolved={}, do_run=True, is_random=True, use_on_y=True)
        ]
 
    test_tf = [
        RandTransform(tfm=TfmCrop (crop_pad), kwargs={}, p=1.0, resolved={}, do_run=True, is_random=True, use_on_y=True)
        ]

    return train_tf, test_tf


def fastai_transform_test():
    train_tf = [
        RandTransform(tfm=TfmCrop (crop_pad), kwargs={'row_pct': (0, 1), 'col_pct': (0, 1), 'padding_mode': 'reflection'}, p=1.0, resolved={}, do_run=True, is_random=True, use_on_y=True)
        ]
 
    test_tf = [
        RandTransform(tfm=TfmCrop (crop_pad), kwargs={}, p=1.0, resolved={}, do_run=True, is_random=True, use_on_y=True)
        ]

    return train_tf, test_tf


def custom_over_sampler(df, factor=5, num_classes=10):
    worst_classes = [20, 27, 16, 26, 18, 17, 22,  6, 13, 12, 21, 19, 15, 25,
                        5, 11,  3, 24,  8,  4,  2,  7, 23,  9,  0, 14,  1, 10]
    cto = worst_classes[:num_classes]
    np_df = df['Target'].apply(label_gen_np)
    c = []
    for i in range(len(df)):
        if(np.any(np_df[i][cto] == 1)):
            c.append(i)
    df = df.append([df.iloc[c]]*(factor-1),ignore_index=True)
    return df
