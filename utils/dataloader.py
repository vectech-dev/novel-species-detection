# Owned by Johns Hopkins University, created prior to 5/28/2020
from __future__ import print_function, division
import os, glob
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler

from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split
from IPython.core.debugger import set_trace

from skimage import io, transform
import cv2
from fastai.vision import open_image
from .preprocessing import (alb_transform_train, alb_transform_test,
    custom_over_sampler, fastai_transform_train, fastai_transform_test)
from .datasets import prepare_data_df, prepare_trap_df_set2, prepare_trap_df_set4, prepare_trap_df_set6


# import warnings
# warnings.filterwarnings("ignore")

train_images_path = f"./data/train/"
val_images_path = f"./data/val/"
test_images_path = f"./data/test/"
phone_labels_path = f"./data/phone/"

# Class and genus specifications
class_names = sorted(glob.glob(train_images_path + '*'))
class_names = [' '.join(c.split('/')[-1].split('_')[:-1]) for c in class_names]
genus_names = [i.split(' ')[0] for i in class_names]
genus_names = sorted(list(set(genus_names)))
genus_map = {genus_names[i]: i for i in range(len(genus_names))}
classID_to_genusID_map = {i: genus_map[class_names[i].split(' ')[0]] 
                                for i in range(len(class_names))}

# padding code
pad_im = cv2.imread('./data/pad.jpg')
pad_im = cv2.cvtColor(pad_im, cv2.COLOR_BGR2RGB)
top_pad = np.repeat(pad_im[:5,:,:].mean(0, keepdims=True), 80, axis=0).astype('uint8')
bottom_pad = np.repeat(pad_im[-5:,:,:].mean(0, keepdims=True), 80, axis=0).astype('uint8')


def to_one_hot_label(label, num_classes, type='torch'):
    if type=='torch':
        labelt = torch.zeros(num_classes)
    if type=='numpy' or type=='np':
        labelt = np.zeros(num_classes)
    labelt[label] = 1
    return labelt

def load_image(impath, pil=False):
    if pil:
        image = Image.open(impath)

    else:
        image = cv2.imread(impath, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Image not found at {}".format(impath))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image

def make_square(img):
    if img.shape[0] > img.shape[1]:
        img = np.rollaxis(img, 1, 0)
    toppadlen = (img.shape[1] - img.shape[0])//2
    bottompadlen = img.shape[1] - img.shape[0] - toppadlen
    toppad = img[:5,:,:].mean(0, keepdims=True).astype(img.dtype)
    toppad = np.repeat(toppad, toppadlen, 0)
    bottompad = img[-5:,:,:].mean(0, keepdims=True).astype(img.dtype)
    bottompad = np.repeat(bottompad, bottompadlen, 0)
    return np.concatenate((toppad, img, bottompad), axis=0)

def mixup_loader(idx, df, dataset, colors):
    mixid = df.sample()
    # if dataset=="train":
    #     print(mixid)
    ratio = np.random.rand()

    targets1 = df.loc[idx, 'label']
    targets2 = mixid['label'].values[0]
    # print("Target1, ", targets1, type(targets1), dataset)
    # print("Target2, ", targets2, type(targets2), dataset)
    targets = ratio*targets1 + (1-ratio)*targets2

    image1 = load_image(df.loc[idx, 'Id'], dataset, colors)
    image2 = load_image(mixid['Id'].values[0], dataset, colors)
    image = (ratio*image1 + (1-ratio)*image2).round().astype('uint8')
    # print("ids = {}, {}. Ratio = {}".format(df.loc[idx, 'Id'], mixid[0], ratio))
    return image, targets

class MosDataset(Dataset):
    def __init__(self, config, data_df, transformer=None, one_hot_label=False):
        """
        Params:
            data_df: data DataFrame of image name and labels
            imsize: output image size
        """
        super().__init__()
        self.tfms = None
        self.config = config
        if config.genus:
            self.num_genuses = config.num_genuses
        self.num_species = np.array(config.num_species).sum()
        self.transformer = transformer
        self.mixup = config.mixup
        self.images_df = data_df
        self.one_hot_label = one_hot_label

        # One hot
        if config.genus:
            self.images_df['GenusOneHot'] = torch.zeros((len(self.images_df), self.num_genuses))
        self.images_df['SpeciesOneHot'] = torch.zeros((len(self.images_df), self.num_species))
        for i in range(len(self.images_df)):
            if config.genus:
                self.images_df.at[i, 'GenusOneHot'] = to_one_hot_label(self.images_df.loc[i, 'Genus'], self.num_genuses)
            self.images_df.at[i, 'SpeciesOneHot'] = to_one_hot_label(self.images_df.loc[i, 'Species'], self.num_species)

        if self.config.preload_data:
            print('Preloading images...')
            self.imarray = np.zeros((len(self.images_df), self.config.imsize, 
                                        self.config.imsize, 3), dtype='uint8')
            for idx, impath in enumerate(tqdm(self.images_df['Id'])):
                img = load_image(impath)
                img = cv2.resize(img, (self.config.imsize, self.config.imsize), 
                                    interpolation=cv2.INTER_AREA)
                self.imarray[idx,:,:,:] = img

        # Genus labels
        # self.genus_labels = torch.zeros(len(self.images_df), len(genus_names))
        # for i in range(len(self.images_df)):
        #     self.genus_labels[i, classID_to_genusID_map[int(
        #                         self.images_df['label'][i].argmax())]] = 1.0

    def __len__(self):
        return len(self.images_df)

    def __getitem__(self, idx):
        imagename = self.images_df.loc[idx, 'Id']

        if self.config.preload_data:
            image = self.imarray[idx,:,:,:]
        else:
            image = load_image(imagename, pil=False)
#         image = np.concatenate([top_pad, image, bottom_pad], axis=0)
        image = make_square(image)

        if self.one_hot_label:
            label = self.images_df['SpeciesOneHot'][idx]
            if self.config.genus:
                label = (label, self.images_df['GenusOneHot'][idx])
        else:
            label = self.images_df['Species'][idx]
            if self.config.genus:
                label = (label, self.images_df['Genus'][idx])

        if self.transformer:
            image = self.transformer(image=image)['image']

        else:
            image = transform.resize(image, (self.config.imsize, self.config.imsize))

        # print(image.shape)
        image = torch.from_numpy(image).permute(-1, 0, 1).float()

        return image, label

    def getimage(self, idx):
        image, targets = self.__getitem__(idx)
        image = image.permute(1,2,0).numpy()
        imagename = self.images_df.loc[idx, 'Id']
        return image, targets, imagename


    def cmix(self, idx):
        return mixup_loader(idx, self.images_df, self.dataset, self.colors)

def get_data_loaders(config, eval_mode=False, get_dataset=False, one_hot_labels=True, train_df=None,
                     valid_df=None, DatasetClass=MosDataset):
    '''sets up the torch data loaders for training'''
    if train_df is None:
        print("Reading data split from {}".format(config.DATA_CSV_PATH))
        data_df = pd.read_csv(config.DATA_CSV_PATH)

        train_df = data_df[data_df['Split'] == 'Train'].reset_index(drop=True)
        valid_df = data_df[data_df['Split'] == 'Valid'].reset_index(drop=True)
        # test_df = data_df[data_df['Split'].apply(lambda x: 'Test' in x)].reset_index(drop=True)

        if config.known_only:
            train_df = train_df[train_df['Species'].apply(lambda x: x not in config.unknown_classes)].reset_index(drop=True)
            valid_df = valid_df[valid_df['Species'].apply(lambda x: x not in config.unknown_classes)].reset_index(drop=True)
            # test_df = test_df[test_df['Species'].apply(lambda x: x not in config.unknown_classes)].reset_index(drop=True)

    if config.debug and config.reduce_dataset:
        train_df = train_df.loc[:200]

    # False Additions
    # if config.false_additions:
    #     set2 = prepare_trap_df_set2()
    #     set4 = prepare_trap_df_set4()
    #     # set6 = prepare_trap_df_set6(merge=True)
    #     train_df = pd.concat([train_df, set2, set4]).reset_index(drop=True)

    # if config.phone_additions:
    #     phone_train_df = pd.read_csv(phone_labels_path + 'phone_train.csv')
    #     phone_train_df['label'] = phone_train_df['label'].apply(lambda x: \
    #                     one_hot_label(x, num_classes = len(train_df['label'][0])))
    #     train_df = pd.concat([train_df, phone_train_df]).reset_index(drop=True)

    #     phone_val_df = pd.read_csv(phone_labels_path + 'phone_val.csv')
    #     phone_val_df['label'] = phone_val_df['label'].apply(lambda x: \
    #                     one_hot_label(x, num_classes = len(train_df['label'][0])))
    #     valid_df = pd.concat([valid_df, phone_val_df]).reset_index(drop=True)

    # Oversampling
    # if not test_size == 0:
    #     train_df = custom_over_sampler(train_df, factor=2, num_classes=10)

    # set up the transformers
    if eval_mode:
        train_tf = alb_transform_test(config.imsize)
    else:
        print("Data Augmemtation with probability ", config.augment_prob)
        train_tf = alb_transform_train(config.imsize, p=config.augment_prob)
    valid_tf = alb_transform_test(config.imsize)

    # train_tf = train_transformer(imsize)
    # valid_tf = test_transformer(imsize)

    # set up the datasets
    train_dataset = DatasetClass(config=config, data_df=train_df, transformer=train_tf, one_hot_label=one_hot_labels)
    valid_dataset = DatasetClass(config=config, data_df=valid_df, transformer=valid_tf, one_hot_label=one_hot_labels)

    if get_dataset:
        return train_dataset, valid_dataset

    train_sampler = SubsetRandomSampler(range(len(train_dataset)))

    # set up the data loaders
    train_loader = DataLoader(train_dataset,
                                   batch_size=config.batch_size,
                                   shuffle=False,
                                   sampler=train_sampler,
                                   num_workers=config.num_workers,
                                   pin_memory=True,
                                   drop_last=False)

    valid_loader = DataLoader(valid_dataset,
                                   batch_size=config.batch_size,
                                   shuffle=False,
                                   num_workers=config.num_workers,
                                   pin_memory=True,
                                   drop_last=False)

    return train_loader, valid_loader

def get_test_loader(config, test_df=None, DatasetClass=None):
    '''sets up the torch data loaders for testing'''
    if test_df is None:
        data_df = pd.read_csv(config.DATA_CSV_PATH)
        test_df = data_df[data_df['Split'].apply(lambda x: 'Test' in x)].reset_index(drop=True)

        if config.known_only:
            test_df = test_df[test_df['Species'].apply(lambda x: x not in config.unknown_classes)].reset_index(drop=True)

    test_tf = alb_transform_test(config.imsize)

    # test_tf = test_transformer(imsize)

    # set up the datasets
    if DatasetClass is None:
        DatasetClass = MRANDataset if 'mran' in config.model_name else MosDataset
    test_dataset = DatasetClass(config=config, data_df=test_df, transformer=test_tf)

    # set up the data loader
    test_loader = DataLoader(test_dataset,
                                   batch_size=config.batch_size,
                                   shuffle=False,
                                   num_workers=config.num_workers,
                                   pin_memory=True,
                                   drop_last=False)

    return test_loader


def encode_label(genus, species, splen=31):
    return splen*genus + species

def decode_label(comblabel, splen=31):
    return (comblabel//splen, comblabel % splen)

def get_fastai_data_bunch(config, test=False, test_df=None):
    from fastai.vision.data import ImageList, get_transforms, imagenet_stats
    data_df = pd.read_csv(config.DATA_CSV_PATH)

    train_df = data_df[data_df['Split'] == 'Train'].reset_index(drop=True)
    valid_df = data_df[data_df['Split'] == 'Valid'].reset_index(drop=True)
    if test_df is None:
        test_df = data_df[data_df['Split'].apply(lambda x: 'Test' in x)].reset_index(drop=True)

    if config.known_only:
        train_df = train_df[train_df['Species'].apply(lambda x: x not in config.unknown_classes)].reset_index(drop=True)
        valid_df = valid_df[valid_df['Species'].apply(lambda x: x not in config.unknown_classes)].reset_index(drop=True)
        test_df = test_df[test_df['Species'].apply(lambda x: x not in config.unknown_classes)].reset_index(drop=True)

    if config.debug and config.reduce_dataset:
        train_df = train_df.loc[:200]

    train_df['valid'] = False
    valid_df['valid'] = True
    test_df['valid'] = True

    if test:
        trainval_df = pd.concat((train_df, test_df)).reset_index(drop=True)
    else:
        trainval_df = pd.concat((train_df, valid_df)).reset_index(drop=True)

    # sp_onehot = torch.zeros(len(trainval_df), 31)
    # sp_onehot[range(len(trainval_df)), trainval_df['Species']] = 1.
    # trainval_df['Species'] = sp_onehot
    # trainval_df['Species'] = trainval_df['Species'].apply(lambda x: float(x))

    if config.genus:
        splen = config.num_species.sum()
        lls = []
        for i in range(len(trainval_df)):
            lls.append(encode_label(trainval_df.loc[i, 'Genus'], trainval_df.loc[i, 'Species']), splen)
        trainval_df['Label'] = lls
        trainval_df = trainval_df.drop(['Genus', 'Species'], axis=1)
        data = (ImageList.from_df(trainval_df, '/')
                        .split_from_df(col='valid')
                        .label_from_df('Label'))
    else:
        # tlabels, vlabels = [], []
        # num_classes = trainval_df['Species'].nunique()
        # for tl in train_df['Species']:
        #     tlabels.append(one_hot_label(tl, num_classes, 'numpy'))
        # for vl in valid_df['Species']:
        #     vlabels.append(one_hot_label(vl, num_classes, 'numpy'))
        data = (ImageList.from_df(trainval_df, '/')
                        .split_from_df(col='valid')
                        .label_from_df('Species'))
                        # .label_from_lists(tlabels, vlabels))
    data.add_test(test_df['Id'])

    if test:
        data = (data.transform(fastai_transform_test())
                .databunch(bs=config.batch_size, num_workers=config.num_workers)
                .presize(config.imsize, scale=(1,1))
                .normalize(imagenet_stats))
    else:
        data = (data.transform(fastai_transform_train())
                .databunch(bs=config.batch_size, num_workers=config.num_workers)
                .presize(config.imsize, scale=(0.35,1))
                .normalize(imagenet_stats))

    return data


class FastDataset(Dataset):
    def __init__(self, config, data_df=None, transformer=None):
        """
        Params:
            data_df: data DataFrame of image name and labels
            imsize: output image size
        """
        super().__init__()
        self.data_df = data_df
        self.transformer = transformer

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        imagename = self.data_df.loc[idx, 'Id']
        image = open_image(imagename)

        label = self.data_df['Species'][idx]
        if self.config.genus:
            label = (label, self.data_df['Genus'][idx])

        # genus_label = one_hot_label(genus_label, self.num_genuses)
        # species_label = one_hot_label(species_label, self.num_species)

        image = image.apply_tfms(self.transformer)
        return image, label

class MRANDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.dataset = MosDataset(*args, **kwargs)
        self.images_df = self.dataset.images_df
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, labels = self.dataset[idx]
#         horflip_idx = torch.arange(image.size(2)-1, -1, -1).long()
#         image_horflip = image[:,:,horflip_idx]
#         return (image, image_horflip, labels), labels
        return (image, image, labels), labels
