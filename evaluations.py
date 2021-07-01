# Owned by Johns Hopkins University, created prior to 5/28/2020
import os, sys
import argparse
import time
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.utils.multiclass import unique_labels

from utils.dataloader import get_data_loaders, get_test_loader
from utils.losses import MosLoss
from utils.metrics import accuracy, macro_f1
from utils.misc import save_pred
from pytorch_toolbelt.inference import functional as TTAF

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generate_preds(net, config, test_loader, test=True, attn=False, realtime=False, use_tta=False, feats_concat=False,features_out=False):
    net.eval() 
    out_dim = np.array(config.num_species).sum()
    if feats_concat:
        out_dim *= 8 

        
    if features_out:
        val_sppreds = torch.Tensor(len(test_loader.dataset), 2048)
        
    else:
        val_sppreds = torch.Tensor(len(test_loader.dataset), out_dim)
        
    if config.genus:
        val_gpreds = torch.Tensor(len(test_loader.dataset), config.num_genuses)
    if not test:
        val_splabels = torch.Tensor(len(test_loader.dataset), out_dim)
        if config.genus:
            val_glabels = torch.Tensor(len(test_loader.dataset), config.num_genuses)
    ci = 0

    t0 = time.time()
    ll = len(test_loader)
    # no gradients during validation
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            valid_imgs = data[0].float().to(device)
            if not test:
                if config.genus:
                    valid_glabels = data[1][0].float().to(device)
                    valid_splabels = data[1][1].float().to(device)
                else:
                    valid_splabels = data[1].float().to(device)

            # get predictions
            if use_tta:
                label_vpreds = d4_image2label(net, valid_imgs, genus=config.genus, feats_concat=feats_concat)
            else:
                label_vpreds = net(valid_imgs)

            if config.genus:
                val_gpreds[ci: ci+label_vpreds[0].shape[0], :] = label_vpreds[0]
                val_sppreds[ci: ci+label_vpreds[1].shape[0], :] = label_vpreds[1]
                ci = ci+label_vpreds[0].shape[0]
            else:
                val_sppreds[ci: ci+label_vpreds.shape[0], :] = label_vpreds
                ci = ci+label_vpreds.shape[0]
            if not test:
                if config.genus:
                    val_glabels[ci: ci+valid_glabels.shape[0], :] = valid_glabels
                    val_splabels[ci: ci+valid_splabels.shape[0], :] = valid_splabels
                else:
                    val_splabels[ci: ci+valid_splabels.shape[0], :] = valid_splabels

            if realtime:
                # make a cool terminal output
                tc = time.time() - t0
                tr = int(tc*(ll-i-1)/(i+1))
                sys.stdout.write('\r')
                sys.stdout.write('B: {:>3}/{:<3} | ETA: {:>4d}s'.
                    format(i+1, ll, tr))
    print('')
    if not test:
        if config.genus:
            return (val_gpreds, val_sppreds), (val_glabels, val_splabels)
        else:
            return (val_sppreds, val_splabels)
    else:
        if config.genus:
            return (val_gpreds, val_sppreds)
        else:
            return val_sppreds

def generate_submission(net, config, SUBM_OUT=None, gen_csv=True, atg=False, realtime=False,
                        test_df=None, use_tta=True, feats_concat=False, features_out=False):
    print('Generating predictions....')

    net.eval()

    test_loader = get_test_loader(config, test_df=test_df)

    
    if config.genus:
        test_gpreds, test_sppreds = generate_preds(net, config, test_loader, test=True, realtime=realtime, use_tta=use_tta, feats_concat=feats_concat)
        test_gpreds = test_gpreds.numpy()
        test_sppreds = test_sppreds.numpy()
    else:
        test_sppreds = generate_preds(net, config, test_loader, test=True, realtime=realtime, use_tta=use_tta, feats_concat=feats_concat, features_out=features_out).numpy()

    loss = MosLoss(config=config)

    if config.genus:
        glabels = torch.stack(list(test_loader.dataset.images_df['GenusOneHot']))
        gloss = loss(torch.Tensor(test_gpreds), glabels)
        gf1 = f1_score(test_gpreds.argmax(1), glabels.numpy().argmax(1), average='macro')
        gacc = accuracy_score(test_gpreds.argmax(1), glabels.numpy().argmax(1))
        print('Avg Test Genus Loss: {:.4}, Avg Test Genus Macro F1: {:.4}, Avg Test Genus Acc. {:.4}'.
                format(gloss, gf1, gacc))
    
    if feats_concat:
        print("Extracted features")
    else:
        splabels = torch.stack(list(test_loader.dataset.images_df['SpeciesOneHot']))
        sploss = loss(torch.Tensor(test_sppreds), splabels)
        spf1 = f1_score(test_sppreds.argmax(1), splabels.numpy().argmax(1), average='macro')
        spacc = accuracy_score(test_sppreds.argmax(1), splabels.numpy().argmax(1))

        print('Avg Test Species Loss: {:.4}, Avg Test Species Macro F1: {:.4}, Avg Test Species Acc. {:.4}'.
            format(sploss, spf1, spacc))

    if gen_csv:
        SPSUBM_OUT = 'tierI_output/{}/{}_probabilities.csv'.format(config.exp_name,config.model_name)
        subm_df = test_loader.dataset.images_df['Id'].copy()
        subm_df = pd.concat([subm_df, pd.DataFrame(data=test_sppreds)], axis=1)
        subm_df.to_csv(SPSUBM_OUT, index=False)
        print("Preds saved to: ", SPSUBM_OUT)
        preds = test_sppreds

        if config.genus:
            GSUBM_OUT = SUBM_OUT.replace('.csv', '_genus.csv').replace('subm', 'preds')
            pd.DataFrame(data=test_gpreds).to_csv(GSUBM_OUT, index=False)
            preds = (test_gpreds, test_sppreds)

        save_pred(config, preds, test_loader.dataset.images_df, SUBM_OUT, atg=atg)

    return preds


def d4_image2label(model, image, genus=False, feats_concat=False):
    """Test-time augmentation for image classification that averages predictions
    of all D4 augmentations applied to input image.
    :param model: Model to use for making predictions.
    :param image: Model input.
    :return: Arithmetically averaged predictions
    """
    output = model(image)
#     print('size of output: ',output.size())
    
    for aug in [TTAF.torch_rot90, TTAF.torch_rot180, TTAF.torch_rot270]:
        x = model(aug(image))
        if genus:
            output = (output[0] + x[0], output[1] + x[1])
        else:
            if feats_concat:
                output = torch.cat([output, x], axis=1)
            else:
                output += x

    image = TTAF.torch_transpose(image)

    for aug in [TTAF.torch_none, TTAF.torch_rot90, TTAF.torch_rot180, TTAF.torch_rot270]:
        x = model(aug(image))
        if genus:
            output = (output[0] + x[0], output[1] + x[1])
        else:
            if feats_concat:
                output = torch.cat([output, x], axis=1)
            else:
                output += x
    
    if feats_concat:
        rev_average = 1.
    else:
        rev_average = float(1.0 / 8.0)
    if genus:
        return (output[0] * rev_average, output[1] * rev_average)
    else:
        return output * rev_average