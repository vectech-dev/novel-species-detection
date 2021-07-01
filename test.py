# Owned by Johns Hopkins University, created prior to 5/28/2020
import os, sys
import importlib
import glob
import argparse
import time
from tqdm import tqdm
import json
import pprint
from collections import namedtuple
import copy

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from utils.dataloader import get_data_loaders, get_fastai_data_bunch, MRANDataset, MosDataset, get_test_loader
from evaluations import generate_submission, generate_preds
from modules.evaluations_mran import generate_submission as generate_submission_mran
from utils.metrics import accuracy, macro_f1
from utils.misc import save_pred
from utils.losses import MosLoss

# from fastai.vision import Learner
from modules.blend_data_augmentation import Learner
from fastai.basic_data import DatasetType
import models.model_list as model_list

from configs.config import config

parser = argparse.ArgumentParser(description='VecTech Mosquito Classification Test')
parser.add_argument('--config', default=None, 
                    help="Config file to use")
parser.add_argument('--feats_concat', action='store_true', default=False,
                    help="Concatenate features instead of averaging")
parser.add_argument('--full_df', action='store_true', default=False,
                    help="Generate preds for the full data set")
parser.add_argument('--full_test', action='store_true', default=False,
                    help="Generate preds for the full test set (with unknown classes)")
parser.add_argument('--modality', default=None, choices=[None, 'm', 'd', 'p'],
                    help="only test on a particular modality images")
parser.add_argument('--outfile', default='', 
                    help="Append arg to the file name")
parser.add_argument('--loadpath', default='', 
                    help="Append arg to the model load path")
parser.add_argument('--metric', default='acc',
                    help='Load best_metric model')
parser.add_argument('--latest', action='store_true', default=False,
                    help='Load the latest checkpoint (default best)')
parser.add_argument('--submission', action='store_true', default=False,
                    help='Generate submission')
parser.add_argument('--atg', action='store_true', default=False,
                    help='assume the predicted genus is true (all true genus)')
parser.add_argument('--features', action='store_true', default=False,
                    help='send features too')
parser.add_argument('--realtime', default=1, type=int, choices=[0, 1],
                    help='print a cool realtime status')
parser.add_argument('--tta', default=1, type=int, choices=[0, 1],
                    help='0 = no TTA, 1 = D8 TTA')
args = parser.parse_args()

if args.config is not None:
    package = args.config.replace('/', '.')
    while package[0] == '.':
        package = package[1:]
    package = package.replace('.py', '')
    print(package)
    config = importlib.import_module(package).config

print('')
pprint.pprint(config)
time.sleep(1)
print('')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    cudnn.benchmark = True

if not os.path.exists('./preds/'+config.exp_name):
    os.makedirs('./preds/'+config.exp_name) 
if not os.path.exists('./subm/'+config.exp_name):
    os.makedirs('./subm/'+config.exp_name)
if not os.path.exists('./tierI_output/'+config.exp_name):
    os.makedirs('./tierI_output/'+config.exp_name)

def main_pred(net = None, opcon = None, attn=False, features_out=False):
    if opcon is not None:
        config = opcon

    model_params = [config.exp_name, config.model_name, args.metric]
    MODEL_CKPT = './model_weights/{}/best_{}.pth/best_{}{}.pth'.format(*model_params, args.loadpath)
    if args.latest:
        MODEL_CKPT = MODEL_CKPT.replace('best_loss.pth', 'basic_model.pth')

    if net is None:
        Net = getattr(model_list, config.model_name)
        net = Net(config=config)

    print('Loading model from ' + MODEL_CKPT)
    try:
        net.load_state_dict(torch.load(MODEL_CKPT))
    except:
        net.load_state_dict(torch.load(MODEL_CKPT)['model'])

    if features_out:
        net.last_linear = nn.Identity()
#         feature_net = torch.nn.Sequential(*(list(net.children())[:-1]))
        
        PRED_OUT = './subm/{0}/best-features_{2}.csv'.format(*model_params)
    else:
        PRED_OUT = './subm/{0}/best_{2}.csv'.format(*model_params)
        
    net = nn.parallel.DataParallel(net)
    net.to(device)
    net.eval()


    
    
    if args.latest:
        PRED_OUT = PRED_OUT.replace('best', 'latest')
    if args.outfile != '':
        PRED_OUT = PRED_OUT.replace('.csv', '_{}.csv'.format(args.outfile))

#     if features_out:
#         feature_net = nn.parallel.DataParallel(feature_net)
#         feature_net.to(device)
#         feature_net.eval()
        
#         FEAT_OUT = './subm/{0}/best-features_{2}.csv'.format(*model_params)
#         if args.latest:
#             FEAT_OUT = FEAT_OUT.replace('best', 'latest')
#         if args.outfile != '':
#             FEAT_OUT = FEAT_OUT.replace('.csv', '_{}.csv'.format(args.outfile))
        
        
    if config.fastai_data:
        raise NotImplementedError('Not implemented')
#         data_df = pd.read_csv(config.DATA_CSV_PATH)
#         test_df = data_df[data_df['Split'].apply(lambda x: 'Valid' == x)].reset_index(drop=True)
#         if config.known_only:
#             test_df = test_df[test_df['Species'].apply(lambda x: x not in config.unknown_classes)].reset_index(drop=True)
        
#         data = get_fastai_data_bunch(config, test=True, test_df=test_df)
#         loss = MosLoss(config=config)
#         learn = Learner(data, net, wd=config.weight_decay,
#                      bn_wd=False, true_wd=True,
#                      loss_func = loss,
#                     )
#         testpreds = learn.get_preds(ds_type=DatasetType.Test)[0].detach().cpu().numpy()
# #         testpreds = learn.TTA(ds_type=DatasetType.Test)
# #         testpreds = []
# #         with torch.no_grad():
# #             for i, batch in enumerate(tqdm(data.test_dl)):
# #                 out = learn.model(batch[0]).detach().cpu().numpy()
# #                 for idx in range(out.shape[0]):
# #                     testpreds.append(out[idx])
# #         testpreds = np.array(testpreds)
# #         print(testpreds.shape)
# #         print(len(test_df))
        
        
#         test_df = save_pred(config, testpreds, test_df, SUBM_OUT=PRED_OUT, atg=False, save=True)
#         print("Species Accuracy: ", (test_df['Species'] == test_df['SpeciesPred']).sum()/len(test_df))

    else:
        data_df = pd.read_csv(config.DATA_CSV_PATH)
        if args.full_df:
            test_df = data_df
            if config.num_species < test_df['Species'].nunique():
                test_df['Species'] = 0
        elif args.full_test:
            test_df = data_df[data_df['Split'].apply(lambda x: 'Test' in x)].reset_index(drop=True)
            if config.num_species > test_df['Species'].nunique():
                test_df['Species'] = 0
        else:
            test_df = data_df[data_df['Split'].apply(lambda x: 'Test' in x)].reset_index(drop=True)
            if config.known_only:
                test_df = test_df[test_df['Species'].apply(lambda x: x not in config.unknown_classes)].reset_index(drop=True)

        if args.modality is not None:
            test_df = test_df[test_df['Modality'] == args.modality].reset_index(drop=True)
        print("Test set length: {}".format(len(test_df)))

        if features_out:
            net.eval()
            test_loader = get_test_loader(config, test_df=test_df)
            features = generate_preds(net, config, test_loader, test=True, realtime=True, use_tta=args.tta,
                                      feats_concat=args.feats_concat, features_out=features_out).numpy()
#             save_pred(config, features, test_loader.dataset.images_df, PRED_OUT.replace('subm','preds'), atg=args.atg)
            feat_df = test_loader.dataset.images_df['Id'].copy()
            feat_df = pd.concat([feat_df, pd.DataFrame(data=features)], axis=1)
            FEAT_OUT='tierI_output/{}/{}_features.csv'.format(config.exp_name,config.model_name)
            feat_df.to_csv(FEAT_OUT, index=False)
            print("Preds saved to: ",FEAT_OUT)
        else:
            generate_subm = generate_submission
            generate_subm(net, config, SUBM_OUT=PRED_OUT, gen_csv=True, 
                realtime=args.realtime, atg=args.atg, test_df=test_df, use_tta=args.tta, feats_concat=args.feats_concat)
#             generate_subm(feature_net, config, SUBM_OUT=FEAT_OUT, gen_csv=True,
#                           realtime=args.realtime, atg=args.atg, test_df=test_df, use_tta=args.tta,
#                           feats_concat=args.feats_concat,features_out=features_out)


if __name__ == '__main__':
    main_pred(opcon=config,features_out=args.features)
