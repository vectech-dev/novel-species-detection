# Owned by Johns Hopkins University, created prior to 5/28/2020
import numpy as np
import os
import cv2
from easydict import EasyDict
config = EasyDict()
FOLD=2

config.exp_name = "paper_redo/open/fold{}".format(FOLD)
config.model_name = "xception"
config.desc = "open classification"
config.pretrained = True
config.preload_data = False
config.wandb = True
config.wandb_project = "Vectech Classification - Network of Experts"		# "Vectech Classification - Network of Experts"
config.gpu = None
config.fp16 = False
config.num_workers = os.cpu_count()
config.fastai_data = False
config.one_hot_labels = False
if config.fastai_data:
	config.one_hot_labels = False
config.debug = False
# config.debug = True     ; config.reduce_dataset = False


# config.DATA_CSV_PATH = 'data/reduced_training/reduced_split_fold1.csv'
config.DATA_CSV_PATH = 'data/finale/datasplit_fold{}.csv'.format(FOLD) # <--paper datasplit location

config.class_map = [
#     'aedes aedes_aegypti',
    'aedes aedes_albopictus',
    'aedes aedes_dorsalis',
    'aedes aedes_japonicus',
#     'aedes aedes_sollicitans',
    'aedes aedes_taeniorhynchus',
    'aedes aedes_vexans',
    'anopheles anopheles_coustani',
    'anopheles anopheles_crucians',
#     'anopheles anopheles_freeborni',
    'anopheles anopheles_funestus',
    'anopheles anopheles_gambiae',
    'anopheles anopheles_punctipennis',
    'anopheles anopheles_quadrimaculatus',    
    'culex culex_erraticus',
#     'culex culex_pipiens_sl',
    'culex culex_salinarius',
    'psorophora psorophora_columbiae',
    'psorophora psorophora_cyanescens',
    'psorophora psorophora_ferox',
    'aedes aedes_spp',
    'anopheles anopheles_spp',
    'culex culex_spp',
    'psorophora psorophora_spp',
    'mosquito']
# config.class_map = ['aedes','anopheles', 'culex', 'psorophora', 'mosquito']
config.class_map = [(i, config.class_map[i]) for i in range(len(config.class_map))]
config.unknown_classes = [16, 17, 18, 19, 20]
config.known_only = False

# config.num_genuses = 6
config.num_species = np.array([16 if config.known_only else 21])
# config.sphead_scaling = 1
# config.nofe_head_design = "combined" # separate or combined

config.loss_dict = {"FocalLoss": {'weight': 1.0, 'mag_scale': 1.0, 'gamma': 0}}

config.optimizer = "ranger"
config.lr = 1e-2
config.lr_finetune = 1e-4
config.weight_decay = 1e-2
config.alpha = 0.99
config.mom = 0.9 # Momentum
config.eps = 1e-6
config.sched_type = "one_cycle" # LR schedule type (one_cycle/flat_and_anneal)

# config.augment_prob = 0.9
config.blend_params = {
    'size': .15,            # range(0.-1.) You can indicate the size of the patched area/s
    'alpha': 1.,           # This is used to define a proba distribution
    'fixed_proba': 0,      # This overrides alpha proba distribution. Will fix the % of the image that is modified
    'grid': True,          # Determine if patches may overlap or not. With True they do not overlap
    'blend_type': 'cut',   # Options: 'zero', 'noise', 'mix', 'cut', 'random'
    'same_size': False,     # All patches may have the same size or not
    'same_crop': False,    # Cropping patches are from the same subregion as input patches (only with 'mix' and 'cut')
    'same_image': False,   # Cropping patches will be from the same or different images (only with 'mix' and 'cut')
}

config.swa = False
config.cosine_annealing = False
config.drop_rate = 0

config.batch_size = 16
config.epochs = 60
config.imsize = 299
config.augment_prob = 0.001

config.genus = False
# config.genus_weight = 0
# config.species_on = 100

config.phone_additions = False
config.False_additions = False
config.mixup = 0.4
config.ricap = 0    # 0.3
config.ann_start = 0.7 # Annealing start

config.lrfinder = False # Run learning rate finder
config.dump = 0 # Print model; don't train"
config.log_file = "./logs/{}".format(config.exp_name) # Log file name

if config.debug:
    config.wandb = False

