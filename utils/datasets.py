# Owned by Johns Hopkins University, created prior to 5/28/2020
import glob
import numpy as np
import pandas as pd 
import torch

def int_to_tensor(x):
    one_hot_label = torch.zeros(len(glob.glob('./data/train/*')))
    one_hot_label[x] = 1
    return one_hot_label

def prepare_data_df(dataset='train'):
    set_classes = glob.glob('./data/{}/*'.format(dataset))
    num_classes = len(set_classes)

    set_list = []
    for cls in set_classes:
        imgs = glob.glob(cls + '/*')
        for img in imgs:
            one_hot_label = torch.zeros(num_classes)
            label = int(cls.split('/')[-1].split('_')[-1])
            one_hot_label[label] = 1
            set_list.append((img, one_hot_label))
    
    set_df = pd.DataFrame(data = set_list, columns=['Id', 'label'])
    return set_df    

def prepare_trap_df_set2():
    class_names = sorted(glob.glob('./data/train/*'))
    class_names = [' '.join(c.split('/')[-1].split('_')[:-1]) for c in class_names]
    class_map = {}
    for i in range(len(class_names)):
        class_map[class_names[i]] = i
    
    set2 = pd.read_excel('/home/vectorweb4/Documents/ImageBase/Trap_Images/set2/mosquitospecimenkey.xlsx')
    
    vial_to_species = dict()
    for i in range(len(set2)):
        vial_to_species[set2.loc[i, 'Vial #'].astype(int)] = set2.loc[i, 'Species']
    
    trap_df = []
    imlist = sorted(glob.glob('/home/vectorweb4/Documents/ImageBase/Trap_Images/set2/Cropped/*'))
    for im in imlist:
        imvial = im.split('/')[-1].split('_')[0].replace('V', '')
        imspecies = vial_to_species[int(imvial)]
        species_id = class_map[imspecies.lower()]
        one_hot_label = torch.zeros(len(class_names))
        one_hot_label[species_id] = 1
        trap_df.append((im, one_hot_label))
        
    trap_df = pd.DataFrame(data = trap_df, columns=['Id', 'label'])
    return trap_df

def prepare_trap_df_set4(miscope=False):
    class_names = sorted(glob.glob('./data/train/*'))
    class_names = [' '.join(c.split('/')[-1].split('_')[:-1]) for c in class_names]
    class_map = {}
    for i in range(len(class_names)):
        class_map[class_names[i]] = i
    
    set4 = pd.read_excel('/home/vectorweb4/Documents/ImageBase/Trap_Images/set4/trap pics.xlsx')
    
    imid_to_species = dict()
    for i in range(len(set4)):
        imid = str(set4.loc[i, 'sticky paper #']) + set4.loc[i, 'mosquito letter']
        imid_to_species[imid] = set4.loc[i, 'species']
    
    trap_df = []
    if miscope:
        imlist = sorted(glob.glob('/home/vectorweb4/Documents/ImageBase/Trap_Images/set4/miscope_trap_pics/*'))
    else:
        imlist = sorted(glob.glob('/home/vectorweb4/Documents/ImageBase/Trap_Images/set4/cropped_trap_pics/*'))

    for im in imlist:
        if miscope:
            imid = im.split('/')[-1].split('_')[0].replace('.jpg', '')
        else:
            imid = ''.join(im.split('/')[-1].split('-')[1:3])

        if imid[-1].isnumeric():
            imid = imid[:-1]
        if imid in imid_to_species:
            imspecies = imid_to_species[imid]
        species_id = class_map[imspecies.lower()]
        one_hot_label = torch.zeros(len(class_names))
        one_hot_label[species_id] = 1
        trap_df.append((im, one_hot_label))
        
    trap_df = pd.DataFrame(data = trap_df, columns=['Id', 'label'])
    return trap_df

def prepare_trap_df_set6(merge=False):
    class_names = sorted(glob.glob('./data/train/*'))
    class_names = [' '.join(c.split('/')[-1].split('_')[:-1]) for c in class_names]
    class_map = {}
    for i in range(len(class_names)):
        class_map[class_names[i]] = i
    
    steps = ['step' + str(i) for i in range(1,7)]
    camid = ['Top', 'Bottom']

    allscns = []
    for c in camid:
        for s in steps:
            allscns.append('./data/set6/cropped/*{}*{}*'.format(c,s))
    
    trap_dfs = {}
    for scn in allscns:
        trap_df = []
        imlist = sorted(glob.glob(scn))

        for im in imlist:
            one_hot_label = torch.zeros(len(class_names))
            one_hot_label[0] = 1    # All Ades Aegypti
            trap_df.append((im, one_hot_label))
        trap_df = pd.DataFrame(data = trap_df, columns=['Id', 'label'])
        scn_id = imlist[0].split('_')[-4] + '_' + imlist[0].split('_')[-2]
        trap_dfs[scn_id] = trap_df

    if merge:
        trap_dfs = pd.concat([trap_dfs[i] for i in trap_dfs.keys()]).reset_index(drop=True)

    return trap_dfs

def prepare_phone_df():
    phone_df = pd.read_csv('./data/phone/phone_test.csv')

    phone_df['label'] = phone_df['label'].apply(int_to_tensor)
    return phone_df

def prepare_superres_df():
    super_dfs = {}
    for i in range(1,7):
        phone_df = pd.read_csv('./data/superresults/superres_step{}.csv'.format(i))

        phone_df['label'] = phone_df['label'].apply(int_to_tensor)
        super_dfs['step_{}'.format(i)] = phone_df

    return super_dfs