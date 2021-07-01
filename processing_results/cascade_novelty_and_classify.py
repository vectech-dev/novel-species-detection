import os, math
import numpy as np, pandas as pd
REPLACE_STRING = '/opt/ImageBase/AWS_sync/cropped/'


def path_data(path):
    fname = path.split('/')[-1]
    specimen = fname.split('_')[0]
    species = ' '.join(path.replace(REPLACE_STRING,'').replace(fname,'').replace('/',' ').split(' ')[:2])
    return species, fname, specimen


def get_species(df):
	# recives a pandas dataframe of known set
	species_list = [] 

	for i in range(df.shape[0]):
	    species, _, _ = path_data(df.iloc[i,0])
	    if species not in species_list:
	        species_list.append(species)
	
	return species_list


def prep_cascade_df(df, known_species):
	# df should be the whole dataset for that iteration

	uk_truth_list = []
	uk_pred_list = []
	species_truth_list = []
	fname_list = []

	for i in range(df.shape[0]):
	    species, fname, _ = path_data(df.iloc[i,0])
	    
	    if species in known_species:
	        uk_truth= 'Known'
	    else:
	        uk_truth = 'Unknown'

	    pred = df.iloc[i,2]    
	    if pred:
	        uk_pred = 'Unknown'
	    else:
	        uk_pred = 'Known'
	        

	    fname_list.append(fname)
	    uk_truth_list.append(uk_truth)
	    uk_pred_list.append(uk_pred)
	    species_truth_list.append(species)

	df['File'] = fname_list
	df['U-K Truth'] = uk_truth_list
	df['Species Truth'] = species_truth_list
	df['U-K Pred'] = uk_pred_list
    
	return df


if __name__ == '__main__':
	res_dir = 'run75/full'
	save_dir = 'closed/'

	for root,dirs,files in os.walk(res_dir):
		for f in files:
			print(os.path.join(root,f))
			full_df = pd.read_csv(os.path.join(root,f))
			known_species = get_species(full_df[full_df['split']=='A']) 
			cascade_df = prep_cascade_df(full_df,known_species)
			cascade_df.to_csv(os.path.join(save_dir,f.replace('.xlsx','.csv').replace('probas_','')), index = False, header=True)
