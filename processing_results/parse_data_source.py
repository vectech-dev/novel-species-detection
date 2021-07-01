import os, sys, pandas as pd
from utils import corrections as cr


LAB_COLONY = ['WRBU', 'WRAR','FMEL','JHSPH']
WILDEGG_LABGROWN = [ 'MEAD']
UNSURE = ['USMA']



def get_species(df):
	taxa_list = []
	idtype_list = [] 
	for i in range(df.shape[0]):
		human_genus = df.iloc[i,6]
		human_species = df.iloc[i,7]
		dna_genus = df.iloc[i,14]
		dna_species = df.iloc[i,15]
		if type(dna_genus) is str and type(dna_species) is str:
			taxa = cr.correct(dna_genus+ ' '+ dna_species)
			idtype = 'DNA'
		elif type(human_genus) is str and type(human_species) is str:
			taxa = cr.correct(human_genus+ ' '+ human_species)
			idtype = 'Human'
		else:
			taxa = None
			idtype = None
		taxa_list.append(taxa)
		idtype_list.append(idtype)
	return taxa_list, idtype_list

def get_source_lists(df):
	source_list = []
	specimen_list = []
	ext_code_list = []

	for i in range(df.shape[0]):
		specimen=df.iloc[i,0]

		ext_code=df.iloc[i,1]
		if type(ext_code) is str:
			ext_code=ext_code.split('-')[0]
			if ext_code in LAB_COLONY:
				source = "LAB_COLONY"
			elif ext_code in WILDEGG_LABGROWN:
				source = "WILDEGG_LABGROWN"
			elif ext_code in UNSURE:
				source = "UNSURE"
			else:
				source = "WILD_CAUGHT"
		else:
			source = None
		ext_code_list.append(ext_code)
		source_list.append(source)
		specimen_list.append(specimen)

	return specimen_list,ext_code_list,source_list


def get_source(df):
	species
	for i in range(df.shape[0]):

if __name__ == '__main__':
	datasheet_dir='/home/adam/Downloads/JHU Image Datasheet.xlsx'
	df = pd.read_excel(datasheet_dir,'Mosquito Table')

	taxa_list, idtype_list = get_species(df)
	specimen_list,ext_code_list,source_list = get_source_lists(df)
	write_df = pd.DataFrame(list(zip(specimen_list,taxa_list,idtype_list,ext_code_list,source_list)), columns =['Specimen', 'Taxon', 'ID-Type','ExternalCode', 'Source']) 

	write_df.to_csv(os.path.join(os.getcwd(),'specimen_source.csv'),index = False, header=True)

