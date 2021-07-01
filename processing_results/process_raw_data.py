import os, sys
import pandas as pd
import numpy as np
import math


SETS = ['A', 'B', 'C', 'D']
SETKEY = {'A':'Known', 'B':'Unknown', 'C':'Unknown', 'D':'Unknown', 'Unknown':'Unknown', 'Known':'Known'}
TRUTHKEY = {'Known':False,'Unknown':True}
# where Set A = K, Set B = U1, Set C = U2, Set D = N, per the paper nomenclature.


class Image():
    def __init__(self,fname,pred,truth):
        self.get_image_desc(fname)
        self.pred_unknown = bool(pred)
        self.set_truth(bool(truth))
        return

    def get_image_desc(self,fname='/opt/ImageBase/AWS_sync/cropped/aedes/aegypti/JHU-000618_01m.jpg'):
        remove = '/opt/ImageBase/AWS_sync/cropped/'
        parts = fname.replace(remove,'').split('/')
        self.genus = parts[0]
        if len(parts)==3:
            self.species = parts[1]
        elif len(parts)==4:
            self.species = parts[1].replace('_',' ')
        self.fname = parts[-1]
        self.specimen = self.fname.split('_')[0]
        self.taxon = self.genus + ' ' + self.species
        # return taxa, genus,species,specimen,short_fn
        return

    def set_truth(self, is_unknown):
        #recives the ground truth of that specimen for this seed and returns whether or not the prediction was correct
        if type(is_unknown) is not bool:
            raise SystemError('is_unknown must be bool type')
        self.truth_unknown = is_unknown
        self.is_correct = self.truth_unknown == self.pred_unknown
        return 
 

 
class Specimens():
    def __init__(self,image,ground_truth_bool):
        self.name = image.specimen
        self.ground_truth_bool = ground_truth_bool
        self.images = [image]
        self.image_cnt = 1
        return
    
    def add_image(self,image):
        self.images.append(image)
        self.image_cnt+=1
        return
    def process_results(self):
        self.correct_image_tally = 0
        for image in self.images:
            self.correct_image_tally += image.is_correct
        self.accuracy = self.correct_image_tally/self.image_cnt

        #confusion matrix layout: [[True Unknown,False Known],[False Unknown,True Known]] or ((TP,FN),(FP,TN))
        self.confusion_matrix_raw = np.array([[0,0],[0,0]],dtype=np.float64)
        if self.ground_truth_bool:
            self.confusion_matrix_raw[0][0]=self.correct_image_tally
            self.confusion_matrix_raw[0][1]=self.image_cnt - self.correct_image_tally
        else:
            self.confusion_matrix_raw[1][1]=self.correct_image_tally
            self.confusion_matrix_raw[1][0]=self.image_cnt - self.correct_image_tally

        self.confusion_matrix_normalized=self.confusion_matrix_raw/self.image_cnt

        return


class Taxon():
    def __init__(self,image,ground_truth_bool):
        self.name = image.taxon
        self.ground_truth_bool = ground_truth_bool 
        self.images = [image]
        self.image_cnt = 1
        self.specimens = {image.specimen:Specimens(image,self.ground_truth_bool)}
        self.specimen_cnt = 1
        return
    def add_image(self,image):
        self.images.append(image)
        self.image_cnt+=1
        if image.specimen not in self.specimens:
            self.specimens[image.specimen]=Specimens(image,self.ground_truth_bool)
            self.specimen_cnt+=1
        else:
            self.specimens[image.specimen].add_image(image)
        return

    def specimen_stats(self):

        # Taxon-level specimen normalized confusion matrix
        self.specimen_normalized_confusion = np.array([[0,0],[0,0]],dtype=np.float64)
        for specimen in self.specimens:
            self.specimens[specimen].process_results()
            self.specimen_normalized_confusion+= self.specimens[specimen].confusion_matrix_normalized
        self.specimen_normalized_confusion = self.specimen_normalized_confusion/self.specimen_cnt

        # Taxon-level specimen StdDev of normalized confusion matrix 
        self.specimen_norm_confusion_stdev = np.array([[0,0],[0,0]],dtype=np.float64)
        for specimen in self.specimens:
            self.specimen_norm_confusion_stdev += np.square(self.specimens[specimen].confusion_matrix_normalized - self.specimen_normalized_confusion)
        self.specimen_norm_confusion_stdev = self.specimen_norm_confusion_stdev/(self.specimen_cnt-1)
        self.specimen_norm_confusion_stdev = np.sqrt(self.specimen_norm_confusion_stdev)
        return

    def process_results(self):
        self.correct_image_tally = 0
        for image in self.images:
            self.correct_image_tally += image.is_correct
        self.accuracy = self.correct_image_tally/self.image_cnt

        #confusion matrix layout: [[True Unknown,False Known],[False Unknown,True Known]] or ((TP,FN),(FP,TN))
        self.confusion_matrix_raw = np.array([[0,0],[0,0]],dtype=np.float64)
        if self.ground_truth_bool:
            self.confusion_matrix_raw[0][0]=self.correct_image_tally
            self.confusion_matrix_raw[0][1]=self.image_cnt - self.correct_image_tally
        else:
            self.confusion_matrix_raw[1][1]=self.correct_image_tally
            self.confusion_matrix_raw[1][0]=self.image_cnt - self.correct_image_tally

        self.confusion_matrix_normalized=self.confusion_matrix_raw/self.image_cnt
        
        self.specimen_stats()

        return


class SpeciesGroup():
    def __init__(self, name,data,pred_column=2):
        self.pred_column = pred_column
        self.name = name
        self.images = []
        self.image_cnt = 0
        self.specimens = {}
        self.specimen_cnt = 0
        self.taxa = {}
        self.taxa_cnt = 0
        self.ground_truth = SETKEY[self.name]
        self.ground_truth_bool = TRUTHKEY[self.ground_truth]
        if self.name in SETS:
            self.get_set_data(data)
        else:
            self.get_U_or_K_data(data)
        self.set_unique()
        self.process_group_results()

    def get_set_data(self, data_frame):
        for i in range(data_frame.shape[0]):
            fname = data_frame.iloc[i,0]
            if data_frame.iloc[i,1] != self.ground_truth_bool:
                sys.exit('Error:'+str(data_frame.iloc[i,1])+' is listed truth for '+ str(i)+ ' of '+ str(data_frame.iloc[i,0]) + ' but ground_truth_bool is ' + str(self.ground_truth_bool))
            image = Image(data_frame.iloc[i,0],data_frame.iloc[i,self.pred_column],self.ground_truth_bool)
            self.images.append(image)
            self.image_cnt += 1

            if image.specimen not in self.specimens:
                self.specimens[image.specimen]=Specimens(image,self.ground_truth_bool)
                self.specimen_cnt += 1
                if image.taxon not in self.taxa:
                    self.taxa[image.taxon] = Taxon(image,self.ground_truth_bool)
                    self.taxa_cnt += 1
                else:
                    self.taxa[image.taxon].add_image(image)
            else:
                #print(self.specimens)
                self.specimens[image.specimen].add_image(image)
                self.taxa[image.taxon].add_image(image)
        return

    def get_U_or_K_data(self, sp_group_list):
        # getting known or unknown data
        for sp_group in sp_group_list:
            self.images.extend(sp_group.images)
            self.image_cnt += sp_group.image_cnt
            self.specimens.update(sp_group.specimens)
            self.specimen_cnt += sp_group.specimen_cnt
            self.taxa.update(sp_group.taxa)
            self.taxa_cnt += sp_group.taxa_cnt

    def set_unique(self):
        # using set() + len() to check all unique list elements 
        flag = len(set(self.taxa)) == self.taxa_cnt  
        if not flag: 
            # print (self.name+ " contains "+str(self.taxa_cnt)+ "unique species") 
        # else :  
            print (self.name+ " does not contain "+str(self.taxa_cnt)+ " unique species. Actually " + str(len(set(species))) + ' unique species.')
        return

    def specimen_stats(self):
        # Taxon-level specimen normalized confusion matrix
        self.specimen_normalized_confusion = np.array([[0.,0.],[0.,0.]],dtype=np.float64)
        for specimen in self.specimens:
            self.specimens[specimen].process_results()
            self.specimen_normalized_confusion= np.add(self.specimen_normalized_confusion,self.specimens[specimen].confusion_matrix_normalized)
        self.specimen_normalized_confusion = self.specimen_normalized_confusion/self.specimen_cnt

        # Taxon-level specimen StdDev of normalized confusion matrix 
        self.specimen_norm_confusion_stdev = np.array([[0,0],[0,0]],dtype=np.float64)
        for specimen in self.specimens:
            self.specimen_norm_confusion_stdev += np.square(self.specimens[specimen].confusion_matrix_normalized - self.specimen_normalized_confusion)
        self.specimen_norm_confusion_stdev = self.specimen_norm_confusion_stdev/(self.specimen_cnt-1)
        self.specimen_norm_confusion_stdev = np.sqrt(self.specimen_norm_confusion_stdev)
        return

    def taxon_stats(self):
        # SpeciesGroup-level taxon normalized confusion matrix
        self.taxon_normalized_confusion = np.array([[0,0],[0,0]],dtype=np.float64)
        self.taxon_norm_accuracy = 0
        for taxon in self.taxa:
            self.taxa[taxon].process_results()
            self.taxon_normalized_confusion += self.taxa[taxon].confusion_matrix_normalized
            self.taxon_norm_accuracy += self.taxa[taxon].accuracy
        self.taxon_normalized_confusion = self.taxon_normalized_confusion/self.taxa_cnt
        self.taxon_norm_accuracy = self.taxon_norm_accuracy/self.taxa_cnt

        # SpeciesGroup-level taxon StdDev of normalized confusion matrix 
        self.taxon_norm_confusion_stdev = np.array([[0,0],[0,0]],dtype=np.float64)
        for taxon in self.taxa:
            self.taxon_norm_confusion_stdev += np.square(self.taxa[taxon].confusion_matrix_normalized - self.taxon_normalized_confusion)
        self.taxon_norm_confusion_stdev = self.taxon_norm_confusion_stdev/(self.taxa_cnt-1)
        self.taxon_norm_confusion_stdev = np.sqrt(self.taxon_norm_confusion_stdev)
        return

    def process_group_results(self):
        self.correct_image_tally = 0
        for image in self.images:
            self.correct_image_tally += image.is_correct
        self.accuracy = self.correct_image_tally/self.image_cnt

        #confusion matrix layout: [[True Unknown,False Known],[False Unknown,True Known]] or ((TP,FN),(FP,TN))
        self.confusion_matrix_raw = np.array([[0,0],[0,0]],dtype=np.float64)
        if self.ground_truth_bool:
            self.confusion_matrix_raw[0][0]=self.correct_image_tally
            self.confusion_matrix_raw[0][1]=self.image_cnt - self.correct_image_tally
        else:
            self.confusion_matrix_raw[1][1]=self.correct_image_tally
            self.confusion_matrix_raw[1][0]=self.image_cnt - self.correct_image_tally

        self.confusion_matrix_normalized=self.confusion_matrix_raw/self.image_cnt
       
        self.specimen_stats()
        self.taxon_stats()

        return


class Seed():
    def __init__(self,csv_fname,pred_column):
        self.fname = csv_fname 
        self.name = csv_fname.split('_')[-1].split('.')[0]
        self.fold = csv_fname.split('_')[-2]
        self.data = {}
        self.speciesGroupLists = {}
        self.seed_df = pd.read_csv(csv_fname)
        for group in SETS:
            print(csv_fname,group)
            self.data[group] = SpeciesGroup(group, self.seed_df[self.seed_df["split"]==group],pred_column)
            self.speciesGroupLists[group]=list(self.data[group].taxa.keys())
            # print(str(self.data[group].confusion_matrix_raw))
            # print(str(self.data[group].confusion_matrix_normalized))
            # print(str(self.data[group].taxon_normalized_confusion))
            # print(str(self.data[group].specimen_normalized_confusion))
            # print(list(self.data[group].taxa.keys()))
        self.process_results()
        return

    

    def process_results(self):
        self.known_samples = 0
        self.unknown_samples = 0
        self.known_species = 0
        self.unknown_species = 0

        self.confusion_matrix_raw = np.array([[0,0],[0,0]],dtype=np.float64)
        self.confusion_matrix_perc = np.array([[0,0],[0,0]],dtype=np.float64)
        self.taxon_norm_confusion = np.array([[0,0],[0,0]],dtype=np.float64)
        self.accuracy = 0
        self.taxon_norm_accuracy = 0

        for group in SETS:
            if self.data[group].ground_truth_bool:
                self.unknown_samples += self.data[group].image_cnt
                self.unknown_species += self.data[group].taxa_cnt
            else:
                self.known_samples += self.data[group].image_cnt
                self.known_species += self.data[group].taxa_cnt
            self.confusion_matrix_raw += self.data[group].confusion_matrix_raw
            self.taxon_norm_confusion += self.data[group].taxon_normalized_confusion*self.data[group].taxa_cnt
            self.accuracy += self.data[group].accuracy*self.data[group].image_cnt
            for taxon in self.data[group].taxa:
                self.taxon_norm_accuracy += self.data[group].taxa[taxon].accuracy

        self.confusion_matrix_perc[0][0] = self.confusion_matrix_raw[0][0]/self.unknown_samples
        self.confusion_matrix_perc[0][1] = self.confusion_matrix_raw[0][1]/self.unknown_samples
        self.confusion_matrix_perc[1][0] = self.confusion_matrix_raw[1][0]/self.known_samples
        self.confusion_matrix_perc[1][1] = self.confusion_matrix_raw[1][1]/self.known_samples

        self.taxon_norm_confusion[0][0] = self.taxon_norm_confusion[0][0]/self.unknown_species
        self.taxon_norm_confusion[0][1] = self.taxon_norm_confusion[0][1]/self.unknown_species
        self.taxon_norm_confusion[1][0] = self.taxon_norm_confusion[1][0]/self.known_species
        self.taxon_norm_confusion[1][1] = self.taxon_norm_confusion[1][1]/self.known_species
        self.accuracy= self.accuracy/(self.unknown_samples+self.known_samples)
        self.taxon_norm_accuracy = self.taxon_norm_accuracy/(self.known_species+self.unknown_species)

        # sensitivity = recall = tp / t = tp / (tp + fn)
        self.sensativity = self.confusion_matrix_raw[0][0]/(self.confusion_matrix_raw[0][0]+self.confusion_matrix_raw[0][1])
        self.taxon_norm_sensativity = self.taxon_norm_confusion[0][0]/(self.taxon_norm_confusion[0][0]+self.taxon_norm_confusion[0][1])
        # specificity = tn / n = tn / (tn + fp)
        self.specificity = self.confusion_matrix_raw[1][1]/(self.confusion_matrix_raw[1][0]+self.confusion_matrix_raw[1][1])
        self.taxon_norm_specificity = self.taxon_norm_confusion[1][1]/(self.taxon_norm_confusion[1][0]+self.taxon_norm_confusion[1][1])
        #precision = tp / p = tp / (tp + fp)
        self.precision = self.confusion_matrix_raw[0][0]/(self.confusion_matrix_raw[0][0]+self.confusion_matrix_raw[1][0])
        self.taxon_norm_precision =  self.taxon_norm_confusion[0][0]/(self.taxon_norm_confusion[0][0]+self.taxon_norm_confusion[1][0])

        self.macrof1 = 2*self.taxon_norm_precision*self.taxon_norm_sensativity/(self.taxon_norm_precision+self.taxon_norm_sensativity)

class AllData():
    def __init__(self,data_dir,pred_column):
        self.data_dir = data_dir
        self.get_data(pred_column)
        self.avg_seeds()
        # print(self.seeds[0].taxon_norm_confusion)
        # print(self.seeds[0].taxon_norm_sensativity)
        # print(self.seeds[0].taxon_norm_specificity)
        # print(self.seeds[0].taxon_norm_precision)

        # print(self.seeds[0].confusion_matrix_perc)
        # # print(self.seeds[0].confusion_matrix_raw)
        # print(self.seeds[0].sensativity)
        # print(self.seeds[0].specificity)
        # print(self.seeds[0].precision)


    def get_data(self,pred_column):
        self.seeds = []
        print(self.data_dir)
        for root,dirs,files in os.walk(self.data_dir):
            # print(root, files)
            for f in files:
                if f.split('.')[-1]=='csv':
                    seed = Seed(os.path.join(root,f),pred_column)
                    self.seeds.append(seed)
        self.seed_cnt = len(self.seeds)
        print('Seed count: ',self.seed_cnt)

    def get_lists(self):
        self.all_taxa = []
        self.known_taxa = []
        for seed in self.seeds:
            for group in SETS:
                print(seed.speciesGroupLists[group])
                for sp in seed.speciesGroupLists[group]:
                    if TRUTHKEY[SETKEY[group]]:
                        if sp not in self.all_taxa:
                            self.all_taxa.append(sp)
                    else:
                        if sp not in self.all_taxa:
                            self.all_taxa.append(sp)
                        if sp not in self.known_taxa:
                            self.known_taxa.append(sp)

    #TODO: compile results
    def avg_seeds(self):
        self.confusion_matrix =  np.array([[0,0],[0,0]],dtype=np.float64)
        self.confusion_matrix_stdev =  np.array([[0,0],[0,0]],dtype=np.float64)
        self.taxon_norm_confusion =  np.array([[0,0],[0,0]],dtype=np.float64)
        self.taxon_norm_confusion_stdev =  np.array([[0,0],[0,0]],dtype=np.float64)
        self.precision = 0
        self.precision_stdev = 0
        self.taxon_norm_precision = 0
        self.taxon_norm_precision_stdev = 0
        self.taxon_norm_accuracy = 0
        self.accuracy = 0 
        self.taxon_norm_accuracy_stdev = 0
        self.accuracy_stdev = 0 
        self.macrof1_avg=0
        self.macrof1_stdev=0

        self.min_known_taxa = 0
        self.max_known_taxa = 0
        self.avg_known_taxa = 0
        self.min_unknown_taxa = 0
        self.max_unknown_taxa = 0
        self.avg_unknown_taxa = 0
        self.min_known_samples = 0
        self.max_known_samples = 0
        self.avg_known_samples = 0
        self.min_unknown_samples = 0
        self.max_unknown_samples = 0
        self.avg_unknown_samples = 0




        for seed in self.seeds:
            self.confusion_matrix += seed.confusion_matrix_perc
            self.taxon_norm_confusion += seed.taxon_norm_confusion
            self.precision += seed.precision
            self.taxon_norm_precision += seed.taxon_norm_precision
            self.taxon_norm_accuracy += seed.taxon_norm_accuracy
            self.accuracy += seed.accuracy
            self.macrof1_avg += seed.macrof1

        self.confusion_matrix = self.confusion_matrix/self.seed_cnt
        self.taxon_norm_confusion = self.taxon_norm_confusion/self.seed_cnt
        self.precision = self.precision/self.seed_cnt
        self.taxon_norm_precision = self.taxon_norm_precision/self.seed_cnt
        self.taxon_norm_accuracy = self.taxon_norm_accuracy/self.seed_cnt
        self.accuracy = self.accuracy/self.seed_cnt
        self.macrof1_avg =self.macrof1_avg/self.seed_cnt

        for seed in self.seeds:
            self.confusion_matrix_stdev += np.square(seed.confusion_matrix_perc - self.confusion_matrix)
            self.taxon_norm_confusion_stdev += np.square(seed.taxon_norm_confusion - self.taxon_norm_confusion)
            self.precision_stdev += np.square(seed.precision - self.precision)
            self.taxon_norm_precision_stdev += np.square(seed.taxon_norm_precision - self.taxon_norm_precision)
            self.taxon_norm_accuracy_stdev += np.square(seed.taxon_norm_accuracy - self.taxon_norm_accuracy)
            self.accuracy_stdev += np.square(seed.accuracy - self.accuracy)
            self.macrof1_stdev += np.square(seed.macrof1 - self.macrof1_avg)
        self.confusion_matrix_stdev = self.confusion_matrix_stdev/(self.seed_cnt-1)
        self.taxon_norm_confusion_stdev = self.taxon_norm_confusion_stdev/(self.seed_cnt-1)
        self.precision_stdev = self.precision_stdev/(self.seed_cnt-1)
        self.taxon_norm_precision_stdev = self.taxon_norm_precision_stdev /(self.seed_cnt-1)
        self.taxon_norm_accuracy_stdev = self.taxon_norm_accuracy_stdev  /(self.seed_cnt-1)
        self.accuracy_stdev = self.accuracy_stdev /(self.seed_cnt-1)
        self.macrof1_stdev = self.macrof1_stdev/ (self.seed_cnt-1) 

        self.confusion_matrix_stdev = np.sqrt(self.confusion_matrix_stdev)
        self.taxon_norm_confusion_stdev = np.sqrt(self.taxon_norm_confusion_stdev)
        self.precision_stdev = np.sqrt(self.precision_stdev)
        self.taxon_norm_precision_stdev = np.sqrt(self.taxon_norm_precision_stdev)
        self.taxon_norm_accuracy_stdev = np.sqrt(self.taxon_norm_accuracy_stdev)
        self.accuracy_stdev = np.sqrt(self.accuracy_stdev)
        self.macrof1_stdev = np.sqrt(self.macrof1_stdev)

        for seed in self.seeds:
            # now describe how the sample cnts vary
            if self.min_unknown_taxa == 0:
                self.min_known_taxa = seed.known_species
                self.max_known_taxa = seed.known_species
                self.avg_known_taxa = seed.known_species
                self.min_unknown_taxa = seed.unknown_species
                self.max_unknown_taxa = seed.unknown_species
                self.avg_unknown_taxa = seed.unknown_species
                self.min_known_samples = seed.known_samples
                self.max_known_samples = seed.known_samples
                self.avg_known_samples = seed.known_samples
                self.min_unknown_samples = seed.unknown_samples
                self.max_unknown_samples = seed.unknown_samples
                self.avg_unknown_samples = seed.unknown_samples
            else:
                self.min_known_taxa = min(seed.known_species,self.min_known_taxa)
                self.max_known_taxa = max(seed.known_species,self.max_known_taxa) 
                self.avg_known_taxa += seed.known_species
                self.min_unknown_taxa = min(seed.unknown_species,self.min_unknown_taxa )
                self.max_unknown_taxa = max(seed.unknown_species,self.max_unknown_taxa )
                self.avg_unknown_taxa += seed.unknown_species
                self.min_known_samples = min(seed.known_samples,self.min_known_samples)
                self.max_known_samples = max(seed.known_samples,self.max_known_samples)
                self.avg_known_samples += seed.known_samples
                self.min_unknown_samples = min(seed.unknown_samples,self.min_unknown_samples)
                self.max_unknown_samples = max(seed.unknown_samples,self.max_unknown_samples)
                self.avg_unknown_samples += seed.unknown_samples
        self.avg_known_samples = self.avg_known_samples/self.seed_cnt
        self.avg_unknown_samples = self.avg_unknown_samples/self.seed_cnt
        self.avg_known_taxa = self.avg_known_taxa/self.seed_cnt
        self.avg_unknown_taxa = self.avg_unknown_taxa/self.seed_cnt

        self.group_samples={}
        self.group_accuracy={}
        self.group_accuracy_stdev={}
        for seed in self.seeds:
            for group in SETS:
                if group not in self.group_samples:
                    self.group_samples[group]={'min_taxa':seed.data[group].taxa_cnt,'max_taxa':seed.data[group].taxa_cnt,'avg_taxa':seed.data[group].taxa_cnt,
                    'min_samples':seed.data[group].image_cnt,'max_samples':seed.data[group].image_cnt,'avg_samples':seed.data[group].image_cnt}
                    self.group_accuracy[group]={'species':seed.data[group].taxon_norm_accuracy,'samples':seed.data[group].accuracy}
                else:
                    self.group_samples[group]={'min_taxa':min(seed.data[group].taxa_cnt,self.group_samples[group]['min_taxa']),
                    'max_taxa':max(seed.data[group].taxa_cnt,self.group_samples[group]['max_taxa']),
                    'avg_taxa':seed.data[group].taxa_cnt+self.group_samples[group]['avg_taxa'],
                    'min_samples':min(seed.data[group].image_cnt,self.group_samples[group]['min_samples']),
                    'max_samples':max(seed.data[group].image_cnt,self.group_samples[group]['max_samples']),
                    'avg_samples':seed.data[group].image_cnt+self.group_samples[group]['avg_samples']}
                    self.group_accuracy[group]={'species':seed.data[group].taxon_norm_accuracy+self.group_accuracy[group]['species'],
                    'samples':seed.data[group].accuracy+self.group_accuracy[group]['samples']}
        for group in SETS:
            self.group_accuracy[group]={'species':self.group_accuracy[group]['species']/self.seed_cnt,'samples':self.group_accuracy[group]['samples']/self.seed_cnt}
            self.group_samples[group]['avg_taxa']=self.group_samples[group]['avg_taxa']/self.seed_cnt
            self.group_samples[group]['avg_samples']=self.group_samples[group]['avg_samples']/self.seed_cnt
        for seed in self.seeds:
            for group in SETS:
                if group not in self.group_accuracy_stdev:
                    self.group_accuracy_stdev[group]={'species':np.square(seed.data[group].taxon_norm_accuracy-self.group_accuracy[group]['species']),
                    'samples':np.square(seed.data[group].accuracy-self.group_accuracy[group]['samples'])}
                else:
                    self.group_accuracy_stdev[group]['species']+=np.square(seed.data[group].taxon_norm_accuracy-self.group_accuracy[group]['species'])
                    self.group_accuracy_stdev[group]['samples']+=np.square(seed.data[group].accuracy-self.group_accuracy[group]['samples'])
        for group in SETS:
            self.group_accuracy_stdev[group]={'species':np.sqrt(self.group_accuracy_stdev[group]['species']/(self.seed_cnt-1)),
            'samples':np.sqrt(self.group_accuracy_stdev[group]['samples']/(self.seed_cnt-1))}



if __name__=='__main__':


#     all_results= ['run75/full',2],
#     alldata = AllData(all_results[0],all_results[1])
    res_dir='/home/adam/VecTech/repos/novel-species-detection/tierIII_output/run75'
    # all_results= {'full_methods': [res_dir+'/full',2],
    #     'odin': [res_dir+'/odin',2],
    #     'svote': [res_dir+'/svote',3],
    #     't2_rf16': [res_dir+'/t2/16xcept',2],
    #     't2_svm16': [res_dir+'/t2/16xcept',4],
    #     't2_wdnn16': [res_dir+'/t2/16xcept',6],
    #     't2_rf21': [res_dir+'/t2/21xcept',2],
    #     't2_svm21':[res_dir+'/t2/21xcept',4],
    #     't2_wdnn21':[res_dir+'/t2/21xcept',6],
    #     'softmax16':[res_dir+'/extras/closedsoftmax_breakdown',2],
    #     'openset21':[res_dir+'/extras/openset_breakdown',2]}
    all_results= {
        'odin': [res_dir+'/odin',2]}
    

    # alldata = AllData(t2_rf16[0],t2_rf16[1])
    alldata = AllData(all_results['odin'][0],all_results['odin'][1])
    for method in all_results:
        print(res_dir,method)
        alldata = AllData(all_results[method][0],all_results[method][1])
        results=open(res_dir+'/'+method+'.txt','w')
     
        results.writelines(["Confusion matrix and stdev\n"])
        results.writelines([str(alldata.confusion_matrix),'\n'])
        results.writelines([str(alldata.confusion_matrix_stdev),'\n'])
        results.writelines(["Precision: ",str(alldata.precision)," +- ",str(alldata.precision_stdev),'\n','\n'])
        results.writelines(["Confusion matrix and stdev, Taxon normalized",'\n'])
        results.writelines([str(alldata.taxon_norm_confusion),'\n'])
        results.writelines([str(alldata.taxon_norm_confusion_stdev),'\n','\n'])
        results.writelines(["Precision, taxon normalized: ",str(alldata.taxon_norm_precision)," +- ",str(alldata.taxon_norm_precision_stdev),'\n'])
        results.writelines(["Macro F1-score: ",str(alldata.macrof1_avg)," +- ", str(alldata.macrof1_stdev),'\n'])
        results.writelines(["Accuracy: ",str(alldata.accuracy)," +- ",str(alldata.accuracy_stdev),'\n'])
        results.writelines(["Accuracy, taxon normalized: ",str(alldata.taxon_norm_accuracy)," +- ",str(alldata.taxon_norm_accuracy_stdev),'\n','\n'])
        results.writelines(["\nSample description: [min avg max] \n"])
        results.writelines(["Known samples: ",str(alldata.min_known_samples),' ',str(alldata.avg_known_samples),' ',str(alldata.max_known_samples),'\n'])
        results.writelines(["Unknown samples: ",str(alldata.min_unknown_samples),' ',str(alldata.avg_unknown_samples),' ',str(alldata.max_unknown_samples),'\n'])
        results.writelines(["Known taxa: ",str(alldata.min_known_taxa),' ',str(alldata.avg_known_taxa),' ',str(alldata.max_known_taxa),'\n'])
        results.writelines(["Unknown taxa: ",str(alldata.min_unknown_taxa),' ',str(alldata.avg_unknown_taxa),' ',str(alldata.max_unknown_taxa),'\n'])
        results.writelines(["\nGroup Data",'\n'])
        results.writelines(["group samples",'\n'])
        results.writelines([str(alldata.group_samples),'\n'])
        results.writelines(["group accuracy",'\n'])
        results.writelines([str(alldata.group_accuracy),'\n'])
        results.writelines(["group accuracy stdev",'\n'])
        results.writelines([str(alldata.group_accuracy_stdev),'\n'])
        results.close()

    # print("Confusion matrix and stdev")
    # print(alldata.confusion_matrix)
    # print(alldata.confusion_matrix_stdev)
    # print("Precision: ",alldata.precision," +- ",alldata.precision_stdev)
    # print("Confusion matrix and stdev, Taxon normalized")
    # print(alldata.taxon_norm_confusion)
    # print(alldata.taxon_norm_confusion_stdev)
    # print("Precision, taxon normalized: ",alldata.taxon_norm_precision," +- ",alldata.taxon_norm_precision_stdev)
    # print("Accuracy: ",alldata.accuracy," +- ",alldata.accuracy_stdev)
    # print("Accuracy, taxon normalized: ",alldata.taxon_norm_accuracy," +- ",alldata.taxon_norm_accuracy_stdev)
    # print("\nSample description: [min avg max] ")
    # print("Known samples: ",alldata.min_known_samples,alldata.avg_known_samples,alldata.max_known_samples)
    # print("Unknown samples: ",alldata.min_unknown_samples,alldata.avg_unknown_samples,alldata.max_unknown_samples)
    # print("Known taxa: ",alldata.min_known_taxa,alldata.avg_known_taxa,alldata.max_known_taxa)
    # print("Unknown taxa: ",alldata.min_unknown_taxa,alldata.avg_unknown_taxa,alldata.max_unknown_taxa)
    # print("\nGroup Data")
    # print("group samples")
    # print(alldata.group_samples)
    # print("group accuracy")
    # print(alldata.group_accuracy)
    # print("group accuracy stdev")
    # print(alldata.group_accuracy_stdev)

