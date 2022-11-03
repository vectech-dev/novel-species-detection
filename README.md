# novel-species-detection
"Convolutional neural networks that identify mosquito species and know when they don't know", _in draft_
Algorithm structure can be viewed in the paper.

# Requirements and Installation
This code base will function as expected Ubuntu 20.04/18.04 and is run with Python3, occassionally with Jupyter notebooks. Training should be done on a computer with appropriate hardware (modern GPUs with >8GB memory). [Cuda support](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) is necessary for accessing such hardware for training with this repository. Creation of a virtual envrionment for this work is recommended. The dependencies and versions are listed in the environment.yml file. If you need have not installed conda, do so following [these guidelines](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html). Create a virtual environment with conda: ```conda env create -f environment.yml -n <env_name>```. Use ```conda activate <env_name>``` to activate the envrionment and begin running experiments. For more instruction on conda environments, please follow the details listed [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file). Additionally, for Tiers II and III, install using pip any outstanding requirements using the requirements.txt file: ```pip install -i requirements.txt```. Supposing Cuda is already installed and the GPU drivers for your GPUs are appropriately installed, setup and installation may take up to 2 hours depending on your internet connection. Download the [images](https://novel-species-detection-paper.s3.us-east-2.amazonaws.com/ImageBase.zip) used in the paper, extract the zip file to your /opt/ directory (another directory may be used, but this requires changing the files paths in the datasplit csv files).  Model files as generated by the config files in this repo are available for download [here](https://novel-species-detection-paper.s3.us-east-2.amazonaws.com/CNN_model_files.zip).

# Details
## Folders and Files
* *configs/* in here you will find the configuration files for Xception closed and Xception open. When a classification notebook is run, the file "config/config.py" will be duplicated and placed in a folder named after the experiment name in the config file, and placed in the folder "configs/old_configs/". So simply modify the config file to change the parameters of the experiment. Old configs (in particular those under paper_redo and bigset) can be used to replicate the results in the paper. 
* *data/* in here you will find the datasplits for Tier I Xception and the 39 species classification (referred to as bigset), and the notebooks for generating them. You will also find a smaller test datasplit for verifying the funtion of the repo more rapidly, and pad.jpg, which is used to help make the images square prior to downsampling.
* *model_weights/* not in the github, but will be generated as a location to store the model weights from tier I
* *models/* location for scripts to be imported which will download the pretrained xception network.
* *modules/* loss functions optimizers etc 
* *utils/* misc functions called for training testing evaluation etc 
* *notebooks/* notebooks for training Tier I components (and producing features and outputs for other tiers), 39 species classification, and figure generation. Everytime you run an experiment from the classification-explore*.ipynb, it reads the parameters from config/config.py, unless otherwise specified (by pointing to a config file located elsewhere, such as the "configs/old\_configs"). Then it copies the file of path config/config.py into 'config/old\_configs/{}.py'.format(config.exp_name)
* *subm/* results for each photo are submitted here with probabilities.
* *tierI_output/* features and probabilties for each photo are submitted here for the entire dataset.

* *evaluations.py* - helps generate submission files, feature files and probability files
* *test.py* -for testing just CNN classification independent of other elements

## Tier I components
* Before training any Tier I component, double check that the configuration file is set to desired parameters 
* All Tier I components will report outputs relevant to Tier II or Tier III in directory 'tierI-output/'
* Tier 1 [Closed](https://novel-species-detection-paper.s3.us-east-2.amazonaws.com/T1_closed_FeaturesProbabilities.zip) and [Open](https://novel-species-detection-paper.s3.us-east-2.amazonaws.com/T1_open_FeaturesProbabilities.zip) features and probabilities  as generated in the paper are available for download.
#### Xception
* To train, test, or output features, follow *'notebooks/classification-explore.ipynb'*. Training on a RTX 2070 Super GPU takes approximately two hours, but is estimated in the notebook with real-time updates.   
* To toggle between open and closed sets, and folds, change configuration file (*config/config.py*) 
* *'notebooks/classification-explore-bigset.ipynb'* will train the 39 species classification. The results of this experiment as produced in the paper are available [here](https://novel-species-detection-paper.s3.us-east-2.amazonaws.com/closed_39Species.zip).

## Tier II Components
* see ReadMe.md in the *unknown/* directory 

## Tier III Components
* see ReadMe.md in the *unknown/* directory

## Results Analysis
All relevant scripts are located in the *results_processing/* folder. See the additional readme in that folder. Processing results over the folds indicated in the paper can be done as follows: 
* For averaging confusion matrices for just classification without unknown detection, go to the notebooks/figure_generation.ipynb.
* For cascading novelty detection with classification, use: 1. cascade_novelty_and_classify.py, 2. prep_cascaded_test_sheets.ipynb, 3. avg_cascades.ipynb
* For condensing results in preparation for McNemar's test, use comparison.ipynb
Each script or function within these notebooks should complete within about one minute. The expected output for processing the results of the paper is as reported in the abstract:  ```Closed-set classification of 16 known species achieved 97.04±0.87% accuracy independently, and 89.07±5.58% when cascaded with novelty detection. Closed-set classification of 39 species produces a macro F1-score of 86.07±1.81%. ```. These results are the expected output from processing the results from the model files or the given outputs of the model files (eg. features.csv), or processing the results from outputs of Tier III as given. However, if the system is retrained, variability is expected given the stochastic nature of training neural networks. [This zip file](https://novel-species-detection-paper.s3.us-east-2.amazonaws.com/misc_results_files.zip) contains the outputs of these results processing along with the relevent outputs from Tiers II and III as as produced in the paper. 

# Copyright information 
Shield: [![CC BY NC 4.0][cc-by-nc-shield]][cc-by-nc]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by-nc].

[![CC BY NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: http://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://i.creativecommons.org/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY%20NC%204.0-lightgrey.svg
