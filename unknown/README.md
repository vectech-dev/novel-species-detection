# Unknown Species Detection (Tiers 2 and 3)

## Getting Tier 1 Data
After running the tier1 data, you should have 2 config files per fold, one for the open set and one for the closed set. To run tier 2 or tier 3 methods, you will need to first set two env variable (CLOSED_CONFIG and OPEN_CONFIG) which correspond to the path to each of these config files. The path can be either an absolute path, or a relative path with respect to the root of this project. 

For example, if you're on a unix system and you have already re-run "paper_redo", you could use the following commands to set up for fold 1.

```
export CLOSED_CONFIG="configs/old_configs/paper_redo/closed/fold1/config.py"
export OPEN_CONFIG="configs/old_configs/paper_redo/open/fold1/config.py"
```

There is an example of automating all folds to run in ```unknown/run_all_folds.sh```

## Tier 2
You can use ```python tier2_main.py``` to run grid search and parameter optimization on individual tier 2 methods. Use the ```--help``` command for more details. Since hyperparameters used in the paper are already provided and used by default (see below for more), this file is not necessary to obtain the final unknown classifications.

### Hyperparameter
Hyperparameters are stored in ```unknown/data/tier2/params/<alg>``` where alg is the tier 2 algorithm (rfor, svm, or wnn). The hyperparameters used in the paper are provided in the ```unknown/data/tier2/params/<alg>/finale``` folder. For a particular algorithm, the finale params will be used if the hyperparameter folder is otherwise empty. If not, the closed set will use any json file with "16" in the filename and the open set will use any json file with "21" in the filename. If there are multiple such files, it will error.

### Grid Search
If you run a grid search for a particular algorithm, it will automatically save the best parameters to the appropriate parameter folder. By default, it will overwrite previous parameters but it won't overwrite the finale params, as they are in a seperate folder.

## Tier 3
You can use ```python tier3_main.py``` to run grid search and parameter optimization on the tier 3 method and get final unknown classification results. Use the ```--help``` command for more details. It is best to run folds of the same experiment using the same "run" input. If ```--test``` is active, the results will appear in the ```tierIII_output/<run>``` directory.

Remember, there is an example of automating all folds to run in ```unknown/run_all_folds.sh```

### Caching
The tier3 utilizes a caching approach to storing tier 2 results. When you start a tier3 run, it will check the cache and, if it can't find anything, create tier 2 results. 

### Hyperparameters
These work the same as in tier2, but they are stored in  ```unknown/data/tier3/params/garb```. There is no concept of open vs closed sets in tier 3, so it will use params from any json file that it can find in the params folder.

### Grid Search
This also works the same as in tier2. Grid search uses a different cache, because the train set is split into a train and valid set for validation.

## Extras
There is a ```get_extras.py``` script which can be used for getting some extra information that was used in the paper. Use ```--help``` for more info. Running it requires that you run all folds and seeds for the given run. It will then create add tier2 results, raw svote results, and naive methods for unknown detection in your ```tierIII_output/<run>``` directory.

## Compatability
If you're on a Windows system, it is recommended to use this repo with WSL. Otherwise, you may need to change some unix paths to get things working.

