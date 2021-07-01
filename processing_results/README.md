# algorithm-paper-results

## File and folder descriptions
* *cascade_novelty_and_classify.py* this take novelty result, and if known, will get the closed set classification result for that sample, for each fold. This must be run prior to  *prep_cascaded_test_sheets.ipynb*  and *avg_cascades.ipynb*
* *avg_cascades.ipynb*: this notebook intakes the cascaded confusion matrices for each fold, averages them, calculates some basic metrics on cascaded classification  (where unknown is an indepedent category) and then produces a cnfusion matrix figure for them. 
* *comparison.ipynb*: this notebook is used to create a spreadsheet from which to calculate the McNemars Chi Squared values and associated p-values for comparing alternative methods to the full methods. The calculation of chi-squared values is done in a .xlsx file with functions, after using this notebook to create the .xlsx file in the right format. 
* *Datafolds condensing.ipynb*: This notebook is used to condense the datasplit files for each fold and iteration to a single file to create the 'Supplimental Information - Datasplits.csv' file.
* *prep_cascaded_test_sheets.ipynb*: this notebook will give you confusion matrices for each fold cascaded with novelty detection. This is only for each fold, not averaged.
* *Samples.ipynb*: This notebook will get you condensed sample data as listed in the methods section of the paper. 
* *parse_data_source.py*: this script gets us the origin of the specimens in the database using the JHU Image Datasheet.xlsx.
* *process_raw_data.py*: this script processes all the unknown results for our methods and any alternative methods. Format is determined in the main function. Results must come in a csv with a column of unknown results per photo, where 0=known and 1=unknown. 
* *utils/*: This folder contains some utility files for correcting the name of a species if raw entered data by human is misspelled. 

## processing results
### Notes on getting unknown detection metrics for any method
 Ideally, outputs have been saved to tierIII_output/{}/{}/\*\_fold{1-5}\_seed{0-4}.csv".format(run_name, method_name), where method_name={full, t2_16xcept, svote, extras/openset, extras/closedsoftmax}. For each results csv, a row represents a photo, with at least a column for ID and a column for  the prediction (as 0=known or 1=unknown). Each method should have its own csv, with the exception of the tier II methods, which should be groupped with the tier I method to which they correspond. In *process_raw_data.py*, in the main function, for the variable all_results which defines the location of each method's results, verify that the number next to each method in the dict variable all_results corresponds to the column in that resulst file where the predictions are located. Also verify that the correct results file is called. This script will output text files with the results for each method. These metrics are only unknown detection metrics. For cascading any method with classification results, seperate steps are required. 
### Notes on classification cascading and averaging
* For averaging confusion matrices for just classification without unknown detection, go to the *notebooks/figure_generation.ipynb*. 
* For cascading novelty detection with classification, use: 1. *cascade_novelty_and_classify.py*, 2. *prep_cascaded_test_sheets.ipynb*, 3. *avg_cascades.ipynb*
* For condensing results in preparation for McNemar's test, use *comparison.ipynb*
