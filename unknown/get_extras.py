import os
import sys
sys.path.append(
  os.path.dirname(
  os.path.dirname(
  os.path.abspath(__file__))))
import argparse
from unknown.config.trials import trials
from unknown.utils.extras import do_closedsoftmax_extras, do_svote_extras, do_openset_extras, do_t2_16xcept_extras
from unknown.utils.paths import get_t3_config_path

def main(run, trial, folds, seeds):
  print("------------- Closed Softmax -------------\n")
  do_closedsoftmax_extras(run, trial=trial, folds=folds, seeds=seeds)
  print("------------- Open Grouping -------------\n")
  do_openset_extras(run, trial=trial, folds=folds, seeds=seeds)
  print("------------- T2 16Xcept -------------\n")
  do_t2_16xcept_extras(run, trial=trial, folds=folds, seeds=seeds)
  print("------------- Svote -------------\n")
  do_svote_extras(run, trial=trial, folds=folds, seeds=seeds)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Get extra information after running all seeds and folds')
  parser.add_argument('run', type=str, help='Which run to use')
  parser.add_argument('--folds', '-f', type=int, nargs='+', default=range(1,5+1), help='Which folds to use')
  parser.add_argument('--seeds', '-ns', type=int, nargs='+', default=range(5), help='Which species_seeds to use')
  parser.add_argument('--trial', '-t', type=str, default="T1", help='Which data to use (Default: T1 as used in paper)')

  args = parser.parse_args()
  
  # validate
  try:
    for fold in args.folds:
      closed_cfg_dir = get_t3_config_path(args.run, True, fold)
      if not os.path.exists(closed_cfg_dir):
        raise ValueError(f"Could not find results for fold{fold}.\nAll folds and species_seeds must be run for this run before running this code.")
      open_cfg_dir = get_t3_config_path(args.run, False, fold)
      if not os.path.exists(closed_cfg_dir):
        raise ValueError(f"Could not find results for fold{fold}.\nAll folds and species_seeds must be run for this run before running this code.")

    # Trial
    trial = trials.get(args.trial, None)
    if trial is None:
      raise ValueError(f"Invalid trial name {args.trial}")
    if any([s >= 5 or s < 0 for s in args.seeds]):
      raise ValueError(f"Invalid seeds. All seeds must be between 0 and 4 inclusive")
    if any([f > 5 or f <= 0 for f in args.folds]):
      raise ValueError(f"Invalid folds. All folds must be between 1 and 5 inclusive")
  except ValueError as err:
    print(err)
  else:     
    main(args.run, trial, args.folds, args.seeds)
