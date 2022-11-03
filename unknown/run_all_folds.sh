#!/bin/bash
set -e
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
RUN_NAME=$1
if [ -z "$RUN_NAME" ]; then echo "Please supply run-num"; exit; else echo "Running all 5 folds for Experiment '$RUN_NAME'"; fi
cd $DIR
for FOLD in 1 2 3 4 5
do
    echo "Running Fold $FOLD"
    export CLOSED_CONFIG="configs/old_configs/paper_redo/closed/fold$FOLD/config.py"
    export OPEN_CONFIG="configs/old_configs/paper_redo/open/fold$FOLD/config.py"
    python tier3_main.py $RUN_NAME --test
done