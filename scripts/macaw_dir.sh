#!/bin/bash

source env/bin/activate
which python

NAME="macaw_dir"
LOG_DIR="log/iclr_rebuttal/multiseed"
TASK_CONFIG="config/cheetah_dir/2tasks_offline.json"
MACAW_PARAMS="config/alg/standard.json"

./scripts/runner.sh $NAME $LOG_DIR $TASK_CONFIG $MACAW_PARAMS
