#!/bin/bash

source env/bin/activate
which python

NAME="macaw_ant"
LOG_DIR="log"
TASK_CONFIG="config/ant_dir/50tasks_offline.json"
MACAW_PARAMS="config/alg/standard.json"

./scripts/runner.sh $NAME $LOG_DIR $TASK_CONFIG $MACAW_PARAMS
