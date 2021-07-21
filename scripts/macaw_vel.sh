#!/bin/bash

source env/bin/activate
which python

NAME="macaw_vel_hotstart"
LOG_DIR="log"
TASK_CONFIG="config/cheetah_vel/40tasks_offline.json"
MACAW_PARAMS="config/alg/standard_loadvel.json"

./scripts/runner.sh $NAME $LOG_DIR $TASK_CONFIG $MACAW_PARAMS
