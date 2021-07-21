#!/bin/bash

source env/bin/activate "$(conda shell.bash hook)"
which python

LOG_DIR="log/NeurIPS3"
MACAW_PARAMS="config/alg/standard.json"

########################################################################

NAME="macaw_vel_end"
TASK_CONFIG="config/cheetah_vel/40tasks_offline.json"
OVERRIDE="config/alg/overrides/end.json"
./scripts/runner.sh $NAME $LOG_DIR $TASK_CONFIG $MACAW_PARAMS $OVERRIDE &

########################################################################

NAME="macaw_vel_middle"
TASK_CONFIG="config/cheetah_vel/40tasks_offline.json"
OVERRIDE="config/alg/overrides/middle.json"
./scripts/runner.sh $NAME $LOG_DIR $TASK_CONFIG $MACAW_PARAMS $OVERRIDE &

########################################################################

NAME="macaw_vel_start"
TASK_CONFIG="config/cheetah_vel/40tasks_offline.json"
OVERRIDE="config/alg/overrides/start.json"
./scripts/runner.sh $NAME $LOG_DIR $TASK_CONFIG $MACAW_PARAMS $OVERRIDE

########################################################################
