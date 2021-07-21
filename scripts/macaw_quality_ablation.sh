#!/bin/bash

source env/bin/activate
which python

LOG_DIR="log"
MACAW_PARAMS="config/alg/standard.json"
TASK_CONFIG="config/cheetah_vel/40tasks_offline.json"

########################################################################

NAME="macaw_vel_end"
OVERRIDE="config/alg/overrides/end.json"
./scripts/runner.sh $NAME $LOG_DIR $TASK_CONFIG $MACAW_PARAMS $OVERRIDE &

########################################################################

NAME="macaw_vel_middle"
OVERRIDE="config/alg/overrides/middle.json"
./scripts/runner.sh $NAME $LOG_DIR $TASK_CONFIG $MACAW_PARAMS $OVERRIDE &

########################################################################

NAME="macaw_vel_start"
OVERRIDE="config/alg/overrides/start.json"
./scripts/runner.sh $NAME $LOG_DIR $TASK_CONFIG $MACAW_PARAMS $OVERRIDE

########################################################################
