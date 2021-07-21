#!/bin/bash

if [[ $(hostname) != "iris-ws-3.stanford.edu" ]]; then
    eval "$(conda shell.bash hook)"
    conda activate macaw
    which python
fi

LOG_DIR="log/iclr_rebuttal/task"
MACAW_PARAMS="config/alg/standard.json"
OVERRIDE="config/alg/overrides/1instance.json"

########################################################################

NAME="macaw_vel_sixteenth"
TASK_CONFIG="config/cheetah_vel/sixteenth_tasks_offline.json"
./scripts/runner.sh $NAME $LOG_DIR $TASK_CONFIG $MACAW_PARAMS $OVERRIDE &

########################################################################

NAME="macaw_vel_half"
TASK_CONFIG="config/cheetah_vel/half_tasks_offline.json"
./scripts/runner.sh $NAME $LOG_DIR $TASK_CONFIG $MACAW_PARAMS $OVERRIDE &

########################################################################

NAME="macaw_vel_quarter"
TASK_CONFIG="config/cheetah_vel/quarter_tasks_offline.json"
./scripts/runner.sh $NAME $LOG_DIR $TASK_CONFIG $MACAW_PARAMS $OVERRIDE &

########################################################################

NAME="macaw_vel_eighth"
TASK_CONFIG="config/cheetah_vel/eighth_tasks_offline.json"
./scripts/runner.sh $NAME $LOG_DIR $TASK_CONFIG $MACAW_PARAMS $OVERRIDE

########################################################################
