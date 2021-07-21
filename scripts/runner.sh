#!/bin/bash
# Run MACAW in a given configuration

NAME=$1
LOG_DIR=$2
TASK_CONFIG=$3
MACAW_PARAMS=$4
if [ -z $5 ]
then
    MACAW_OVERRIDE_PARAMS="config/alg/overrides/no_override.json"
else
    MACAW_OVERRIDE_PARAMS=$5
fi

CMD="python -u -m run --device cuda:0 --name $NAME --log_dir $LOG_DIR --task_config $TASK_CONFIG --macaw_params $MACAW_PARAMS --macaw_override_params $MACAW_OVERRIDE_PARAMS"

echo "***************************************************"
echo "***************************************************"
echo "RUNNING COMMAND:"
echo $CMD
echo "FROM DIRECTORY:"
pwd
echo "***************************************************"

echo "***************************************************"
echo "SAVING TO $LOG_DIR/$NAME"
echo "***************************************************"

echo "***************************************************"
echo "TASK CONFIGURATION"
cat $TASK_CONFIG
echo "***************************************************"

echo "***************************************************"
echo "MACAW CONFIGURATION"
cat $MACAW_PARAMS
echo "OVERRIDES"
cat $MACAW_OVERRIDE_PARAMS
echo "***************************************************"
echo "***************************************************"

$CMD
