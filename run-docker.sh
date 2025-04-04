#!/bin/bash


source env.sh


# Initialization
mkdir -p ${MODEL_PATH}
mkdir -p ${TEST_OUTPUTS_PATH}


# Main
docker run \
       -it \
       -v ${TRAIN_SET_PATH}:/challenge/training_data \
       -v ${HOLDOUT_SET_PATH}:/challenge/test_data \
       -v ${MODEL_PATH}:/challenge/model \
       -v ${TEST_OUTPUTS_PATH}:/challenge/holdout_outputs \
       --name ${CONTAINER_NAME} \
       --rm \
       --runtime=nvidia \
       ${IMAGE_NAME} \
       bash