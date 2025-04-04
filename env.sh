#!/bin/bash


# Docker configuration
export IMAGE_NAME=deeptryp
export CONTAINER_NAME=deeptryp


# Path configuration
training_data=/var/opt/ballas/physionet2025/
holdout_data=/var/opt/ballas/physio25_holdout_data_less/
results_path=/home/aballas/tmp/physionet2025/results/docker

export TRAIN_SET_PATH="${training_data}"
export HOLDOUT_SET_PATH="${holdout_data}"
export MODEL_PATH="${results_path}/model"
export TEST_OUTPUTS_PATH="${results_path}/holdout_outputs"