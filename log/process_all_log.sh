#!/bin/bash

folder="./"
target_folder="./"

if [ $# -eq 2 ]; then
    folder=$1
    target_folder=$2
elif [ $# -eq 1 ]; then
    folder=$1
    target_folder=$1
fi

bash draw_log.sh $folder $target_folder
python get_time.py $folder $target_folder
python get_device_workload.py $folder $target_folder