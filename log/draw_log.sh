#!/bin/bash

folder="./"
target_folder="./"
extension=".log"

if [ $# -eq 2 ]; then
    folder=$1
    target_folder=$2
fi

for filename in $folder/*$extension
do
    [ -e "$filename" ] || continue

    if [[ "$filename" == *"/server.log" || "$filename" == *"/client.log" ]]; then
        continue
    fi
    
    python draw_log.py $filename $target_folder
done
