#!/bin/bash

cpu_device_num=1
gpu_device_num=4
cpuworker=0
gpuworker=4
task=balance_24
job_policy=dySequ
device_policy=dySequ
iternum=500
beta=2

if [ $# -eq 9 ]; then
    cpuworker=$1
    gpuworker=$2
    task=$3
    job_policy=$4
    device_policy=$5
    iternum=$6
    cpu_device_num=$7
    gpu_device_num=$8
    beta=$9
fi

echo -e "\e[92m server start tasklist: ${task} job_policy:${job_policy} device_policy:${device_policy} iter_num:${iternum} gpu_num:${gpu_device_num} beta:${beta} \e[0m"

python main.py ../data/${task}.txt ${cpuworker} ${gpuworker} ${job_policy} ${device_policy} ${iternum} ${cpu_device_num} ${gpu_device_num} ${beta} > ../../log/server.log 2>&1 

echo -e "\e[92m server done tasklist: ${task} job_policy:${job_policy} device_policy:${device_policy} iter_num:${iternum} gpu_num:${gpu_device_num} beta:${beta} \e[0m"

python ../../log/postprocess.py ${beta}beta_${task}_${job_policy}_${device_policy}_${iternum}iter_${cpu_device_num}cDevice_${gpu_device_num}gDevice