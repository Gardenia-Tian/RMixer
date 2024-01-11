#!/bin/bash
cpuworker=0
gpuworker=4
task=balance_24
job_policy=dySequ
device_policy=dySequ
iternum=500
beta=2

if [ $# -eq 9 ]; 
then
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

echo -e "\e[96m client start tasklist: ${task} job_policy:${job_policy} device_policy:${device_policy} iter_num:${iternum} \e[0m"

python -u client.py ../data/${task}.txt ${beta}beta_${task}_${cpu_device_num}cD_${gpu_device_num}gD_${job_policy}_${device_policy}_${iternum}.txt > ../../log/client.log 2>&1 

echo -e "\e[96m client done tasklist: ${task} job_policy:${job_policy} device_policy:${device_policy} iter_num:${iternum} \e[0m"
