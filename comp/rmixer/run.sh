#!/bin/bash
tasklist=("balance_24" "balance_72" "bandwidth_24" "bandwidth_72" "computing_24" "computing_72" "memory_24" "memory_72")
job_policylist=("dySequ" "dyArin" "dyType" "dyMemo" "static" "cognn" "dyMix")
device_policylist=("dySequ" "dyArin" "dyMemo" "dyWork" "dyMix")

cpuworker=2
gpuworker=4
job_policyid=6
device_policyid=4
taskid=6

task=${tasklist[${taskid}]}
job_policy=${job_policylist[${job_policyid}]}
device_policy=${device_policylist[${device_policyid}]}

iternum=500
beta=4
cpu_device_num=1
gpu_device_num=1


echo -e "\e[31m server start tasklist: ${task} job_policy:${job_policy} device_policy:${device_policy} iter_num:${iternum} gpu_num:${gpu_device_num} \e[0m"
cd ../../schedule/rmixer

# Start server script
bash run_server.sh ${cpuworker} ${gpuworker} ${task} ${job_policy} ${device_policy} ${iternum} ${cpu_device_num} ${gpu_device_num} ${beta} &

# Wait for the server to start and start listening
sleep 20


cd ../../schedule/client
bash run_client.sh ${cpuworker} ${gpuworker} ${task} ${job_policy} ${device_policy} ${iternum} ${cpu_device_num} ${gpu_device_num} ${beta}

wait
echo -e "\e[31m done tasklist: ${task} job_policy:${job_policy} device_policy:${device_policy} iter_num:${iternum} \e[0m"
            