#!/bin/bash
getTiming() {
    start=$1
    end=$2
    start_s=$(echo $start | cut -d '.' -f 1)
    start_ns=$(echo $start | cut -d '.' -f 2)
    end_s=$(echo $end | cut -d '.' -f 1)
    end_ns=$(echo $end | cut -d '.' -f 2)
    time=$(((10#$end_s - 10#$start_s) * 1000 + (10#$end_ns / 1000000 - 10#$start_ns / 1000000)))
    echo "scale=3; $time/1000" | bc
}

nvidia-cuda-mps-control -d
tasklist=("balance_24" "balance_72" "bandwidth_24" "bandwidth_72" "computing_24" "computing_72" "memory_24" "memory_72")
tasklen=(24 72 24 72 24 72 24 72)


GPU_device_num=4  # Set the number of GPUs you have
iternum=500  # Set your desired value for iternum
tasks_per_gpu=4  # Set the number of tasks per GPU
tasklistId=0

if [ $# -eq 2 ]; then
    GPU_device_num=$1
    tasklistId=$2
fi


total_tasks=${tasklen[$tasklistId]}
tasks_per_iteration=$((GPU_device_num * tasks_per_gpu))
total_iterations=$(( (total_tasks + tasks_per_iteration - 1) / tasks_per_iteration ))
output=${tasklist[$tasklistId]}_${GPU_device_num}GPU_mps.txt

echo -e "\e[42m\e[33m Start tasklist: ${tasklist[${tasklistId}]} policy: GPU_MPS \e[0m"

start=$(date +%s.%N)
for iteration in $(seq 0 $((total_iterations - 1))); do
    for gpu_task_id in $(seq 0 $((GPU_device_num - 1 ))); do
        start_task=$((iteration * tasks_per_iteration + gpu_task_id * tasks_per_gpu))
        end_task=$((start_task + tasks_per_gpu - 1))
        for taskId in $(seq $start_task $end_task); do
            if [ $taskId -ge $total_tasks ]; then
                break
            fi
            CUDA_VISIBLE_DEVICES=${gpu_task_id} python ../../models/mps_train.py ../../schedule/data/${tasklist[$tasklistId]}.txt ${taskId} $iternum gpu:$((gpu_task_id + 1)) >> ${output} &
        done
    done
    wait  # Wait for all background processes to finish
done
end=$(date +%s.%N)

echo -e "\e[42m\e[33m Done tasklist: ${tasklist[${tasklistId}]} policy: GPU_MPS \e[0m"

result=$(getTiming $start $end)
echo $result >>$output

echo quit | nvidia-cuda-mps-control