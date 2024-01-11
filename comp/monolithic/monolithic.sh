#!/bin/bash
getTiming(){ 
    start=$1
    end=$2
    start_s=$(echo $start | cut -d '.' -f 1)
    start_ns=$(echo $start | cut -d '.' -f 2)
    end_s=$(echo $end | cut -d '.' -f 1)
    end_ns=$(echo $end | cut -d '.' -f 2)
    time=$(( ( ${end_s#0} - ${start_s#0} ) * 1000 + ( ${end_ns#0} / 1000000 - ${start_ns#0} / 1000000 ) ))
    echo "$time ms"
}

tasklist=("balance_24" "balance_72" "bandwidth_24" "bandwidth_72" "computing_24" "computing_72" "memory_24" "memory_72")
tasklen=(24 72 24 72 24 72 24 72)

taskid=0


iternum=500
# 起始设备号
device_id=4
device_num=2
task=${tasklist[${taskid}]}

if [ $# -eq 4 ]; then
    task=$1
    iternum=$2
    device_id=$3
    device_num=$4
fi

forNum=$(((${tasklen[${tasklistId}]} + 1) / 2 - 1))

start=$(date +%s.%N)

for i in `seq 0 ${forNum}`;
do
    right=$((device_id + device_num - 1))
    for j in `seq ${device_id} ${right}`;
    do
        id_i=$((i*device_num+j-device_id))
        CUDA_VISIBLE_DEVICES=$j python ../../models/mps_train.py ../../schedule/data/${task}.txt ${id_i} $iternum $j >> series_${task}_${iternum}_gpu.log &
    done
    wait
done

end=$(date +%s.%N)
result=$(getTiming $start $end) 
echo $result >> monolithic_${task}_${iternum}_${device_num}.log

# cd /home/tianrui/code/mixrec/comp/gpu_mps
# bash gpu_mps.sh