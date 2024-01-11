# Returns the device with the largest remaining memory
def get_device_memo(device_list):
    max_cpu_device_mem = 0
    max_gpu_device_mem = 0

    cpu_device = -1
    gpu_device = -1

    for device in device_list:
        if device.device_type == 'gpu' and device.mem_total - device.mem_used > max_gpu_device_mem and device.worker_num < device.max_worker_num:
            max_gpu_device_mem = device.mem_total - device.mem_used
            gpu_device = device
        elif device.device_type == 'cpu' and device.mem_total - device.mem_used > max_cpu_device_mem and device.worker_num < device.max_worker_num:
            max_cpu_device_mem = device.mem_total - device.mem_used
            cpu_device = device

    if gpu_device != -1 :
        return gpu_device
    
    return cpu_device


# Return the first executable device in order
def get_device_sequ(device_list):
    for device in device_list:
        if device.worker_num < device.max_worker_num:
            return device
    return -1

# Returns the device with the fewest workers
def get_device_work(device_list):
    cpu_device = -1
    gpu_device = -1
    min_cpu_device_num = 1000
    min_gpu_device_num = 1000
    for device in device_list:
        if device.device_type == 'gpu' and device.worker_num < device.max_worker_num:
            if (device.worker_num < min_gpu_device_num) :
                gpu_device = device
                min_gpu_device_num = device.worker_num
        elif device.device_type == 'cpu' and device.worker_num < device.max_worker_num:
            if (device.worker_num < min_cpu_device_num) :
                cpu_device = device
                min_cpu_device_num = device.worker_num
    if gpu_device != -1:
        return gpu_device
    return cpu_device

# Return the device with the lowest usage
def get_device_arin(device_list):
    min_cpu_device_util = 1000000
    min_gpu_device_util = 1000000

    cpu_device = -1
    gpu_device = -1
    for device in device_list:
        if device.device_type == 'gpu' and device.utilization < min_gpu_device_util:
            min_gpu_device_util = device.utilization
            gpu_device = device
        elif device.device_type == 'cpu' and device.utilization < min_cpu_device_util:
            min_cpu_device_util = device.utilization
            cpu_device = device
    if gpu_device != -1:
        return gpu_device
    return cpu_device


# Return the device with the largest remaining memory
def get_device_mix(device_list):
    min_gpu_device_util = 100
    min_cpu_device_util = 100
    min_gpu_device_worker = 1000
    min_cpu_device_worker = 1000
    min_gpu_device_memo = float('inf')
    min_cpu_device_memo = float('inf')

    cpu_device = -1
    gpu_device = -1

    
    for device in device_list:
        if device.device_type == 'gpu' and device.mem_used < min_gpu_device_memo and device.worker_num < device.max_worker_num:
            min_gpu_device_memo = device.mem_used
            gpu_device = device
        elif device.device_type == 'cpu' and device.mem_used < min_cpu_device_memo and device.worker_num < device.max_worker_num:
            min_cpu_device_memo = device.mem_used
            cpu_device = device
    return gpu_device, cpu_device



def get_device_policy(device_list, device_policy):
    if device_policy == "dyWork":
        return get_device_work(device_list)
    elif device_policy == "dySequ":
        return get_device_sequ(device_list)
    elif device_policy == "dyArin":
        return get_device_arin(device_list)
    elif device_policy == 'dyMemo':
        return get_device_memo(device_list)
    elif device_policy == 'dyMix':
        return get_device_mix(device_list)
