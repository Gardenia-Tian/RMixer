import functools

# Sort by memory size from largest to smallest
def sort_job_memo(job_list,job_contexts):
    def sort_mem(job1, job2):
        agent1, task_name1, config_file1, identifier1 = job1[0], job1[1], job1[2], job1[3]
        job1_context = job_contexts[hash('{}_{}_{}'.format(task_name1, config_file1, identifier1))]
        agent2, task_name2, config_file2, identifier2 = job2[0], job2[1], job2[2], job2[3]
        job2_context = job_contexts[hash('{}_{}_{}'.format(task_name2, config_file2, identifier2))]
        if job2_context.memory > job1_context.memory:
            return 1
        elif job2_context.memory == job1_context.memory:
            return 0
        else:
            return -1

    job_list.sort(key=functools.cmp_to_key(sort_mem))
    
    return job_list[:]

# Rank them in order of computational strength from greatest to least
def sort_job_arin(job_list,job_contexts):
    def sort_arin(job1, job2):
        agent1, task_name1, config_file1, identifier1 = job1[0], job1[1], job1[2], job1[3]
        job1_context = job_contexts[hash('{}_{}_{}'.format(task_name1, config_file1, identifier1))]
        agent2, task_name2, config_file2, identifier2 = job2[0], job2[1], job2[2], job2[3]
        job2_context = job_contexts[hash('{}_{}_{}'.format(task_name2, config_file2, identifier2))]
        if job2_context.arithmetic_intensity_dict[job2_context.model_name] > job1_context.arithmetic_intensity_dict[job1_context.model_name]:
            return 1
        elif job2_context.arithmetic_intensity_dict[job2_context.model_name] == job1_context.arithmetic_intensity_dict[job1_context.model_name]:
            return 0
        else:
            return -1
    job_list.sort(key=functools.cmp_to_key(sort_arin))
    return job_list[:]


def sort_job_type(job_list,job_contexts):
    def sort_type(job1, job2):
        agent1, task_name1, config_file1, identifier1 = job1[0], job1[1], job1[2], job1[3]
        job1_context = job_contexts[hash('{}_{}_{}'.format(task_name1, config_file1, identifier1))]
        agent2, task_name2, config_file2, identifier2 = job2[0], job2[1], job2[2], job2[3]
        job2_context = job_contexts[hash('{}_{}_{}'.format(task_name2, config_file2, identifier2))]
        if job2_context.type > job1_context.type:
            return 1
        elif job2_context.type < job1_context.type:
            return -1
        else:
            if job2_context.memory >= job1_context.memory:
                return 1
            elif job2_context.memory < job1_context.memory:
                return -1
            return 0
    
    job_list.sort(key=functools.cmp_to_key(sort_type))
    return job_list[:]


def get_job_front(job_list, job_contexts,device,job_policy='dynamic'):
    for job in job_list:
        agent, task_name, config_file, identifier = job[0], job[1], job[2], job[3]
        job_context = job_contexts[hash('{}_{}_{}'.format(task_name, config_file, identifier))]        
        if job_context.assign == False and job_context.finish == False:
            if device.executable(job, job_contexts, job_policy):
                return job
    return -1
    
def get_job_back(job_list, job_contexts,device,job_policy='dynamic'):
    for i in range(len(job_list)-1, -1, -1):
        agent, task_name, config_file, identifier = job_list[i][0],job_list[i][1],job_list[i][2],job_list[i][3]
        job_context = job_contexts[hash('{}_{}_{}'.format(task_name, config_file, identifier))]
        if job_context.assign == False and job_context.finish == False:
            if device.executable(job_list[i], job_contexts, job_policy):
                return job_list[i] 
    return -1

# Obtain jobs according to the type complementarity principle
def get_job_type(job_list, job_contexts, device):
    highest_type = 1
    max_type = max(device.high_arin_type, device.high_memo_type, device.both_low_type)

    if max_type == device.high_arin_type:
        highest_type = 1
    elif max_type == device.high_memo_type:
        highest_type = 0
    else:
        highest_type = -1

    for job in job_list:
        agent, task_name, config_file, identifier = job[0], job[1], job[2], job[3]
        job_context = job_contexts[hash('{}_{}_{}'.format(task_name, config_file, identifier))]        
        if job_context.assign == False and job_context.finish == False and job_context.type != highest_type:
            if device.executable(job, job_contexts):
                return job, job_context.type
    for job in job_list:
        agent, task_name, config_file, identifier = job[0], job[1], job[2], job[3]
        job_context = job_contexts[hash('{}_{}_{}'.format(task_name, config_file, identifier))]        
        if job_context.assign == False and job_context.finish == False:
            if device.executable(job, job_contexts):
                return job, job_context.type
    return -1, -2

# According to the principle of complementary calculation strength, the larger one will be taken from the back, and the smaller one will be taken from the past
def get_job_arin(job_list, job_contexts, device):
    job = -1
    job_arin = 0
    if device.high_num > device.low_num:
        job = get_job_back(job_list, job_contexts,device)
        if job != -1:
            device.low_num += 1
            job_arin = 0
    else:
        job = get_job_front(job_list, job_contexts,device)
        if job!=-1:
            device.high_num += 1
            job_arin = 1

    return job, job_arin

def get_job_memo(job_list, job_contexts, device):
    job = -1
    job_memo = 0
    if device.high_num > device.low_num:
        job = get_job_back(job_list, job_contexts,device)
        if job != -1:
            device.low_num += 1
            job_memo = 0
    else:
        job = get_job_front(job_list, job_contexts,device)
        if job!=-1:
            device.high_num += 1
            job_memo = 1

    return job, job_memo

# Sequential fetch job
def get_job_sequ(job_list, job_contexts, device):
    return get_job_front(job_list, job_contexts, device, 'static'),0

# Obtain jobs under a synchronization policy
def get_job_static(job_list, job_contexts, device):
    highest_type = 1
    max_type = max(device.high_arin_type, device.high_memo_type, device.both_low_type)

    if max_type == device.high_arin_type:
        highest_type = 1
    elif max_type == device.high_memo_type:
        highest_type = 0
    else:
        highest_type = -1

    for job in job_list:
        agent, task_name, config_file, identifier = job[0], job[1], job[2], job[3]
        job_context = job_contexts[hash('{}_{}_{}'.format(task_name, config_file, identifier))]        
        if job_context.assign == False and job_context.finish == False and job_context.type != highest_type:
            if device.executable(job, job_contexts,'static'):
                return job, job_context.type
    for job in job_list:
        agent, task_name, config_file, identifier = job[0], job[1], job[2], job[3]
        job_context = job_contexts[hash('{}_{}_{}'.format(task_name, config_file, identifier))]        
        if job_context.assign == False and job_context.finish == False:
            if device.executable(job, job_contexts,'static'):
                return job, job_context.type
    return -1, -2

# The policy for getting jobs when using the CPU
def get_job_mix(job_list, job_contexts, device):
    if device == -1:
        return -1,-2
    highest_type = 1
    max_type = max(device.high_arin_type, device.high_memo_type, device.both_low_type)

    if max_type == device.high_arin_type:
        highest_type = 1
    elif max_type == device.high_memo_type:
        highest_type = 0
    else:
        highest_type = -1

    if device.device_type == 'gpu':
        for i in range(len(job_list)):
            job = job_list[i]
            agent, task_name, config_file, identifier = job[0], job[1], job[2], job[3]
            job_context = job_contexts[hash('{}_{}_{}'.format(task_name, config_file, identifier))]
            if job_context.assign == False and job_context.finish == False and job_context.type != highest_type and device.executable(job, job_contexts):
                return job, job_context.type
        for job in job_list:
            agent, task_name, config_file, identifier = job[0], job[1], job[2], job[3]
            job_context = job_contexts[hash('{}_{}_{}'.format(task_name, config_file, identifier))]        
            if job_context.assign == False and job_context.finish == False:
                if device.executable(job, job_contexts):
                    return job, job_context.type
    elif device.device_type == 'cpu':
        for i in range(len(job_list)-1, -1, -1):
            agent, task_name, config_file, identifier = job_list[i][0],job_list[i][1],job_list[i][2],job_list[i][3]
            job_context = job_contexts[hash('{}_{}_{}'.format(task_name, config_file, identifier))]
            if job_context.assign == False and job_context.finish == False and device.executable(job_list[i], job_contexts):
                return job_list[i], i
        
    return -1,-2

# BMC strategy proposed in CoGNN 
def get_job_cognn(job_list, job_contexts, device):
    job = -1
    job_memo = 0
    if device.high_num > device.low_num:
        job = get_job_back(job_list, job_contexts,device,'static')
        if job != -1:
            device.low_num += 1
            job_memo = 0
    else:
        job = get_job_front(job_list, job_contexts,device,'static')
        if job!=-1:
            device.high_num += 1
            job_memo = 1

    return job, job_memo

def get_job_policy(job_list, job_contexts, job_policy, device):
    if job_policy == "dySequ":
        return get_job_sequ(job_list, job_contexts,device)
    elif job_policy == "dyArin":
        return get_job_arin(job_list, job_contexts, device)
    elif job_policy == "dyType":
        return get_job_type(job_list, job_contexts, device)
    elif job_policy == 'dyMemo':
        return get_job_memo(job_list, job_contexts, device)
    elif job_policy == 'dyRand':
        return get_job_sequ(job_list, job_contexts, device)
    elif job_policy == 'static':
        return get_job_static(job_list, job_contexts, device)
    elif job_policy == 'cognn':
        return get_job_cognn(job_list, job_contexts, device)
    elif job_policy == 'dyMix':
        # gpu优先执行计算强度高的, 所以按照计算强度的顺序返回
        return get_job_mix(device, job_list, job_contexts)