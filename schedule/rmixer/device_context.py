import multiprocessing as mp
from util.util import *
from rmixer.worker import WorkerProc

class DeviceContext():
    def __init__(self, device_type, device_id, max_worker_num):
        self.device_type = device_type
        self.device_id = device_id
        self.max_worker_num = max_worker_num
        self.mem_total = 0
        self.mem_used = 0
        if self.device_type == 'cpu':
            self.memo_info = get_cpu_mem_info
            self.util_info = get_cpu_util_info
        elif self.device_type == 'gpu':
            self.memo_info = get_gpu_mem_info
            self.util_info = get_gpu_util_info
            self.mem_used += max_worker_num * 310

        self.mem_total, self.mem_used = self.memo_info(int(self.device_id))
        self.utilization = self.util_info(int(self.device_id))

        self.high_arin_type = 0
        self.high_memo_type = 0
        self.both_low_type = 0

        self.high_num = 0
        self.low_num = 0

        self.worker_list = []
        self.worker_num = 0
        self.finish_counter = 0

    # Determines whether a task can be executed on the current device
    def executable(self, job, job_contexts,job_policy = 'dynamic'):
        if job == -1:
            return False
        agent, task_name, config_file, identifier = job[0], job[1], job[2], job[3]
        job_context = job_contexts[hash('{}_{}_{}'.format(task_name, config_file,identifier))]
        
        job_memory = job_context.get_model_memory()
        # Determine if there are enough workers
        if self.worker_num < self.max_worker_num:
            # Get the current memory usage, if it is CPU then device_id is invalid
            total, used = self.memo_info(int(self.device_id))
           
            # If the memory difference is detected for an asynchronous policy, the system returns the error. 
            # If the memory difference is detected for a synchronous policy, 
            # the system must use the busy mechanism to wait for memory release to prevent OOM error
            if job_policy == 'dynamic':
                # Prevents inaccurate estimates or memory release delays
                if self.device_type == 'gpu' and used - self.mem_used > 1000:
                    return False
            else:
                # Prevents inaccurate estimates or memory release delays
                while self.device_type == 'gpu' and used - self.mem_used > 1000:
                    total,used = get_gpu_mem_info(int(self.device_id))

            if job_context.memory * 1.2 + self.mem_used < self.mem_total:
                return True
            else:
                return False
        return False

    # Assign job on this device
    def assign_job(self, job, job_contexts, model_list, iter_num, job_type = 0):
        agent, task_name, config_file, identifier = job[0], job[1], job[2], job[3]
        job_context = job_contexts[hash('{}_{}_{}'.format(task_name, config_file,identifier))]
        
        self.worker_num += 1
        self.mem_used += job_context.memory * 1.2
        self.utilization += job_context.arithmetic_intensity_dict[task_name]

        # update the record of job types
        if job_context.type == 1:
            self.high_arin_type += 1
        elif job_context.type == 0:
            self.high_memo_type += 1
        else:
            self.both_low_type += 1

        model_name = []
        for model in model_list:
            if model[0] == task_name and model[1] == config_file and model[2] == identifier:
                model_name = model

        p_parent, p_child = mp.Pipe()
        worker = WorkerProc(agent, p_child, model_name,self.device_type + ':' + str(self.device_id), iter_num, job_type)
        # start up job
        worker.start()
        self.worker_list.append((p_parent, worker))
        job_contexts[hash('{}_{}_{}'.format(task_name, config_file,identifier))].assign = True
        timestamp("schedule", "device: {:3s}:{:2}  tasknum: {:3}".format(self.device_type, self.device_id, len(self.worker_list)))


    # Poll each worker to check that the work is complete
    def recv_message(self, job_contexts):
        finish_counter = 0
        for i in range(len(self.worker_list)-1,-1,-1):
            new_pipe, worker = self.worker_list[i][0], self.worker_list[i][1]
            if new_pipe.poll():
                res = new_pipe.recv()
                task_name, config_file, identifier, job_type = res[0], res[1], res[2], res[3]
                job_contexts[hash('{}_{}_{}'.format(task_name, config_file,identifier))].finish = True
                job_context = job_contexts[hash('{}_{}_{}'.format(task_name, config_file,identifier))]
                self.worker_num -= 1
                self.mem_used = max(self.mem_used - job_context.memory * 1.2 ,308 * self.max_worker_num)
                self.utilization -= job_context.arithmetic_intensity_dict[task_name]
                if job_context.model_name in job_context.high_arin_type_set:
                    self.high_arin_type -= 1
                elif job_context.model_name in job_context.high_memo_type_set:
                    self.high_memo_type -= 1
                else:
                    self.both_low_type -= 1
                self.finish_counter += 1
                self.high_num -= job_type
                self.low_num -= (1-job_type)
                finish_counter += 1
                del self.worker_list[i]    
                timestamp('device: {:3s}:{:2}'.format(self.device_type, self.device_id), 'has finished: {:3}, latest finish_model {:2}-{:16s}'.format(self.finish_counter ,identifier, task_name))
        return finish_counter


