import threading
import random

from util.util import timestamp
from rmixer.job_policy import *
from rmixer.device_policy import *
from rmixer.job_context import JobContext
import warnings
warnings.filterwarnings("ignore")
from models.utils.utils_single import load_yaml

    
class FrontendScheduleThd(threading.Thread):
    def __init__(self, qin, model_list, worker_list, device_list, job_policy, device_policy, cpu_device_num, iter_num=100, beta=2):
        super(FrontendScheduleThd, self).__init__()
        self.model_list = model_list
        self.qin = qin
        self.job_policy = job_policy
        self.device_policy = device_policy
        self.iter_num = iter_num
        self.beta=beta
        
        self.device_list = device_list
        self.cpu_device_num = cpu_device_num
        self.gpu_device_num = len(device_list) - cpu_device_num
        self.res_counter = 0
        # Indicates the current device index
        self.device_ptr = 0


    def run(self):
        timestamp('schedule', 'start')
        # Load models
        job_contexts = {}
        
        for model_name in self.model_list:
            hash_name = hash('{}_{}_{}'.format(model_name[0], model_name[1],model_name[2]))
            job_contexts[hash_name] = self._load_model(model_name)

        timestamp('schedule', 'load_model')

        job_list = []
        while True:
            # Get request           
            agent, task_name, config_file_name, identifier = self.qin.get()
            job_list.append([agent, task_name, config_file_name, identifier])
            timestamp('schedule', 'get_request')
            if len(job_list) == len(self.model_list):
                break
        
        # Print device information
        for device in self.device_list:
            timestamp("device: {:3s}:{:2} ".format(device.device_type, device.device_id), "max_worker:{:3}".format(device.max_worker_num))

        self._run_priority(job_list, job_contexts)  
        self._run_schedule(job_list, job_contexts)
        

    ###########################################################
    #                  basic method                           #
    ###########################################################
    def _load_model(self, model_name):
        """
        Load a model based on the provided model name.

        Parameters:
        - model_name: A list containing the model file and model config.

        Returns:
        - job_context: An instance of JobContext containing the loaded model.
        """
        model_config = load_yaml(model_name[1])
        job_context = JobContext(model_name[0],model_config)
        return job_context

    def _recv_message(self,job_contexts):
        """
        Receive messages from devices.

        Parameters:
        - job_contexts: A dictionary containing job contexts.

        Updates the result counter based on finished jobs.
        """
        for device in self.device_list:
            finish_counter = device.recv_message(job_contexts)
            self.res_counter += finish_counter
            if finish_counter != 0:
                timestamp('schedule', 'all finished: {:3}'.format(self.res_counter))

    def _run_priority(self,job_list, job_contexts):
        """
        Run the priority-based scheduling algorithm.

        Parameters:
        - job_list: A list of jobs to be scheduled.
        - job_contexts: A dictionary containing job contexts.

        Depending on the job_policy, it sorts the job_list accordingly.
        """
        if self.job_policy == "static":
            job_list = sort_job_memo(job_list, job_contexts)
        elif self.job_policy == "cognn":
            job_list = sort_job_memo(job_list, job_contexts)
        else:
            if self.job_policy == "dyMemo":
                job_list = sort_job_memo(job_list, job_contexts)
            elif self.job_policy == "dyArin":
                job_list = sort_job_arin(job_list, job_contexts)
            elif self.job_policy == "dyRand":
                random.shuffle(job_list)
            elif self.job_policy == 'dySequ':
                job_list = sort_job_type(job_list, job_contexts)
            elif self.job_policy == 'dyType':
                job_list = sort_job_memo(job_list, job_contexts)
            elif self.job_policy == "dyMix":
                job_list = sort_job_arin(job_list, job_contexts)
        
    def _run_schedule(self,job_list, job_contexts):
        """
        Run the scheduling algorithm based on the selected policy.

        Parameters:
        - job_list: A list of jobs to be scheduled.
        - job_contexts: A dictionary containing job contexts.

        Depending on the job_policy, it runs the corresponding scheduling algorithm.
        """
        if self.job_policy == "static" or self.job_policy == "cognn":
            self._static_schedule_multi(job_list, job_contexts)
        elif self.job_policy == "dySequ" or  \
             self.job_policy == "dyMemo" or  \
             self.job_policy == "dyArin" or  \
             self.job_policy == "dyType" or  \
             self.job_policy == "dyRand" or  \
             self.job_policy == "dyBase":
            self._dynamic_schedule_multi(job_list, job_contexts)
        elif self.job_policy == 'dyMix':
            self._dynamic_schedule_multi_mix(job_list, job_contexts)

    ###########################################################
    #                  multi card method                      #
    ###########################################################
    
    def _static_schedule_multi(self, job_list, job_contexts):
        """
        Synchronization scheduling algorithm for multi-card setup.

        Parameters:
        - job_list: A list of jobs to be scheduled.
        - job_contexts: A dictionary containing job contexts.

        Schedules jobs based on a synchronous policy.
        """
        i = 0
        while i < len(job_list):
            this_loop_jobs_num = 0
            while True:
                device = get_device_policy(self.device_list, self.device_policy)
                # 如果返回-1, 说明当前没有可执行的设备
                if device == -1:
                    break
                timestamp('schedule', 'get device {:3s}:{:2}'.format(device.device_type, device.device_id))
                job, job_type = get_job_policy(job_list, job_contexts, self.job_policy, device)
                if job == -1:
                    break
                timestamp("schedule", "device: {:3s}:{:2} get job: {:10s} {:2}".format(device.device_type, device.device_id, job[1],job[3]))    

                device.assign_job(job, job_contexts, self.model_list, self.iter_num, job_type)  
                i += 1
                this_loop_jobs_num += 1
            
            timestamp("--------------dispatch round--------------","")

            local_res_counter = self.res_counter
            while True:
                self._recv_message(job_contexts)
                
                if self.res_counter - local_res_counter == this_loop_jobs_num:
                    break
            timestamp("--------------reply    round--------------","")
            
    def _dynamic_schedule_multi(self, job_list, job_contexts):   
        """
        Asynchronous scheduling of multiple jobs on available devices.

        Args:
            job_list (list): List of jobs to be scheduled.
            job_contexts (dict): Dictionary containing job contexts.

        """
        while self.res_counter < len(job_list):
            device = get_device_policy(self.device_list, self.device_policy)
            # If -1 is returned, there are currently no available devices
            if device == -1:
                self._recv_message(job_contexts)
                continue 
            
            # Only check if the job is executable, without changing any state
            job, job_type = get_job_policy(job_list, job_contexts, self.job_policy, device)
            if job == -1:
                self._recv_message(job_contexts)
                continue
            timestamp("device: {:3s}:{:2} ".format(device.device_type, device.device_id), "get job: {:10s} {:2}".format(job[1],job[3]))    
            # Modify states here
            device.assign_job(job, job_contexts, self.model_list, self.iter_num, job_type)

    def _dynamic_schedule_multi_mix(self, job_list, job_contexts):
        """
        Asynchronous scheduling of multiple jobs on a mix of GPU and CPU devices.

        Args:
            job_list (list): List of jobs to be scheduled.
            job_contexts (dict): Dictionary containing job contexts.

        """
        cpu_power = 0
        gpu_power = 0
        for device in self.device_list:
            if device.device_type == 'cpu':
                cpu_power += device.max_worker_num
            if device.device_type == 'gpu':
                gpu_power += device.max_worker_num
        threshold = float(gpu_power) / (gpu_power + cpu_power)
        timestamp('schedule', 'threshold : {:3.2f}  beta : {:3}'.format(threshold,self.beta))

        while self.res_counter < len(job_list):
            gpu_device, cpu_device = get_device_mix(self.device_list) 

            device = gpu_device
            job, job_index = get_job_mix(job_list, job_contexts, device)
            if job == -1 and self.res_counter < len(job_list) - self.beta * gpu_power:
                device = cpu_device  
                job, job_index = get_job_mix(job_list, job_contexts, device)
                if job == -1 or job_index < threshold * len(job_list):
                    self._recv_message(job_contexts)
                    continue
                device.assign_job(job, job_contexts, self.model_list, self.iter_num, job_index)
            elif job == -1:
                self._recv_message(job_contexts)
                continue
            else:
                device.assign_job(job, job_contexts, self.model_list, self.iter_num, job_index)
            
