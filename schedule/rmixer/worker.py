from multiprocessing import Process
import time
import os
from rmixer.worker_common import ModelSummary
from util.util import timestamp
from multiprocessing import Process

class WorkerProc(Process):
    def __init__(self, agent, pipe, model_name,device, iter_num, job_type = 0):
        super(WorkerProc, self).__init__()
        self.agent = agent
        self.pipe = pipe
        self.model_name = model_name
        self.device = device
        self.iter_num = iter_num
        self.model_name.append(job_type)
        
    def run(self):
        self.id = os.getpid()

        t1 = time.time()

        model_summary = ModelSummary(self.model_name, self.device, self.iter_num)
        timestamp('worker {:<10d}'.format(self.id)+ self.device, 'import models  ' + self.model_name[0] + ' '+ self.model_name[2])
        model_summary.execute()
        t2 = time.time()

        timestamp('worker {:<10d}'.format(self.id)+ self.device, 'complete  {:15} {:2}'.format(self.model_name[0],self.model_name[2]) + '  time  {:8.2f}'.format(t2 - t1))

        self.pipe.send(self.model_name)
        self.agent.send(b'FNSH')
