import sys
import os
from queue import Queue
import multiprocessing as mp


__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

from rmixer.frontend_tcp import FrontendTcpThd
from rmixer.frontend_schedule import FrontendScheduleThd
from rmixer.device_context import DeviceContext
from util.util import timestamp, TcpAgent, TcpServer

def main():

    timestamp('frontend', 'start')

    # Load model list (task & data)
    model_list_file_name = sys.argv[1]
    cpu_workers = int(sys.argv[2])
    gpu_workers = int(sys.argv[3])
    job_policy = sys.argv[4]
    device_policy = sys.argv[5]
    if len(sys.argv) > 6:
        iter_num = int(sys.argv[6])
    else:
        iter_num = 100
   
    # need to decide device number in advance if there are multi devices
    if len(sys.argv) > 8:
        cpu_device_num = int(sys.argv[7])
        gpu_device_num = int(sys.argv[8])

    if len(sys.argv) > 9:
        beta = int(sys.argv[9])    
    # # the sixth argument decide which device, to implement load balanceing we need a device chooser
    # if len(sys.argv) > 6:
    #     device_id = sys.argv[6]
    # else:
    #     device_id = 0

    model_list = []
    with open(model_list_file_name) as f:
        for line in f.readlines():
            if len(line.split()) != 3:
                continue
            model_list.append([line.split()[0], line.split()[1], line.split()[2]])

    device_list = []

    # for i in range(1, gpu_device_num + 1 ):
    for i in range(gpu_device_num):
        device_list.append(DeviceContext('gpu', i, gpu_workers))
    for i in range(cpu_device_num):
        device_list.append(DeviceContext('cpu', i, cpu_workers))

    # Warm up CUDA and allocate shared cache
    # TODO: Check if this statement works 
    # XXX:here will output "INFO init", but it isn't when run it alone
    
    # Create workers
    worker_list = []

    # Create request queue and scheduler thread
    requests_queue = Queue()
    t_sch = FrontendScheduleThd( requests_queue, model_list, worker_list, device_list, 
                                 job_policy, device_policy, cpu_device_num, iter_num, beta)
    t_sch.start()
    # timestamp('frontend', 'start_schedule')

    # Accept connections
    server = TcpServer('localhost', 12346)
    # timestamp('tcp', 'listen')
    ask_cnt = 0
    while True:
        conn, _ = server.accept()
        agent = TcpAgent(conn)
        timestamp('tcp', 'connected')
        t_tcp = FrontendTcpThd(requests_queue, agent)
        t_tcp.start()
        ask_cnt+=1
        if(ask_cnt == len(model_list)):
            break;

    # Wait for end
    t_sch.join()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
