import sys
import time
import socket
import pynvml
import psutil
import logging

def timestamp(name, stage, logger_name='server'):
    # Gets the current timestamp
    current_time = time.time() + 8*3600 # Add the current time to the time difference to get UTC+8
    # Converts the timestamp to the struct_time type
    time_struct = time.gmtime(current_time)
    # Converts the struct_time type to seconds
    time_str = time.strftime("%H:%M:%S", time_struct)
    # print ('\033[91m TIMESTAMP, %s, %s, %s  \033[0m' % (name, stage, time_str), file=sys.stderr)
    print ('TIMESTAMP, %s, %s, %s' % (time_str, name, stage), file=sys.stderr)
    

def get_gpu_mem_info(gpu_id=4):
    pynvml.nvmlInit()
    if gpu_id < 0 or gpu_id >= pynvml.nvmlDeviceGetCount():
        print(r'The GPU corresponding to gpu_id {} does not exist!'.format(gpu_id))
        return 0, 0, 0

    handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
    total = round(meminfo.total / 1024 / 1024, 2)
    used = round(meminfo.used / 1024 / 1024, 2)
    free = round(meminfo.free / 1024 / 1024, 2)
    return total, used

def get_cpu_mem_info(cpu_id=0):
    meminfo = psutil.virtual_memory()
    total = round(meminfo.total / 1024 / 1024, 2)
    used = round(meminfo.used / 1024 / 1024, 2)
    return total, used


def get_gpu_util_info(gpu_id=4):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    # Obtain the current GPU usage
    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
    return utilization.gpu

def get_cpu_util_info(cpu_id=0):
    cpu_usage = psutil.cpu_percent(interval=1)
    return cpu_usage


class TcpServer():
    def __init__(self, address, port):
        self.address = address
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.address, self.port))
        self.sock.listen(1)

    def __del__(self):
        self.sock.close()

    def accept(self):
        conn, address = self.sock.accept()
        return conn, address


class TcpAgent:
    def __init__(self, conn):
        self.conn = conn

    def __del__(self):
        self.conn.close()

    def send(self, msg):
        self.conn.sendall(msg)

    def recv(self, msg_len):
        return self.conn.recv(msg_len, socket.MSG_WAITALL)

    def settimeout(self, t):
        self.conn.settimeout(t)


class TcpClient(TcpAgent):
    def __init__(self, address, port):
        super().__init__(None)
        self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.conn.connect((address, port))