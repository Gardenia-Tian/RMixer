import os
import sys
import time
import struct
import statistics
import logging
import argparse

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

from util.util import TcpClient, timestamp

# STAR:deldete layers num, whether need to add depends on following design
def send_request(client, task_name, config_file, identifier):
    timestamp('client', 'before_request_%s' % task_name)

    # Serialize task name
    task_name_b = task_name.encode()
    task_name_length = len(task_name_b)
    task_name_length_b = struct.pack('I', task_name_length)

    # Serialize data
    config_file_b = config_file.encode()
    config_file_length = len(config_file_b)
    config_file_length_b = struct.pack('I', config_file_length)

    identifier_b = identifier.encode()
    identifier_length = len(identifier_b)
    identifier_length_b = struct.pack('I', identifier_length)

    timestamp('client', 'after_serialization')

    # Send task name / data
    client.send(task_name_length_b)
    client.send(task_name_b)
    client.send(config_file_length_b)
    client.send(config_file_b)
    client.send(identifier_length_b)
    client.send(identifier_b)


    timestamp('client', 'after_request_%s' % task_name)


def recv_response(client):
    reply_b = client.recv(4)
    reply = reply_b.decode()
    ('client', 'after_reply' + reply)


def close_connection(client):
    model_name_length = 0
    model_name_length_b = struct.pack('I', model_name_length)
    client.send(model_name_length_b)
    timestamp('client', 'close_connection')


def main():
    # Load model list (task & data)
    model_list_file_name = sys.argv[1]
    model_list = []
    with open(model_list_file_name) as f:
        for line in f.readlines():
            if len(line.split()) != 3:
                continue
            model_list.append([line.split()[0], line.split()[1], line.split()[2]])

    # Send training request
    client_train = []
    for i in range(len(model_list)):
        client_train.append(TcpClient('localhost', 12346))
        send_request(client_train[i], model_list[i][0], model_list[i][1], model_list[i][2])

#    time.sleep(4)

    timestamp('client', 'after_connect')
    time_1 = time.time()

    # Recv training reply
    for i in range(len(model_list)):
        recv_response(client_train[i])

    time_2 = time.time()
    duration = (time_2 - time_1) * 1000
    
    # Close connection
    for i in range(len(model_list)):
        close_connection(client_train[i])

    # time.sleep(1)
    timestamp('**********', '**********')
    timestamp(model_list_file_name, ' ')


if __name__ == '__main__':
    t1 = time.time()
    main()
    t2 = time.time()
    print(t2 - t1)
    path = "../../comp/rmixer/" + sys.argv[2]
    with open(path, "a+") as f:
        f.write(str(t2-t1) + ', ')