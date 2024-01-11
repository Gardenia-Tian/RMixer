import threading
import struct

class FrontendTcpThd(threading.Thread):
    def __init__(self, qout, agent):
        """
        Constructor for FrontendTcpThd thread.

        Parameters:
        - qout: A queue for putting received data.
        - agent: The TCP socket connection with the client.
        """
        super(FrontendTcpThd, self).__init__()
        self.qout = qout
        self.agent = agent

    def run(self):
        """
        The main execution method of the thread.

        Receives data from the TCP socket, unpacks it, and puts it into the queue.
        """
        # Receive the length of the task name (4 bytes)
        task_name_length_b = self.agent.recv(4)
        task_name_length = struct.unpack('I', task_name_length_b)[0]
        # Check if the task name length is not 0
        if task_name_length != 0:
            # Receive the task name based on the length
            task_name_b = self.agent.recv(task_name_length)
            task_name = task_name_b.decode()

            # Receive the length of the config file name (4 bytes)
            config_file_length_b = self.agent.recv(4)
            config_file_length = struct.unpack('I', config_file_length_b)[0]
            
            # Receive the config file based on the length
            config_file_b = self.agent.recv(config_file_length)
            config_file = config_file_b.decode()
            
            # Receive the length of the identifier (4 bytes)
            identifier_length_b = self.agent.recv(4)
            identifier_length = struct.unpack('I', identifier_length_b)[0]
            
            # Receive the identifier based on the length
            identifier_b = self.agent.recv(identifier_length)
            identifier = identifier_b.decode()

            # Put the received data into the queue
            self.qout.put((self.agent, task_name, config_file,identifier))
