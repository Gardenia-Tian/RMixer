import matplotlib.pyplot as plt
import matplotlib.dates as dates
from datetime import datetime, timedelta
import numpy as np
import operator
import sys
import os
import csv

def get_running_times(filename):
    log_file = filename
    workers = {}
    with open(log_file, "r") as f:
        for line in f:
            if line.startswith('TIMESTAMP'):
                parts = line.strip().split(", ")
                timestamp = datetime.strptime(parts[1], "%H:%M:%S")
                if parts[2].startswith("worker"):
                    worker_id = parts[2].split()[1]
                    device = parts[2].split()[2]
                    if worker_id not in workers:
                        workers[worker_id] = []
                    workers[worker_id].append((timestamp, parts[3], device))

    running_times = {}
    for worker_id in workers:

        # Sort the operations by timestamp
        operations = sorted(workers[worker_id], key=operator.itemgetter(0))

        # Group the operations by model name
        models = {}
        for operation in operations:
            if operation[1] not in models:
                models[operation[1]] = []
            models[operation[1]].append(operation)

        # Compute the running times for each model
        start_time = None
        end_time = None
        total_time = 0
        
        for model_name in models:
            for operation in models[model_name]:
                if operation[1].startswith("import"):
                    start_time = operation[0]     
                elif operation[1].startswith("complete"):
                    end_time = operation[0]
        total_time = (end_time - start_time).total_seconds()  
        m_name = model_name.split()[1]
        if operation[2] not in running_times:
            running_times[operation[2]] = 0
        running_times[operation[2]] = running_times[operation[2]] + total_time
    
    
    return dict(sorted(running_times.items(), key=lambda item: item[0]))


def draw_device_workload_for_each_file(filename):
    """
    Draw a bar chart representing the workload on each device for a single log file.

    Args:
        filename (str): Path to the log file.

    """
    running_times = get_running_times(filename)
    devices = list(running_times.keys())
    times = list(running_times.values())

    plt.figure(figsize=(10, 6))  # Set the figure size
    plt.bar(devices, times, color='blue')  # Create a bar chart

    # Add title and labels
    plt.title('Time for Each Device')
    plt.xlabel('Device')
    plt.ylabel('Time')
    file_name = os.path.basename(filename)
    # Display the bar chart
    plt.savefig('workload_' + file_name + '.pdf', dpi=800)

def draw_device_workload_for_folder(folder_path, target_folder):
    """
    Draw a grouped bar chart representing the workload on each device for multiple log files in a folder.

    Args:
        folder_path (str): Path to the folder containing log files.
        target_folder (str): Path to the target folder to save the generated chart.

    """
    log_files = [f for f in os.listdir(folder_path) if f.endswith('.log')]
    
    log_filenames = []
    workload_for_logs = {}
    workload_for_logs_unsorted = {}
    # Iterate through log files and process each log file
    for log_file in log_files:
        log_path = os.path.join(folder_path, log_file)
        workload_for_logs_unsorted[log_file] = get_running_times(log_path)
        
    workload_for_logs = dict(sorted(workload_for_logs_unsorted.items(), key=lambda item: item[0]))
    print(workload_for_logs)
    log_files = list(workload_for_logs.keys())
    for log_file in log_files:
        temp = log_file.strip().split("_")
        temp_name = log_file
        log_filenames.append(temp_name)
    # Find the length of the longest running time list
    max_length = max(len(data) for data in workload_for_logs.values())
    logs_with_max_device = max(workload_for_logs, key=lambda k: len(workload_for_logs[k]))
    all_devices = list(workload_for_logs[logs_with_max_device].keys())
    # Fill in missing values in the dictionary
    for log_name, running_times in workload_for_logs.items():
        for device in all_devices:
            if device not in running_times:
                running_times[device] = 0
        
        workload_for_logs[log_name] = dict(sorted(running_times.items(), key=lambda item: item[0]))
            
    # Extract log names and corresponding running time dictionaries
    log_names = list(workload_for_logs.keys())

    # Dynamically calculate width
    total_width = 0.8  # Total width
    num_logs = len(log_files)  # Number of logs
    num_devices = len(all_devices)
    width = total_width / num_devices  # Calculate width

    # Create a grouped bar chart
    plt.figure(figsize=(12, 8))  # Set the figure size

    ind = np.arange(num_logs)  # Position of each group of bars
    colors = ['#6A0DAD', '#9D50BB', '#D4657C', '#FFA500', '#00CCB5', '#00A3E0', '#0F4C81',]

    for i in range(len(all_devices)):
        running_times = []
        for log_name in log_files:
            running_times.append(workload_for_logs[log_name][all_devices[i]])
        plt.bar(ind + i*width, running_times, width=width, label=all_devices[i], color=colors[i])

    # Add title and labels
    plt.title(folder_path)
    plt.xlabel('Log')
    plt.ylabel('Time')

    # Set x-axis labels
    plt.xticks(ind + (total_width - width) / 2, log_filenames, rotation=30)

    # Display legend
    plt.legend()

    # Display the chart
    plt.savefig(target_folder + 'workload.pdf', dpi=800)   


if __name__ == '__main__':
    num_arguments = len(sys.argv)
    folder_path = sys.argv[1]
    target_folder = folder_path
    if num_arguments > 2:
        target_folder = sys.argv[2]
    draw_device_workload_for_folder(folder_path, target_folder)
    