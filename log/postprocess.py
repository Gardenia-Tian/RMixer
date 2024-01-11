import time
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from datetime import datetime, timedelta
import operator

def draw_running(log_file, save_name):
    """
    Draw a Gantt chart representing the running times of different models on worker devices.

    Args:
        log_file (str): Path to the log file.
        save_name (str): Name to save the generated chart.

    """
    workers = {}
    with open(log_file, "r") as f:
        for line in f:
            parts = line.strip().split(", ")
            timestamp = datetime.strptime(parts[1], "%H:%M:%S")

            if parts[2].startswith("worker"):
                worker_id = parts[2].split()[1]
                device = parts[2].split()[2]
                if worker_id not in workers:
                    workers[worker_id] = []
                workers[worker_id].append((timestamp, parts[3], device))

    running_times = []
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
        running_times.append([worker_id, model_name, start_time, end_time, total_time, operation[2]])

    # Initialize the plot
    fig, ax = plt.subplots()

    min_time = datetime.max
    max_time = datetime.min
    worker_names = []

    for i in range(len(running_times)):
        worker_id, model_name, start_time, end_time, total_time, device = running_times[i][0], \
                                                                          running_times[i][1], \
                                                                          running_times[i][2], \
                                                                          running_times[i][3], \
                                                                          running_times[i][4], \
                                                                          running_times[i][5]
        min_time = min(min_time, start_time, end_time)
        max_time = max(max_time, start_time, end_time)

    start = min_time.replace(second=0, microsecond=0)
    end = (max_time + timedelta(minutes=2)).replace(second=0, microsecond=0)

    xticks = [0]  # Offset seconds relative to the first task
    xtick_labels = ['0']  # Relative time

    # Loop through each worker and plot their job duration
    for i in range(len(running_times)):
        worker_id, model_name, start_time, end_time, total_time, device = running_times[i][0], \
                                                                         running_times[i][1], \
                                                                         running_times[i][2], \
                                                                         running_times[i][3], \
                                                                         running_times[i][4], \
                                                                         running_times[i][5]
        model_name = (model_name.split("time")[0].strip()).split("complete")[1].strip()
        offset_seconds = (start_time - min_time).total_seconds()

        if device == 'gpu':
            ax.broken_barh([(offset_seconds, total_time)], (10 * i, 9), facecolors='tab:blue')
        else:
            ax.broken_barh([(offset_seconds, total_time)], (10 * i, 9), facecolors='tab:red')
        worker_names.append(model_name)
        ax.text(offset_seconds, 10 * i + 5, model_name, ha='left', va='center')
        xticks.append(offset_seconds)
        xtick_labels.append('{}s'.format(offset_seconds))

    end = (max_time - min_time).total_seconds() + 20
    start = 0
    # Set up the chart
    ax.set_ylim(0, len(running_times) * 10)
    ax.set_xlim(start, end)
    ax.set_xlabel('Time')
    ax.set_yticks([10 * i + 5 for i in range(len(running_times))])
    ax.set_yticklabels(worker_names)

    plt.savefig(save_name + '.pdf', dpi=800)
    print((max_time - min_time).total_seconds())
    print(max_time, min_time)

# Obtain current timestamp
t = time.localtime()
t_utc8 = time.gmtime(time.time() + 8 * 3600)  # Add time difference to get UTC+8 time
timestamp = time.strftime("_%m_%d_%H_%M_%S", t_utc8)

current_dir = os.path.dirname(os.path.abspath(__file__))


# Read the last non-empty line from the client log file
with open(current_dir + "/client.log", "r") as input_file:
    lines = input_file.readlines()
    last_line = ""
    for line in reversed(lines):
        if line.strip():
            last_line = line
            break

# Read content from the server log file
with open(current_dir + '/server.log', 'r') as file:
    content = file.readlines()

# Filter and sort lines
timestamp_dict = {}
other_lines = []
for line in content:
    if line.startswith('W') or line.startswith('Epoch 0'):
        continue
    elif line.startswith('TIMESTAMP'):
        # Classify line information by time; store information with the same time in the same dictionary value
        time_str = line.split(',')[1].strip()
        if time_str in timestamp_dict:
            timestamp_dict[time_str].append(line)
        else:
            timestamp_dict[time_str] = [line]
    else:
        other_lines.append(line)

# Sort and output based on time
sorted_timestamp = sorted(timestamp_dict.keys())
filename = current_dir + '/' + sys.argv[1] + timestamp + '.log'
with open(filename, 'w') as file:
    for time_str in sorted_timestamp:
        # Output all information at the same time
        for line in timestamp_dict[time_str]:
            file.write(line)

save_name = current_dir + '/' + sys.argv[1] + timestamp + '.pdf'

if last_line:
    with open(filename, 'a') as file:
        file.write(last_line)
        for line in other_lines:
            file.write(line)
