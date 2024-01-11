import matplotlib.pyplot as plt
import matplotlib.dates as dates
from datetime import datetime, timedelta
import operator
import sys


def draw_running(filename,savepath='./'):
    memo_dict = {
        "amazon_dien":"6.75",
        "beauty_bert4rec":"4.36",
        "criteo_difm":"3.5",
        "alidisplay_dmr":"2.28",
        "amazon_bst":"2.03",
        "criteo_dcn2":"1.69",
        "criteo_widedeep":"1.43",
        "criteo_dlrm":"1.2",
        "amazon_din":"1.10",
        "kdd_dpin":"1.09",
        "criteo_dcn":"1.01",
        "criteo_deepfm":"1.0",
    }
    log_file = filename
    print(log_file)
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
        m_name = model_name.split()[1]
        m_id = model_name.split()[2]
        # m_name = ((model_name.split("time")[0].strip()).split("complete")[1].strip())model_name.split()[0]
        running_times.append([worker_id, m_name, start_time, end_time,total_time,operation[2],m_id])

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(16, 12))

    min_time = datetime.max
    max_time = datetime.min
    worker_names = []

    for i in range(len(running_times)):
        worker_id, model_name, start_time, end_time,total_time,device,model_id = running_times[i][0],running_times[i][1],running_times[i][2],running_times[i][3],running_times[i][4],running_times[i][5],running_times[i][6]
        min_time = min(min_time, start_time, end_time)
        max_time = max(max_time, start_time, end_time)

    start = min_time.replace(second=0, microsecond=0)
    end = (max_time + timedelta(minutes=2)).replace(second=0, microsecond=0)

    device_colors = {
        'gpu:0': 'tab:red',
        'gpu:1': 'tab:orange',
        'gpu:2': 'yellow',
        'gpu:3': 'tab:green',
        'gpu:4': 'tab:blue',
        'gpu:5': 'tab:purple',
        'cpu:0': 'tab:gray',
        'cpu:1': 'tab:pink'
    }




    xticks = [0]  # Offset in seconds relative to the first task
    xtick_labels = ['0']  # Relative time

    # Loop through each worker and plot their job duration
    for i in range(len(running_times)):
        worker_id, model_name, start_time, end_time,total_time,device,model_id = running_times[i][0],running_times[i][1],running_times[i][2],running_times[i][3],running_times[i][4],running_times[i][5],running_times[i][6]
        # total_time = timedelta(seconds=int(total_time))
        offset_seconds = (start_time - min_time).total_seconds()  


        color = device_colors.get(device, 'tab:red')  # 默认为红色
       
        ax.broken_barh([(offset_seconds, total_time)], (10*i, 9), facecolors=color)  
        worker_names.append(model_name)
        ax.text(offset_seconds, 10*i+5, device + " " + model_name + "-" + model_id + " " + str(total_time), ha='left', va='center', fontweight='bold')
        xticks.append(offset_seconds)
        xtick_labels.append('{}s'.format(offset_seconds))

    end = (max_time - min_time).total_seconds() + 20
    start = 0
    # Set up the chart
    ax.set_ylim(0, len(running_times)*10)
    ax.set_xlim(start, end)
    ax.set_xlabel('Time')


    title = filename.split('/')[-1]
    plt.title(title)

    start_str = min_time.strftime('%H:%M:%S')
    end_str = max_time.strftime('%H:%M:%S')
    plt.text(0.02, 0.95, '{:<10}-{:>10}  total: {:6.2f}'.format(start_str,end_str,(max_time - min_time).total_seconds()),transform=ax.transAxes,
        fontsize=16, fontweight='bold', va='top')

    plt.savefig(savepath + title + '.pdf',dpi=800)



if __name__ == '__main__':
    filename = sys.argv[1]
    savepath = sys.argv[2]
    
    draw_running(filename,savepath)
    print("done ", filename)