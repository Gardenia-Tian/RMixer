import matplotlib.pyplot as plt
import matplotlib.dates as dates
from datetime import datetime, timedelta
import numpy as np
import operator
import sys
import os
import csv

def get_average_time(filename):
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
        # m_name = ((model_name.split("time")[0].strip()).split("complete")[1].strip())model_name.split()[0]
        running_times.append([worker_id, m_name, start_time, end_time,total_time,operation[2]])

    # 创建一个字典来存储每个模型的总运行时间和计数
    model_runtimes = {}
    model_counts = {}

    # 遍历running_times列表
    for entry in running_times:
        m_name = entry[1]
        total_time = entry[4]
        # 更新总运行时间和计数
        if m_name in model_runtimes:
            model_runtimes[m_name] += total_time
            model_counts[m_name] += 1
        else:
            model_runtimes[m_name] = total_time
            model_counts[m_name] = 1

    # 计算每个模型的平均运行时间
    model_averages = {}
    for m_name, total_time in model_runtimes.items():
        count = model_counts[m_name]
        average_time = total_time / count
        model_averages[m_name] = average_time

    sorted_averages = dict(sorted(model_averages.items(), key=lambda item: item[0]))
    return list(sorted_averages.keys()), list(sorted_averages.values())

def process_logs_in_folder(folder_path,target_folder):
    log_files = [f for f in os.listdir(folder_path) if f.endswith('.log')]
    log_files.sort()
    model_averages_dict = {}
    log_filenames = []
    all_model_name = ['avazu_widedeep', 'avazu_flen', 'alidisplay_dmr', 'amazon_bst', 'amazon_dien', 'amazon_din', 'beauty_bert4rec', 'criteo_dcn', 'criteo_dcn2', 'criteo_deepfm', 'criteo_difm', 'criteo_dlrm', 'criteo_widedeep', 'kdd_dpin']
    # all_single_time_500 = [ 71.0, 135.0, 108.0, 65.0, 34.0, 30.0, 38.0, 30.0, 107.0, 19.0, 29.0, 52.0]
    all_single_time_500 = [ 59.5589, 43.77, 71.0, 54.5, 108.0, 58.2, 34.0, 30.0, 38.0, 30.0, 33.0, 19.0, 29.0, 52.0]
    # all_single_time_100 = [ 16.0, 38.0, 74.0, 53.0, 8.0, 7.0, 9.0, 7.0, 24.0, 6.0, 6.0, 12.0]
    # 遍历日志文件并处理每个日志文件
    for log_file in log_files:
        log_path = os.path.join(folder_path, log_file)
        # temp = log_file.strip().split("_")
        # log_filename = temp[0]+'_'+ temp[1] + '_' + temp[4] + '_' + temp[4] + '_' + temp[7]
        log_filename = log_file
        log_filenames.append(log_filename)
        model_name, model_averages = get_average_time(log_path)
        model_averages_dict[log_filename] = model_averages
    
    model_averages_dict['single'] = [all_single_time_500[all_model_name.index(model)] for model in model_name]

    colors = ['#FFC300', '#FF5733', '#C70039', '#900C3F', '#581845', '#2E4053', '#1E8449', '#1ABC9C', '#16A085', '#2ECC71', '#27AE60', '#3498DB', '#2980B9', '#34495E']
    # 获取日志文件数量
    num_logs = len(model_averages_dict)

    # 动态计算 bar_width
    bar_width = 0.8 / num_logs
    # 生成 x 轴坐标
    x = np.arange(len(model_name))
    plt.figure(figsize=(12, 8))
    # 遍历每个日志文件并绘制柱状图
    for i, (log_name, runtimes) in enumerate(model_averages_dict.items()):
        # 计算每个柱状图的 x 坐标
        x_values = x + (i - (num_logs - 1) / 2) * bar_width
        plt.bar(x_values, runtimes, width=bar_width, label=log_name, color=colors[i])

    # 设置 x 轴标签和标题
    plt.xlabel('model')
    plt.ylabel('time')
    # plt.title('模型运行时间柱状图')

    # 设置 x 轴刻度标签
    plt.xticks(x, model_name, rotation=90)
   
    # 添加图例
    plt.legend()
    # plt.legend(loc='upper right', bbox_to_anchor=(2, 2))
    # 显示柱状图
    plt.savefig(target_folder  + 'time.pdf',dpi=800)
    



if __name__ == '__main__':
    num_arguments = len(sys.argv)
    folder_path = sys.argv[1]
    target_folder = folder_path
    if num_arguments >1:
        target_folder = sys.argv[2]
    process_logs_in_folder(folder_path, target_folder )
    