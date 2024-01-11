import paddle
import os
import paddle.nn as nn
import time
import logging
import sys
import importlib
import paddle.profiler as profiler

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

from utils.utils_single import load_yaml, load_dy_model_class, get_abs_model, create_data_loader
from utils.save_load import load_model, save_model
from paddle.io import DistributedBatchSampler, DataLoader
import argparse

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def my_on_trace_ready(prof): # 定义回调函数，性能分析器结束采集数据时会被调用
      callback = profiler.export_chrome_tracing('./profiler_log') # 创建导出性能数据到 profiler_demo 文件夹的回调函数
      callback(prof)  # 执行该导出函数
      prof.summary(sorted_by=profiler.SortedKeys.GPUTotal) 

def parse_args():
    parser = argparse.ArgumentParser(description='paddle-rec run')
    parser.add_argument("-m", "--config_yaml", type=str)
    parser.add_argument("-o", "--opt", nargs='*', type=str)
    args = parser.parse_args()
    args.abs_dir = os.path.dirname(os.path.abspath(args.config_yaml))
    args.config_yaml = get_abs_model(args.config_yaml)
    return args

def main(args):
    # load config
    config = load_yaml(args.config_yaml)
    dy_model_class = load_dy_model_class(args.abs_dir)
    config["config_abs_dir"] = args.abs_dir
    # modify config from command
    # if args.opt:
    #     for parameter in args.opt:
    #         parameter = parameter.strip()
    #         key, value = parameter.split("=")
    #         if type(config.get(key)) is int:
    #             value = int(value)
    #         if type(config.get(key)) is float:
    #             value = float(value)
    #         if type(config.get(key)) is bool:
    #             value = (True if value.lower() == "true" else False)
    #         config[key] = value

    # tools.vars
    use_gpu = config.get("runner.use_gpu", True)
    use_auc = config.get("runner.use_auc", False)
    train_data_dir = config.get("runner.train_data_dir", None)
    epochs = config.get("runner.epochs", None)
    print_interval = config.get("runner.print_interval", None)
    train_batch_size = config.get("runner.train_batch_size", None)
    model_save_path = config.get("runner.model_save_path", "model_output")
    model_init_path = config.get("runner.model_init_path", None)
    seed = config.get("runner.seed", 12345)
    paddle.seed(seed)

    # logger.info("**************common.configs**********")
    # logger.info(
    #     "use_gpu: {}, train_batch_size: {}, train_data_dir: {}, epochs: {}, print_interval: {}, model_save_path: {}".
    #     format(use_gpu, train_batch_size, train_data_dir, epochs, print_interval, model_save_path))
    # logger.info("**************common.configs**********")

    # use_gpu = True
    # place = paddle.set_device(use_gpu if use_gpu.startswith('gpu') else 'cpu')
    place = paddle.set_device('gpu')
    dy_model = dy_model_class.create_model(config)

    if model_init_path is not None:
        load_model(model_init_path, dy_model)

    prof= profiler.Profiler(targets=[profiler.ProfilerTarget.CPU, profiler.ProfilerTarget.GPU], scheduler = [2,10], on_trace_ready=my_on_trace_ready, timer_only=False)
    # to do : add optimizer function
    optimizer = dy_model_class.create_optimizer(dy_model, config)

    # logger.info("read data")
    train_dataloader = create_data_loader(config=config, place=place)

    last_epoch_id = config.get("last_epoch", -1)
    step_num = 0
    prof.start()
    for epoch_id in range(last_epoch_id + 1, epochs):
        
        # set train mode
        dy_model.train()
        metric_list, metric_list_name = dy_model_class.create_metrics()
        #auc_metric = paddle.metric.Auc("ROC")
        epoch_begin = time.time()
        interval_begin = time.time()
        train_reader_cost = 0.0
        train_run_cost = 0.0
        total_samples = 0
        reader_start = time.time()

        #we will drop the last incomplete batch when dataset size is not divisible by the batch size
        assert any(train_dataloader()), "train_dataloader is null, please ensure batch size < dataset size!"
        for batch_id, batch in enumerate(train_dataloader()):
            train_reader_cost += time.time() - reader_start
            optimizer.clear_grad()
            train_start = time.time()
            batch_size = len(batch[0])
            loss, metric_list, tensor_print_dict = dy_model_class.train_forward(dy_model, metric_list, batch, config)
            loss.backward()
            optimizer.step()
            # prof.step()
            train_run_cost += time.time() - train_start
            total_samples += batch_size

            if batch_id == 500:
                break

            if batch_id % print_interval == 0:
                metric_str = ""
                for metric_id in range(len(metric_list_name)):
                    metric_str += (metric_list_name[metric_id] + ":{:.6f}, ".format(metric_list[metric_id].accumulate()))
                tensor_print_str = ""
                if tensor_print_dict is not None:
                    for var_name, var in tensor_print_dict.items():
                        tensor_print_str += ("{}:".format(var_name) +str(var.numpy()).strip("[]") + ",")
                # logger.info(
                #     "epoch: {}, batch_id: {}, ".format(
                #         epoch_id, batch_id) + metric_str + tensor_print_str +
                #     " avg_reader_cost: {:.5f} sec, avg_batch_cost: {:.5f} sec, avg_samples: {:.5f}, ips: {:.5f} ins/s".
                #     format(train_reader_cost / print_interval, (
                #         train_reader_cost + train_run_cost) / print_interval,
                #         total_samples / print_interval, total_samples / (
                #             train_reader_cost + train_run_cost)))
                train_reader_cost = 0.0
                train_run_cost = 0.0
                total_samples = 0
            reader_start = time.time()
            step_num = step_num + 1
        # metric_str = ""
        # for metric_id in range(len(metric_list_name)):
        #     metric_str += (
        #         metric_list_name[metric_id] +
        #         ": {:.6f},".format(metric_list[metric_id].accumulate()))
        #     if use_auc:
        #         metric_list[metric_id].reset()


        # tensor_print_str = ""
        # if tensor_print_dict is not None:
        #     for var_name, var in tensor_print_dict.items():
        #         tensor_print_str += (
        #             "{}:".format(var_name) + str(var.numpy()).strip("[]") + ","
        #         )

        # logger.info("epoch: {} done, ".format(epoch_id) + metric_str +
        #             tensor_print_str + " epoch time: {:.2f} s".format(
        #                 time.time() - epoch_begin))

        # save_model(dy_model, optimizer, model_save_path, epoch_id, prefix='rec')
    prof.stop()            


if __name__ == '__main__':
    args = parse_args()
    t1 = time.time()
    main(args)
    t2 = time.time()
    print(t2 - t1)
