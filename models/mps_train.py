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

# os.environ['CUDA_VISIBLE_DEVICES'] = '4'

def read_list(model_list_file_name):
    model_list = []
    with open(model_list_file_name) as f:
        for line in f.readlines():
            if len(line.split()) < 2:
                continue
            model_list.append([line.split()[0], line.split()[1]])
    return model_list


def main(config_yaml, abs_dir, model_name, iter_num,device_id):
    # load config
    config = load_yaml(config_yaml)
    dy_model_class = load_dy_model_class(abs_dir)
    config["config_abs_dir"] = abs_dir
    
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

    logger.info("**************common.configs**********")
    logger.info(
        "use_gpu: {}, train_batch_size: {}, train_data_dir: {}, epochs: {}, print_interval: {}, model_save_path: {}".
        format(use_gpu, train_batch_size, train_data_dir, epochs, print_interval, model_save_path))
    logger.info("**************common.configs**********")

    # use_gpu = False
    place = paddle.set_device('gpu' if use_gpu else 'cpu')

    dy_model = dy_model_class.create_model(config)

    if model_init_path is not None:
        load_model(model_init_path, dy_model)

    # prof= profiler.Profiler(targets=[profiler.ProfilerTarget.CPU, profiler.ProfilerTarget.GPU], scheduler = [2,10], on_trace_ready=my_on_trace_ready, timer_only=False)
    # to do : add optimizer function
    optimizer = dy_model_class.create_optimizer(dy_model, config)

    logger.info("read data")
    train_dataloader = create_data_loader(config=config, place=place)

    last_epoch_id = config.get("last_epoch", -1)
    step_num = 0
    # prof.start()
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
            if batch_id == iter_num:
                break
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

        break


if __name__ == '__main__':
    model_list = read_list(sys.argv[1])
    model_id = int(sys.argv[2])
    model_name = model_list[model_id][0]
    config_yaml = get_abs_model(model_list[model_id][1])
    abs_dir = os.path.dirname(os.path.abspath(model_list[model_id][1]))
    iter_num = int(sys.argv[3])
    device_id = sys.argv[4]
    t1 = time.time()
    main(config_yaml, abs_dir, model_name, iter_num,device_id)
    t2 = time.time()
    print(model_name, t2 - t1)
