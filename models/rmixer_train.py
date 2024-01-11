import paddle
import time
import warnings
warnings.filterwarnings("ignore")

from models.utils.utils_single import load_yaml, load_dy_model_class, get_abs_model, create_data_loader
from paddle.io import DistributedBatchSampler, DataLoader


def train(dy_model, dy_model_class, config,use_gpu, iter_num):
    
    # tools.vars
    use_auc = config.get("runner.use_auc", False)
    train_data_dir = config.get("runner.train_data_dir", None)
    epochs = config.get("runner.epochs", None)
    print_interval = config.get("runner.print_interval", None)
    train_batch_size = config.get("runner.train_batch_size", None)
    model_save_path = config.get("runner.model_save_path", "model_output")
    model_init_path = config.get("runner.model_init_path", None)
    seed = config.get("runner.seed", 12345)
    paddle.seed(seed)
    place = paddle.set_device(use_gpu if use_gpu.startswith("gpu") else 'cpu')
    optimizer = dy_model_class.create_optimizer(dy_model, config)

    train_dataloader = create_data_loader(config=config, place=place)

    last_epoch_id = config.get("last_epoch", -1)
    step_num = 0
    # prof.start()
    for epoch_id in range(last_epoch_id + 1, epochs):
        # set train mode
        dy_model.train()
        metric_list, metric_list_name = dy_model_class.create_metrics()
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
                train_reader_cost = 0.0
                train_run_cost = 0.0
                total_samples = 0
            reader_start = time.time()
            step_num = step_num + 1 
        break
