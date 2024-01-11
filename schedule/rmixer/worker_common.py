import importlib
import os
import paddle
import logging
logging.getLogger('google').setLevel(logging.ERROR)

from models.utils.utils_single import load_yaml
import models.rmixer_train


### Class
class ModelSummary():
    def __init__(self, model_name, use_gpu, iter_num):
        """ """
        self.task_name, self.config_file, self.identifier = model_name[0], model_name[1],  model_name[2]
        self.use_gpu = use_gpu
        self.iter_num = iter_num
        self.load_model()

    def execute(self): 
        models.rmixer_train.train(self.dy_model,self.dy_model_class,self.config,self.use_gpu,self.iter_num)
        
    def load_model(self):
        device_id = int(self.use_gpu.split(':')[-1])
        place = paddle.set_device(self.use_gpu if self.use_gpu.startswith('gpu') else 'cpu')
        self.config = load_yaml(self.config_file)
        self.config["config_abs_dir"],_ = os.path.split(self.config_file)
        dygraph_model = importlib.import_module('.' + self.task_name +'.dygraph_model',package="models")
        self.dy_model_class = dygraph_model.DygraphModel() 
        self.dy_model = self.dy_model_class.create_model(self.config)
