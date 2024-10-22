import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math

# import net
import models.criteo_widedeep.net as net



class DygraphModel():
    # define model
    def create_model(self, config):
        sparse_feature_number = config.get("hyper_parameters.sparse_feature_number")
        sparse_feature_dim = config.get("hyper_parameters.sparse_feature_dim")
        fc_sizes = config.get("hyper_parameters.fc_sizes")
        sparse_fea_num = config.get('hyper_parameters.sparse_fea_num')
        dense_feature_dim = config.get('hyper_parameters.dense_input_dim')
        sparse_input_slot = config.get('hyper_parameters.sparse_inputs_slots')

        wide_deep_model = net.WideDeepLayer(sparse_feature_number, sparse_feature_dim, dense_feature_dim,sparse_input_slot - 1, fc_sizes)       
        return wide_deep_model

    # define feeds which convert numpy of batch data to paddle.tensor 
    def create_feeds(self, batch_data, config):
        dense_feature_dim = config.get('hyper_parameters.dense_input_dim')
        sparse_tensor = []
        for b in batch_data[:-1]:
            sparse_tensor.append(paddle.to_tensor(b.numpy().astype('int64').reshape(-1, 1)))
        dense_tensor = paddle.to_tensor(batch_data[-1].numpy().astype('float32').reshape(-1, dense_feature_dim))
            
        label = sparse_tensor[0]

        return label, sparse_tensor[1:], dense_tensor

    # define loss function by predicts and label
    def create_loss(self, pred, label):
        cost = paddle.nn.functional.log_loss(input=pred, label=paddle.cast(label, dtype="float32"))
        avg_cost = paddle.mean(x=cost)
        return avg_cost

    # define optimizer 
    def create_optimizer(self, dy_model, config):
        lr = config.get("hyper_parameters.optimizer.learning_rate", 0.001)
        optimizer = paddle.optimizer.Adam(learning_rate=lr, parameters=dy_model.parameters())
        return optimizer

    # define metrics such as auc/acc
    # multi-task need to define multi metric
    def create_metrics(self):
        metrics_list_name = ["auc"]
        auc_metric = paddle.metric.Auc("ROC")
        metrics_list = [auc_metric]
        return metrics_list, metrics_list_name

    # construct train forward phase  
    def train_forward(self, dy_model, metrics_list, batch_data, config):
        label, sparse_tensor, dense_tensor = self.create_feeds(batch_data, config)
                                                              
        pred = dy_model.forward(sparse_tensor, dense_tensor)
        loss = self.create_loss(pred, label)
        # update metrics
        predict_2d = paddle.concat(x=[1 - pred, pred], axis=1)
        metrics_list[0].update(preds=predict_2d.numpy(), labels=label.numpy())

        print_dict = {'loss': loss}
        #print_dict = None
        return loss, metrics_list, print_dict

    def infer_forward(self, dy_model, metrics_list, batch_data, config):
        label, sparse_tensor, dense_tensor = self.create_feeds(batch_data,
                                                               config)

        pred = dy_model.forward(sparse_tensor, dense_tensor)
        # update metrics
        predict_2d = paddle.concat(x=[1 - pred, pred], axis=1)
        metrics_list[0].update(preds=predict_2d.numpy(), labels=label.numpy())

        return metrics_list, None

    # TODO:完善内存估计 
    def calc_mem(self,config):
        return 1623.00