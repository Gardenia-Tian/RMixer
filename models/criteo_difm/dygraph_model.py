import paddle
# import net
import models.criteo_difm.net as net


class DygraphModel():
    # define model
    def create_model(self, config):
        sparse_field_num = config.get("hyper_parameters.sparse_field_num")
        sparse_feature_num = config.get("hyper_parameters.sparse_feature_num")
        sparse_feature_dim = config.get("hyper_parameters.sparse_feature_dim")
        dense_feature_dim = config.get("hyper_parameters.dense_feature_dim")
        fen_layers_size = config.get("hyper_parameters.fen_layers_size")
        dense_layers_size = config.get("hyper_parameters.dense_layers_size")
        att_factor_dim = config.get("hyper_parameters.att_factor_dim")
        att_head_num = config.get("hyper_parameters.att_head_num")

        # ifm_model = net.IFM(sparse_field_num=sparse_field_num,
        #                     sparse_feature_num=sparse_feature_num,
        #                     sparse_feature_dim=sparse_feature_dim,
        #                     dense_feature_dim=dense_feature_dim,
        #                     fen_layers_size=fen_layers_size,
        #                     dense_layers_size=dense_layers_size)
        #
        # return ifm_model

        difm_model = net.DIFM(
            sparse_field_num=sparse_field_num,
            sparse_feature_num=sparse_feature_num,
            sparse_feature_dim=sparse_feature_dim,
            dense_feature_dim=dense_feature_dim,
            fen_layers_size=fen_layers_size,
            dense_layers_size=dense_layers_size,
            att_factor_dim=att_factor_dim,
            att_head_num=att_head_num)
        return difm_model

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
        label, sparse_tensor, dense_tensor = self.create_feeds(batch_data,config)
                                                               
        pred = dy_model.forward(sparse_tensor, dense_tensor)
        loss = self.create_loss(pred, label)
        # update metrics
        predict_2d = paddle.concat(x=[1 - pred, pred], axis=1)
        metrics_list[0].update(preds=predict_2d.numpy(), labels=label.numpy())

        # print_dict format :{'loss': loss}
        print_dict = {"loss": loss}
        return loss, metrics_list, print_dict


    # TODO:完善内存估计 
    def calc_mem(self,config):
        batch_size = config.get("runner.train_batch_size")
        if batch_size == 512: 
            return 1847.0
        elif batch_size == 2000:
            return 3823.00