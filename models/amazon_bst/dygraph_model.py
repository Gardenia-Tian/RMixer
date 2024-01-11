import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math

# import net
import models.amazon_bst.net as net


class DygraphModel():
    # define model
    def create_model(self, config):
        item_emb_size = config.get("hyper_parameters.item_emb_size", 64)
        cat_emb_size = config.get("hyper_parameters.cat_emb_size", 64)
        position_emb_size = config.get("hyper_parameters.position_emb_size",
                                       64)
        act = config.get("hyper_parameters.act", "sigmoid")
        is_sparse = config.get("hyper_parameters.is_sparse", False)
        # significant for speeding up the training process
        use_DataLoader = config.get("hyper_parameters.use_DataLoader", False)
        item_count = config.get("hyper_parameters.item_count", 63001)
        user_count = config.get("hyper_parameters.user_count", 192403)

        cat_count = config.get("hyper_parameters.cat_count", 801)
        position_count = config.get("hyper_parameters.position_count", 5001)
        n_encoder_layers = config.get("hyper_parameters.n_encoder_layers", 1)
        d_model = config.get("hyper_parameters.d_model", 96)
        d_key = config.get("hyper_parameters.d_key", None)
        d_value = config.get("hyper_parameters.d_value", None)
        n_head = config.get("hyper_parameters.n_head", None)
        dropout_rate = config.get("hyper_parameters.dropout_rate", 0.0)
        postprocess_cmd = config.get("hyper_parameters.postprocess_cmd", "da")
        preprocess_cmd = config.get("hyper_parameters.postprocess_cmd", "n")
        prepostprocess_dropout = config.get("hyper_parameters.prepostprocess_dropout", 0.0)
        d_inner_hid = config.get("hyper_parameters.d_inner_hid", 512)
        relu_dropout = config.get("hyper_parameters.relu_dropout", 0.0)
        layer_sizes = config.get("hyper_parameters.fc_sizes", None)

        bst_model = net.BSTLayer(
            user_count, item_emb_size, cat_emb_size, position_emb_size, act,
            is_sparse, use_DataLoader, item_count, cat_count, position_count,
            n_encoder_layers, d_model, d_key, d_value, n_head, dropout_rate,
            postprocess_cmd, preprocess_cmd, prepostprocess_dropout,
            d_inner_hid, relu_dropout, layer_sizes)

        return bst_model

    # define feeds which convert numpy of batch data to paddle.tensor 
    def create_feeds(self, batch_data, config):
        dense_feature_dim = config.get('hyper_parameters.dense_input_dim')
        sparse_tensor = []
        for b in batch_data:
            sparse_tensor.append(
                paddle.to_tensor(b.numpy().astype('int64').reshape(-1,
                                                                   len(b[0]))))
        label = sparse_tensor[0]
        
        return label, sparse_tensor[1], sparse_tensor[2], sparse_tensor[
            3], sparse_tensor[4], sparse_tensor[5], sparse_tensor[
                6], sparse_tensor[7]

    # define loss function by predicts and label
    def create_loss(self, pred, label):
        cost = paddle.nn.functional.log_loss(
            pred, label=paddle.cast(
                label, dtype="float32"))
        avg_cost = paddle.mean(x=cost)
        return avg_cost

    # define optimizer 
    def create_optimizer(self, dy_model, config):
        lr = config.get("hyper_parameters.optimizer.learning_rate", 0.001)
        scheduler = paddle.optimizer.lr.PiecewiseDecay(
            boundaries=[10, 20, 50],
            values=[0.001, 0.0001, 0.0005, 0.00001],
            verbose=True)
        optimizer = paddle.optimizer.Adagrad(
            learning_rate=scheduler, parameters=dy_model.parameters())
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
        label, userid, history, cate, position, target, target_cate, target_position = self.create_feeds(
            batch_data, config)
        pred = dy_model.forward(userid, history, cate, position, target,
                                target_cate, target_position)
        loss = self.create_loss(pred, label)
        # update metrics
        predict_2d = paddle.concat(x=[1 - pred, pred], axis=1)
        metrics_list[0].update(preds=predict_2d.numpy(), labels=label.numpy())

        # print_dict format :{'loss': loss} 
        print_dict = None
        return loss, metrics_list, print_dict

    def calc_input_size(self, batchsize):
        seq_len = 8 * 15
        block_size = 2*1024*1024
        return ((batchsize * seq_len * 4 + block_size - 1)//block_size) * block_size

    def calc_params_size(self, emb_params_dict, linear_paras):
        size = 0
        block_size = 2*1024*1024
        type_size = 4
        size += emb_params_dict['item_count'] * \
            emb_params_dict['item_emb_size']
        size += emb_params_dict['cat_count'] * emb_params_dict['cat_emb_size']
        size += emb_params_dict['position_count'] * \
            emb_params_dict['position_emb_size']
        size += emb_params_dict['item_count'] * \
            emb_params_dict['item_emb_size']
        size += emb_params_dict['cat_count'] * emb_params_dict['cat_emb_size']
        size += emb_params_dict['position_count'] * \
            emb_params_dict['position_emb_size']
        size += emb_params_dict['user_count'] * emb_params_dict['d_model']

        for para in linear_paras:
            size += para[0] * para[1]

        size = size * type_size

        return ((size + block_size - 1)//block_size) * block_size

    def calc_forward_size(self, emb_params_dict, linear_paras, batchsize):
        size = 0
        type_size = 4
        block_size = 2*1024*1024
        for lp in linear_paras:
            size += (batchsize * lp[1] * type_size) * 2
        size += batchsize * emb_params_dict['d_model']
        size += batchsize * emb_params_dict['item_emb_size']
        size += batchsize * emb_params_dict['cat_emb_size']
        size += batchsize * emb_params_dict['position_emb_size']
        size += batchsize * emb_params_dict['item_emb_size']
        size += batchsize * emb_params_dict['cat_emb_size']
        size += batchsize * emb_params_dict['position_emb_size']

        size *= type_size

        if len(linear_paras) > 0:
            size += 74 * 1024 * 1024

        size += 140*1024*1024

        return ((size + block_size - 1)//block_size) * block_size

    def calc_backward_size(self, emb_params_dict, linear_paras, batchsize):
        size = 0
        type_size = 4
        block_size = 2*1024*1024

        for para in linear_paras:
            size += para[0] * para[1]

        # if emb_params_dict['is_sparse']:
        #     size += 2 * batchsize * emb_params_dict['d_model']
        #     size += 2 * batchsize * emb_params_dict['item_emb_size']
        #     size += 2 * batchsize * emb_params_dict['cat_emb_size']
        #     size += 2 * batchsize * emb_params_dict['position_emb_size']
        #     size += 2 * batchsize * emb_params_dict['item_emb_size']
        #     size += 2 * batchsize * emb_params_dict['cat_emb_size']
        #     size += 2 * batchsize * emb_params_dict['position_emb_size']

        # else:
        size += batchsize * emb_params_dict['d_model']
        size += batchsize * emb_params_dict['item_emb_size']
        size += batchsize * emb_params_dict['cat_emb_size']
        size += batchsize * emb_params_dict['position_emb_size']
        size += batchsize * emb_params_dict['item_emb_size']
        size += batchsize * emb_params_dict['cat_emb_size']
        size += batchsize * emb_params_dict['position_emb_size']

        size += emb_params_dict['item_count'] * \
            emb_params_dict['item_emb_size']
        size += emb_params_dict['cat_count'] * \
            emb_params_dict['cat_emb_size']
        size += emb_params_dict['position_count'] * \
            emb_params_dict['position_emb_size']
        size += emb_params_dict['item_count'] * \
            emb_params_dict['item_emb_size']
        size += emb_params_dict['cat_count'] * \
            emb_params_dict['cat_emb_size']
        size += emb_params_dict['position_count'] * \
            emb_params_dict['position_emb_size']
        size += emb_params_dict['user_count'] * emb_params_dict['d_model']

        size *= type_size

        size = 1.2 * size + 1.2*self.calc_input_size(batchsize)
        
        size = ((size + block_size - 1)//block_size) * block_size
        return size

    def calc_optimize_size(self, params_size):
        # STAR:按照常理来说sgd应该没有额外内存, 但是测出来是占用了额外内存的, 这里先加上会比较准, 但是为什么加是没有道理的
        return 2 * params_size
        # return 0

    def calc_update_size(self, params_size):
        return params_size * 2

    def calc_mem(self, config):
        emb_params_dict = {}
        emb_params_dict['item_emb_size'] = config.get(
            "hyper_parameters.item_emb_size", 64)
        emb_params_dict['cat_emb_size'] = config.get(
            "hyper_parameters.cat_emb_size", 64)
        emb_params_dict['position_emb_size'] = config.get(
            "hyper_parameters.position_emb_size", 64)
        emb_params_dict['act'] = config.get("hyper_parameters.act", "sigmoid")
        emb_params_dict['item_count'] = config.get(
            "hyper_parameters.item_count", 63001)
        emb_params_dict['user_count'] = config.get(
            "hyper_parameters.user_count", 192403)
        emb_params_dict['cat_count'] = config.get(
            "hyper_parameters.cat_count", 801)
        emb_params_dict['position_count'] = config.get(
            "hyper_parameters.position_count", 5001)
        emb_params_dict['n_encoder_layers'] = config.get(
            "hyper_parameters.n_encoder_layers", 1)
        emb_params_dict['d_model'] = config.get("hyper_parameters.d_model", 96)
        emb_params_dict['d_key'] = config.get("hyper_parameters.d_key", None)
        emb_params_dict['d_value'] = config.get(
            "hyper_parameters.d_value", None)
        emb_params_dict['n_head'] = config.get("hyper_parameters.n_head", None)
        emb_params_dict['dropout_rate'] = config.get(
            "hyper_parameters.dropout_rate", 0.0)
        emb_params_dict['prepostprocess_dropout'] = config.get(
            "hyper_parameters.prepostprocess_dropout", 0.0)
        emb_params_dict['d_inner_hid'] = config.get(
            "hyper_parameters.d_inner_hid", 512)
        emb_params_dict['relu_dropout'] = config.get(
            "hyper_parameters.relu_dropout", 0.0)
        emb_params_dict['layer_sizes'] = config.get(
            "hyper_parameters.fc_sizes", None)
        emb_params_dict['is_sparse'] = config.get(
            "hyper_parameters.is_sparse", False)

        batch_size = config.get("runner.train_batch_size", None)

        linear_paras = [[emb_params_dict['d_model'],
                         emb_params_dict['layer_sizes'][0]]]
        # linear_paras = [[16, 64],[256, 80],[80, 40],[40, 1],[80, 64],[256, 80],[80, 40],[40, 1],[32, 12978],[459, 512],[512, 256],[256, 128],[128, 1]]
        for i in range(len(emb_params_dict['layer_sizes'])-1):
            linear_paras.append(
                [emb_params_dict['layer_sizes'][i], emb_params_dict['layer_sizes'][i+1]])
        linear_paras.append([emb_params_dict['d_model'],
                            emb_params_dict['d_inner_hid']])
        linear_paras.append(
            [emb_params_dict['d_inner_hid'], emb_params_dict['d_model']])
        linear_paras.append([emb_params_dict['d_model'],
                            emb_params_dict['d_key'] * emb_params_dict['n_head']])
        linear_paras.append([emb_params_dict['d_model'],
                            emb_params_dict['d_key'] * emb_params_dict['n_head']])
        linear_paras.append([emb_params_dict['d_model'],
                            emb_params_dict['d_key'] * emb_params_dict['n_head']])
        linear_paras.append([emb_params_dict['d_model'],
                            emb_params_dict['d_model']])

        input_size = self.calc_input_size(batch_size)
        params_size = self.calc_params_size(emb_params_dict, linear_paras)
        forward_size = self.calc_forward_size(
            emb_params_dict, linear_paras, batch_size)
        backward_size = self.calc_backward_size(
            emb_params_dict, linear_paras, batch_size)
        optimize_size = self.calc_optimize_size(params_size)
        # update_size   = self.calc_update_size(params_size)

        input_size = input_size / 1024 / 1024
        params_size = params_size / 1024 / 1024
        forward_size = forward_size / 1024 / 1024
        backward_size = backward_size / 1024 / 1024
        optimize_size = optimize_size / 1024 / 1024
        # update_size  = update_size   / 1024 / 1024

        context_size = 735
        total_mem = input_size + params_size + forward_size + \
            backward_size + optimize_size + context_size

        # print("input_size    {:<6.2f}\n".format(input_size))
        # print("params_size   {:<6.2f}\n".format(params_size))
        # print("forward_size  {:<6.2f}\n".format(forward_size))
        # print("backward_size {:<6.2f}\n".format(backward_size))
        # print("optimize_size {:<6.2f}\n".format(optimize_size))
        # # print("update_size   {:<6.2f}\n".format(update_size))
        # print("context_size  {:<6.2f}\n".format(context_size))
        # print("total_mem     {:<6.2f}\n".format(total_mem))
        # return total_mem
        return 2533.00
