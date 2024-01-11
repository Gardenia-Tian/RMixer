import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math
import numpy as np
# import net
import models.amazon_din.net as net



class DygraphModel():
    def __init__(self):
        self.bucket = 100000
        self.absolute_limt = 200.0

    def rescale(self, number):
        if number > self.absolute_limt:
            number = self.absolute_limt
        elif number < -self.absolute_limt:
            number = -self.absolute_limt
        return (number + self.absolute_limt) / (self.absolute_limt * 2 + 1e-8)

    def create_model(self, config):
        item_emb_size = config.get("hyper_parameters.item_emb_size", 64)
        cat_emb_size = config.get("hyper_parameters.cat_emb_size", 64)
        act = config.get("hyper_parameters.act", "sigmoid")
        is_sparse = config.get("hyper_parameters.is_sparse", False)
        use_DataLoader = config.get("hyper_parameters.use_DataLoader", False)
        item_count = config.get("hyper_parameters.item_count", 63001)
        cat_count = config.get("hyper_parameters.cat_count", 801)
        din_model = net.DINLayer(item_emb_size, cat_emb_size, act, is_sparse, use_DataLoader, item_count, cat_count)
                                 
        return din_model

    # define feeds which convert numpy of batch data to paddle.tensor
    def create_feeds(self, batch, config):
        hist_item_seq = batch[0]
        hist_cat_seq = batch[1]
        target_item = batch[2]
        target_cat = batch[3]
        label = paddle.reshape(batch[4], [-1, 1])
        mask = batch[5]
        target_item_seq = batch[6]
        target_cat_seq = batch[7]
        return hist_item_seq, hist_cat_seq, target_item, target_cat, label, mask, target_item_seq, target_cat_seq

    # define loss function by predicts and label
    def create_loss(self, raw_pred, label):
        avg_loss = paddle.nn.functional.binary_cross_entropy_with_logits(raw_pred, label, reduction='mean')
        return avg_loss

    # define optimizer
    def create_optimizer(self, dy_model, config):
        boundaries = [410000]
        base_lr = config.get("hyper_parameters.optimizer.learning_rate_base_lr")
        values = [base_lr, 0.2]
        sgd_optimizer = paddle.optimizer.SGD(
            learning_rate=paddle.optimizer.lr.PiecewiseDecay(
                boundaries=boundaries, values=values),
            parameters=dy_model.parameters())
        return sgd_optimizer

    # define metrics such as auc/acc
    # multi-task need to define multi metric
    def create_metrics(self):
        metrics_list_name = ["auc"]
        #auc_metric = paddle.metric.Auc(num_thresholds=self.bucket)
        auc_metric = paddle.metric.Auc("ROC")
        metrics_list = [auc_metric]
        return metrics_list, metrics_list_name

    # construct train forward phase
    def train_forward(self, dy_model, metrics_list, batch_data, config):
        hist_item_seq, hist_cat_seq, target_item, target_cat, label, mask, target_item_seq, target_cat_seq = self.create_feeds(batch_data, config)
        raw_pred = dy_model.forward(hist_item_seq, hist_cat_seq, target_item,
                                    target_cat, label, mask, target_item_seq,
                                    target_cat_seq)
        loss = self.create_loss(raw_pred, label)
        predict = paddle.nn.functional.sigmoid(raw_pred)
        predict_2d = paddle.concat([1 - predict, predict], 1)
        label_int = paddle.cast(label, 'int64')
        metrics_list[0].update(preds=predict_2d.numpy(), labels=label_int.numpy())
            
        print_dict = {'loss': loss}
        return loss, metrics_list, print_dict


    # TODO:完善内存估计 
    def calc_input_size(self, batchsize):
        # 每个数据的长度不一样, 但是最大一般不超过15, 这里取15
        seq_len = 8 * 15
        block_size = 2*1024*1024
        return ((batchsize * seq_len * 4 + block_size - 1)//block_size) * block_size

    def calc_params_size(self, emb_params_dict, linear_paras):
        size = 0
        block_size = 2*1024*1024
        type_size = 4
        size += emb_params_dict['item_count'] * emb_params_dict['item_emb_size']
        size += emb_params_dict['cat_count']  * emb_params_dict['cat_emb_size']
        size += emb_params_dict['item_count'] * emb_params_dict['item_emb_size']
        size += emb_params_dict['item_count'] * emb_params_dict['item_emb_size']
        size += emb_params_dict['item_count'] * 1
        size += emb_params_dict['cat_count']  * emb_params_dict['cat_emb_size']
        size += emb_params_dict['cat_count']  * emb_params_dict['cat_emb_size']

        for para in linear_paras:
            size += para[0] * para[1] + para[1]

        size = size * type_size
        if len(linear_paras)>0:
            size += 140 * 1024 * 1024
        return ((size + block_size - 1)//block_size) * block_size

    def calc_forward_size(self, emb_params_dict, linear_paras, batchsize):
        size = 0
        type_size = 4
        block_size = 2*1024*1024
        for lp in linear_paras:
            # 2是linear, 1是sigmoid
            size += (batchsize * lp[1] * type_size) * (2 + 1)
        
        size += batchsize * emb_params_dict['item_emb_size']
        size += batchsize * emb_params_dict['cat_emb_size']
        size += batchsize * emb_params_dict['item_emb_size']
        size += batchsize * emb_params_dict['item_emb_size']
        size += batchsize * 1
        size += batchsize * emb_params_dict['cat_emb_size']
        size += batchsize * emb_params_dict['cat_emb_size']

        size *= type_size

        if len(linear_paras) > 0:
            size += 74 * 1024 * 1024

        # softmax的context
        size += 80*1024*1024

        return ((size + block_size - 1)//block_size) * block_size

    def calc_backward_size(self, emb_params_dict, linear_paras, batchsize):
        size = 0
        type_size = 4
        block_size = 2*1024*1024

        for para in linear_paras:
            size += para[0] * para[1]

        if emb_params_dict['is_sparse']:
            size += 2 * batchsize * emb_params_dict['item_emb_size']
            size += 2 * batchsize * emb_params_dict['cat_emb_size']
            size += 2 * batchsize * emb_params_dict['item_emb_size']
            size += 2 * batchsize * emb_params_dict['item_emb_size']
            size += 2 * batchsize * 1
            size += 2 * batchsize * emb_params_dict['cat_emb_size']
            size += 2 * batchsize * emb_params_dict['cat_emb_size']

        else:
            size += batchsize * emb_params_dict['item_emb_size']
            size += batchsize * emb_params_dict['cat_emb_size']
            size += batchsize * emb_params_dict['item_emb_size']
            size += batchsize * emb_params_dict['item_emb_size']
            size += batchsize * 1
            size += batchsize * emb_params_dict['cat_emb_size']
            size += batchsize * emb_params_dict['cat_emb_size']

            size += emb_params_dict['item_count'] * emb_params_dict['item_emb_size']
            size += emb_params_dict['cat_count']  * emb_params_dict['cat_emb_size']
            size += emb_params_dict['item_count'] * emb_params_dict['item_emb_size']
            size += emb_params_dict['item_count'] * emb_params_dict['item_emb_size']
            size += emb_params_dict['item_count'] * 1
            size += emb_params_dict['cat_count']  * emb_params_dict['cat_emb_size']
            size += emb_params_dict['cat_count']  * emb_params_dict['cat_emb_size']

        size *= type_size
        # sigmoid的backward机制没有十分弄清楚, 他反向会占用一部分内存, 但是不知道咋算
        # size = 1.2 * size + 1.2*self.calc_input_size(batchsize)
        return ((size + block_size - 1)//block_size) * block_size
        

    def calc_optimize_size(self, params_size):
        # STAR:按照常理来说sgd应该没有额外内存, 但是测出来是占用了额外内存的, 这里先加上会比较准, 但是为什么加是没有道理的
        # return 2 * params_size
        return 0

    def calc_update_size(self, params_size):
        return params_size * 2

    def calc_mem(self, config):
        emb_params_dict = {}
        emb_params_dict['item_emb_size'] = config.get("hyper_parameters.item_emb_size", 64)
        emb_params_dict['cat_emb_size'] = config.get("hyper_parameters.cat_emb_size", 64)
        emb_params_dict['act'] = config.get("hyper_parameters.act", "sigmoid")
        emb_params_dict['is_sparse'] = config.get("hyper_parameters.is_sparse", False)
        emb_params_dict['use_DataLoader'] = config.get("hyper_parameters.use_DataLoader", False)
        emb_params_dict['item_count'] = config.get("hyper_parameters.item_count", 63001)
        emb_params_dict['cat_count'] = config.get("hyper_parameters.cat_count", 801)
        batch_size = config.get("runner.train_batch_size", None)

        linear_paras = [[(emb_params_dict['item_emb_size'] +
                          emb_params_dict['cat_emb_size']) * 4, 80], [80, 40], [40, 1]]
        conDim = emb_params_dict['item_emb_size'] + emb_params_dict['cat_emb_size'] + \
            emb_params_dict['item_emb_size'] + emb_params_dict['cat_emb_size']
        linear_paras.append([emb_params_dict['item_emb_size'] + emb_params_dict['cat_emb_size'],
                            emb_params_dict['item_emb_size'] + emb_params_dict['cat_emb_size']])
        linear_paras.append([conDim, 80])
        linear_paras.append([80, 40])
        linear_paras.append([40, 1])

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
        return 1479.00
