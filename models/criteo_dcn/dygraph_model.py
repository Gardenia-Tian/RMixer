import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math
# import net
import models.criteo_dcn.net as net



class DygraphModel():
    # define model
    def create_model(self, config):
        sparse_feature_number = config.get("hyper_parameters.sparse_feature_number")
        sparse_feature_dim = config.get("hyper_parameters.sparse_feature_dim")

        fc_sizes = config.get("hyper_parameters.fc_sizes")
        dense_feature_dim = config.get('hyper_parameters.dense_input_dim')
        sparse_input_slot = config.get('hyper_parameters.sparse_inputs_slots')
        cross_num = config.get("hyper_parameters.cross_num")
        l2_reg_cross = config.get("hyper_parameters.l2_reg_cross", None)
        clip_by_norm = config.get("hyper_parameters.clip_by_norm", None)
        is_sparse = config.get("hyper_parameters.is_sparse", None)
        dnn_use_bn = config.get("hyper_parameters.dnn_use_bn", None)
        deepcro_model = net.DeepCroLayer(
            sparse_feature_number, sparse_feature_dim, dense_feature_dim,
            sparse_input_slot - 1, fc_sizes, cross_num, clip_by_norm,
            l2_reg_cross, is_sparse
        )  # print("----dygraphModel---deepcro_model--", deepcro_model)
        return deepcro_model

    # define feeds which convert numpy of batch data to paddle.tensor 
    def create_feeds(self, batch_data, config):
        # print("----batch_data", batch_data)
        dense_feature_dim = config.get('hyper_parameters.dense_input_dim')
        sparse_tensor = []
        for b in batch_data[:-1]:
            sparse_tensor.append(
                paddle.to_tensor(b.numpy().astype('int64').reshape(-1, 1)))
        dense_tensor = paddle.to_tensor(batch_data[-1].numpy().astype('float32').reshape(-1, dense_feature_dim))
        label = sparse_tensor[0]
        # print("-----dygraph-----label:----",label.shape)
        # print("-----dygraph-----sparse_tensor[1:]:----", sparse_tensor[1:])
        # print("-----dygraph-----dense_tensor:----", dense_tensor.shape)
        return label, sparse_tensor[1:], dense_tensor

    # define loss function by predicts and label
    def create_loss(self, pred, label):
        # print("---dygraph----pred, label:",pred, label)
        cost = paddle.nn.functional.log_loss(
            input=pred, label=paddle.cast(
                label, dtype="float32"))
        avg_cost = paddle.mean(x=cost)
        # add l2_loss.............
        # print("---dygraph-----cost,avg_cost----",cost,avg_cost)
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
        # print("---dygraph-----label, sparse_tensor, dense_tensor",label, sparse_tensor, dense_tensor)
        pred, l2_loss = dy_model.forward(sparse_tensor, dense_tensor)
        loss = self.create_loss(pred, label) + l2_loss
        # update metrics
        predict_2d = paddle.concat(x=[1 - pred, pred], axis=1)
        # print("---dygraph----pred,loss,predict_2d---",pred,loss,predict_2d)
        # print("---dygraph----metrics_list",metrics_list)
        metrics_list[0].update(preds=predict_2d.numpy(), labels=label.numpy())

        # print_dict format :{'loss': loss} 
        print_dict = None
        return loss, metrics_list, print_dict

    # TODO:完善内存估计 
    def calc_input_size(self,emb_params_dict, batchsize):
        seq_len = emb_params_dict['sparse_input_slot'] - 1
        type_size = 4
        block_size = 2*1024*1024
        return ((batchsize * seq_len * type_size + block_size - 1 )//block_size) * block_size

    def calc_params_size(self,emb_params_dict,linear_paras):
        size = 0
        block_size = 2*1024*1024
        type_size = 4
        # embedding params size
        size += emb_params_dict['sparse_feature_number'] * emb_params_dict['sparse_feature_dim']
        # cross layer size
        size += emb_params_dict['dense_feature_dim'] + (emb_params_dict['sparse_input_slot'] - 1) * emb_params_dict['sparse_feature_dim']
        
        # linear params size
        for param in linear_paras:
            size += param[0] * param[1] + param[1]
        
        size *= type_size
        
        return ((size + block_size - 1)//block_size) * block_size 
    
    def calc_forward_size(self,emb_params_dict,linear_paras,batchsize):
        size = 0
        type_size = 4
        block_size = 2*1024*1024
        input_size = (emb_params_dict['sparse_input_slot'] - 1) * emb_params_dict['sparse_feature_dim'] + emb_params_dict['dense_feature_dim']
        # linear
        for lp in linear_paras:
            size += (batchsize * lp[1] * type_size) * 2
        
        # embedding
      
        size += (emb_params_dict['sparse_input_slot'] - 1) * emb_params_dict['sparse_feature_dim'] * batchsize
        # reshape
        size += (emb_params_dict['sparse_input_slot'] - 1) * emb_params_dict['sparse_feature_dim'] * batchsize
        # concat
        size += (input_size) * batchsize
        

        # cross net
        size += emb_params_dict['cross_num'] * (input_size * 4 * batchsize)

        # concat
        size += linear_paras[-1][0]

        size *= type_size
        if len(linear_paras)>0:
            size += 140 * 1024 * 1024

        return ((size + block_size - 1)//block_size) * block_size 

    def calc_backward_size(self,emb_params_dict,linear_paras,batchsize):
        size = 0
        type_size = 4

        block_size = 2*1024*1024
        # if emb_params_dict['is_sparse']==True:
        size += 2 * emb_params_dict['sparse_feature_dim'] * emb_params_dict['sparse_feature_dim'] * batchsize
        # else:
            # size += emb_params_dict['sparse_feature_dim'] * emb_params_dict['sparse_feature_dim'] * batchsize + emb_params_dict['sparse_feature_number'] * emb_params_dict['sparse_feature_dim']
        
        # linear
        for para in linear_paras:
            size += para[0] * para[1]
        
        size *= type_size
        
        return ((size + block_size - 1)//block_size) * block_size 

    def calc_optimize_size(self,params_size):
        # 由Adam优化器决定
        # block_size = 2*1024*1024
        # type_size = 4
        return params_size * 2
        
    def calc_update_size(self,params_size):
        return params_size * 2    

    def calc_mem(self,config):

        emb_params_dict = {}
        emb_params_dict['sparse_feature_number'] = config.get("hyper_parameters.sparse_feature_number")
        emb_params_dict['sparse_feature_dim'] = config.get("hyper_parameters.sparse_feature_dim")
        emb_params_dict['fc_sizes'] = config.get("hyper_parameters.fc_sizes")
        emb_params_dict['dense_feature_dim'] = config.get('hyper_parameters.dense_input_dim')
        emb_params_dict['sparse_input_slot'] = config.get('hyper_parameters.sparse_inputs_slots')
        emb_params_dict['cross_num'] = config.get("hyper_parameters.cross_num")
        emb_params_dict['l2_reg_cross'] = config.get("hyper_parameters.l2_reg_cross", None)
        emb_params_dict['clip_by_norm'] = config.get("hyper_parameters.clip_by_norm", None)
        emb_params_dict['is_sparse'] = config.get("hyper_parameters.is_sparse", None)
        
        
        batch_size = config.get("runner.train_batch_size", None)

        linear_paras = [[ (emb_params_dict['sparse_input_slot'] - 1)* emb_params_dict['sparse_feature_dim'] + emb_params_dict['dense_feature_dim'],emb_params_dict['fc_sizes'][0]]]
        for i in range(len(emb_params_dict['fc_sizes'])-1):
            linear_paras.append([emb_params_dict['fc_sizes'][i],emb_params_dict['fc_sizes'][i+1]])
        linear_paras.append([emb_params_dict['fc_sizes'][-1] + (emb_params_dict['sparse_input_slot'] - 1) * emb_params_dict['dense_feature_dim'],1])
        
        input_size    = self.calc_input_size(emb_params_dict, batch_size) 
        params_size   = self.calc_params_size(emb_params_dict,linear_paras)
        forward_size  = self.calc_forward_size(emb_params_dict,linear_paras,batch_size)
        backward_size = self.calc_backward_size(emb_params_dict,linear_paras,batch_size)
        optimize_size = self.calc_optimize_size(params_size)
        # update_size   = self.calc_update_size(params_size)

        input_size = input_size / 1024 /1024
        params_size  = params_size   / 1024 / 1024
        forward_size = forward_size  / 1024 / 1024
        backward_size= backward_size / 1024 / 1024
        optimize_size= optimize_size / 1024 / 1024
        # update_size  = update_size   / 1024 / 1024


        context_size = 735
        total_mem = input_size + params_size + forward_size + backward_size + optimize_size  + context_size
        total_mem += emb_params_dict['sparse_feature_number'] * emb_params_dict['sparse_feature_dim'] * 4 / 1024 / 1024
        
        # print("input_size    {:<6.2f}\n".format(input_size))
        # print("params_size   {:<6.2f}\n".format(params_size))
        # print("forward_size  {:<6.2f}\n".format(forward_size))
        # print("backward_size {:<6.2f}\n".format(backward_size))
        # print("optimize_size {:<6.2f}\n".format(optimize_size))
        # # print("update_size   {:<6.2f}\n".format(update_size))
        # print("context_size  {:<6.2f}\n".format(context_size))
        # print("total_mem     {:<6.2f}\n".format(total_mem))
        # return total_mem
        return 1373
    
