import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math

# import net
import models.alidisplay_dmr.net as net



class DygraphModel():
    # define model
    def create_model(self, config):
        user_size = config.get("hyper_parameters.user_size")
        cms_segid_size = config.get("hyper_parameters.cms_segid_size")
        cms_group_id_size = config.get("hyper_parameters.cms_group_id_size")
        final_gender_code_size = config.get(
            "hyper_parameters.final_gender_code_size")
        age_level_size = config.get("hyper_parameters.age_level_size")
        pvalue_level_size = config.get("hyper_parameters.pvalue_level_size")
        shopping_level_size = config.get(
            "hyper_parameters.shopping_level_size")
        occupation_size = config.get("hyper_parameters.occupation_size")
        new_user_class_level_size = config.get(
            "hyper_parameters.new_user_class_level_size")
        adgroup_id_size = config.get("hyper_parameters.adgroup_id_size")
        cate_size = config.get("hyper_parameters.cate_size")
        campaign_id_size = config.get("hyper_parameters.campaign_id_size")
        customer_size = config.get("hyper_parameters.customer_size")
        brand_size = config.get("hyper_parameters.brand_size")
        btag_size = config.get("hyper_parameters.btag_size")
        pid_size = config.get("hyper_parameters.pid_size")
        main_embedding_size = config.get(
            "hyper_parameters.main_embedding_size")
        other_embedding_size = config.get(
            "hyper_parameters.other_embedding_size")

        dmr_model = net.DMRLayer(
            user_size, cms_segid_size, cms_group_id_size,
            final_gender_code_size, age_level_size, pvalue_level_size,
            shopping_level_size, occupation_size, new_user_class_level_size,
            adgroup_id_size, cate_size, campaign_id_size, customer_size,
            brand_size, btag_size, pid_size, main_embedding_size,
            other_embedding_size)

        return dmr_model

    # define feeds which convert numpy of batch data to paddle.tensor
    def create_feeds(self, batch_data, config):
        b = batch_data[0]
        sparse_tensor = b.astype('int64')
        dense_tensor = paddle.to_tensor(b[:, 264].numpy().astype('float32')
                                        .reshape(-1, 1))
        label = sparse_tensor[:, -1].reshape([-1, 1])
        # print(sparse_tensor.shape)
        # print(dense_tensor.shape)
        return label, [sparse_tensor, dense_tensor]

    # define optimizer
    def create_optimizer(self, dy_model, config):
        lr = config.get("hyper_parameters.optimizer.learning_rate", 0.001)
        optimizer = paddle.optimizer.Adam(
            learning_rate=lr, parameters=dy_model.parameters())
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
        label, input_tensor = self.create_feeds(batch_data, config)

        pred, loss = dy_model.forward(input_tensor, False)
        # update metrics
        predict_2d = paddle.concat(x=[1 - pred, pred], axis=1)
        metrics_list[0].update(preds=predict_2d.numpy(), labels=label.numpy())

        print_dict = {'loss': loss}
        # print_dict = None
        return loss, metrics_list, print_dict
        
    def calc_input_size(self,batchsize):
        # TODO:这里目前是根据数据集写死的, 如果数据集变化这里要改变, 要么在self记录所有的数据长度, 要么从config读入
        seq_len = 268
        type_size = 4
        block_size = 2*1024*1024
        return ((batchsize * seq_len * type_size + block_size - 1 )//block_size) * block_size

    def calc_params_size(self,emb_params_dict,linear_paras,att_paras):
        size = 0
        block_size = 2*1024*1024
        type_size = 4
        
        # embedding params size
        size += emb_params_dict['user_size'] * emb_params_dict['main_embedding_size']
        size += emb_params_dict['adgroup_id_size'] * emb_params_dict['main_embedding_size']
        size += emb_params_dict['cate_size'] * emb_params_dict['main_embedding_size']
        size += emb_params_dict['brand_size'] * emb_params_dict['main_embedding_size']
        size += emb_params_dict['btag_size'] * emb_params_dict['other_embedding_size']
        size += emb_params_dict['btag_size'] * emb_params_dict['other_embedding_size']
        size += emb_params_dict['campaign_id_size'] * emb_params_dict['main_embedding_size']
        size += emb_params_dict['customer_size'] * emb_params_dict['main_embedding_size']
        size += emb_params_dict['cms_segid_size'] * emb_params_dict['other_embedding_size']
        size += emb_params_dict['cms_group_id_size'] * emb_params_dict['other_embedding_size']
        size += emb_params_dict['final_gender_code_size'] * emb_params_dict['other_embedding_size']
        size += emb_params_dict['age_level_size'] * emb_params_dict['other_embedding_size']
        size += emb_params_dict['pvalue_level_size'] * emb_params_dict['other_embedding_size']
        size += emb_params_dict['shopping_level_size'] * emb_params_dict['other_embedding_size']
        size += emb_params_dict['occupation_size'] * emb_params_dict['other_embedding_size']
        size += emb_params_dict['new_user_class_level_size'] * emb_params_dict['other_embedding_size']
        size += emb_params_dict['pid_size'] * emb_params_dict['other_embedding_size']
        size += emb_params_dict['history_length'] * emb_params_dict['other_embedding_size']
        size += emb_params_dict['history_length'] * emb_params_dict['other_embedding_size']
        size += emb_params_dict['cate_size'] * emb_params_dict['main_embedding_size']
        
        # linear params size
        for param in linear_paras:
            size += param[0] * param[1] + param[1]

        for param in att_paras:
            size += param[0] * param[1] + param[1]
        
        size *= type_size
        return ((size + block_size - 1)//block_size) * block_size 
    
    def calc_forward_size(self,emb_params_dict,linear_paras,att_paras,batchsize):
        size = 0
        type_size = 4
        block_size = 2*1024*1024
        for lp in linear_paras:
            size += (batchsize * lp[1] ) * 2 
        
        for ap in att_paras:
            size += (batchsize * ap[1] * emb_params_dict['history_length']) * 2

        # 因为有concat, 所以要*2
        size += 2 * batchsize * emb_params_dict['main_embedding_size'] * 5 
        size += 2 * batchsize * emb_params_dict['main_embedding_size'] * emb_params_dict['history_length'] * 2
        
        size += 2 * batchsize * emb_params_dict['other_embedding_size'] * 9 
        size += 2 * batchsize * emb_params_dict['other_embedding_size'] * emb_params_dict['history_length'] * 4
        
        size *= type_size
        # linear context
        if len(linear_paras)>0:
            size += 74 * 1024 * 1024

        # softmax context
        size += 80*1024*1024

        return ((size + block_size - 1)//block_size) * block_size *1.5

    def calc_backward_size(self,emb_params_dict,linear_paras,att_paras,batchsize):
        size = 0
        type_size = 4

        block_size = 2*1024*1024

        # sparse + 2 * outputsize
        size += batchsize *emb_params_dict['main_embedding_size'] * 2 * 5
        size += batchsize * emb_params_dict['history_length'] *emb_params_dict['main_embedding_size'] * 2 * 2
        
        # non sparse + params + outsize
        # btag_embeddings_var
        size += emb_params_dict['btag_size'] * emb_params_dict['other_embedding_size'] + batchsize * emb_params_dict['other_embedding_size'] * emb_params_dict['history_length']
        # dm_btag_embeddings_var
        size += emb_params_dict['btag_size'] * emb_params_dict['other_embedding_size'] + batchsize * emb_params_dict['other_embedding_size'] * emb_params_dict['history_length']
        # cms_segid_embeddings_var
        size += emb_params_dict['cms_segid_size'] * emb_params_dict['other_embedding_size'] + batchsize * emb_params_dict['other_embedding_size']
        # cms_group_id_embeddings_var
        size += emb_params_dict['cms_group_id_size'] * emb_params_dict['other_embedding_size'] + batchsize * emb_params_dict['other_embedding_size']
        # final_gender_code_embeddings_var
        size += emb_params_dict['final_gender_code_size'] * emb_params_dict['other_embedding_size'] + batchsize * emb_params_dict['other_embedding_size']
        # age_level_embeddings_var
        size += emb_params_dict['age_level_size'] * emb_params_dict['other_embedding_size'] + batchsize * emb_params_dict['other_embedding_size']
        # pvalue_level_embeddings_var
        size += emb_params_dict['pvalue_level_size'] * emb_params_dict['other_embedding_size'] + batchsize * emb_params_dict['other_embedding_size']
        # shopping_level_embeddings_var
        size += emb_params_dict['shopping_level_size'] * emb_params_dict['other_embedding_size'] + batchsize * emb_params_dict['other_embedding_size']
        # occupation_embeddings_var
        size += emb_params_dict['occupation_size'] * emb_params_dict['other_embedding_size'] + batchsize * emb_params_dict['other_embedding_size']
        # new_user_class_level_embeddings_var
        size += emb_params_dict['new_user_class_level_size'] * emb_params_dict['other_embedding_size'] + batchsize * emb_params_dict['other_embedding_size']
        # pid_embeddings_var
        size += emb_params_dict['pid_size'] * emb_params_dict['other_embedding_size'] + batchsize * emb_params_dict['other_embedding_size']
        # position_embeddings_var
        size += emb_params_dict['history_length'] * emb_params_dict['other_embedding_size'] + batchsize * emb_params_dict['other_embedding_size']* emb_params_dict['history_length']
        # dm_position_embeddings_var
        size += emb_params_dict['history_length'] * emb_params_dict['other_embedding_size'] + batchsize * emb_params_dict['other_embedding_size']* emb_params_dict['history_length']
        # dm_item_vectors_var
        size += emb_params_dict['cate_size'] * emb_params_dict['main_embedding_size'] + batchsize * emb_params_dict['main_embedding_size']

        # linear
        for para in linear_paras:
            size += para[0] * para[1]

        for para in att_paras:
            size += para[0] * para[1]
        
        size *= type_size
        size = ((size + block_size - 1)//block_size) * block_size
        return size

    def calc_optimize_size(self,params_size):
        # 由Adam优化器决定
        # block_size = 2*1024*1024
        # type_size = 4
        return params_size * 2
        
    def calc_update_size(self,params_size):
        return params_size * 2    

    def calc_mem(self,config):

        emb_params_dict = {}
        emb_params_dict['user_size'] = config.get("hyper_parameters.user_size")
        emb_params_dict['cms_segid_size'] = config.get("hyper_parameters.cms_segid_size")
        emb_params_dict['cms_group_id_size'] = config.get("hyper_parameters.cms_group_id_size")
        emb_params_dict['final_gender_code_size'] = config.get("hyper_parameters.final_gender_code_size")
        emb_params_dict['age_level_size'] = config.get("hyper_parameters.age_level_size")
        emb_params_dict['pvalue_level_size'] = config.get("hyper_parameters.pvalue_level_size")
        emb_params_dict['shopping_level_size'] = config.get("hyper_parameters.shopping_level_size")
        emb_params_dict['occupation_size'] = config.get("hyper_parameters.occupation_size")
        emb_params_dict['new_user_class_level_size'] = config.get("hyper_parameters.new_user_class_level_size")
        emb_params_dict['adgroup_id_size'] = config.get("hyper_parameters.adgroup_id_size")
        emb_params_dict['cate_size'] = config.get("hyper_parameters.cate_size")
        emb_params_dict['campaign_id_size'] = config.get("hyper_parameters.campaign_id_size")
        emb_params_dict['customer_size'] = config.get("hyper_parameters.customer_size")
        emb_params_dict['brand_size'] = config.get("hyper_parameters.brand_size")
        emb_params_dict['btag_size'] = config.get("hyper_parameters.btag_size")
        emb_params_dict['pid_size'] = config.get("hyper_parameters.pid_size")
        emb_params_dict['main_embedding_size'] = config.get("hyper_parameters.main_embedding_size")
        emb_params_dict['other_embedding_size'] = config.get("hyper_parameters.other_embedding_size")
        # TODO:这里也是和数据集相关的, 如果换数据集要改
        emb_params_dict['history_length'] = 50
        batch_size = config.get("runner.train_batch_size", None)
        linear_paras = [[16, 64],[32, 12978],[459, 512],[512, 256],[256, 128],[128, 1]]
        att_paras = [[256, 80],[80, 40],[40, 1],[64, 32],[80, 64],[256, 80],[80, 40],[40, 1],]

        input_size    = self.calc_input_size(batch_size) 
        params_size   = self.calc_params_size(emb_params_dict,linear_paras,att_paras)
        forward_size  = self.calc_forward_size(emb_params_dict,linear_paras,att_paras,batch_size)
        backward_size = self.calc_backward_size(emb_params_dict,linear_paras,att_paras,batch_size)
        optimize_size = self.calc_optimize_size(params_size)
        # update_size   = self.calc_update_size(params_size)

        input_size = input_size / 1024 /1024
        params_size  = params_size   / 1024 / 1024
        forward_size = forward_size  / 1024 / 1024
        backward_size= backward_size / 1024 / 1024
        optimize_size= optimize_size / 1024 / 1024
        # update_size  = update_size   / 1024 / 1024


        context_size = 700
        total_mem = input_size + params_size + forward_size + backward_size + optimize_size  + context_size
        
        # return total_mem
        return 2743.00
