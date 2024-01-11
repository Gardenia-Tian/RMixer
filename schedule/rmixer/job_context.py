import importlib

# record memory consumption and time needed
class JobContext():
    def __init__(self,model_name,config):
        self.arithmetic_intensity_dict = {
            # models                   # utilization 
            "avazu_widedeep":            55.0,
            "criteo_difm":               39.5,  
            "criteo_widedeep":           27.0,  
            "criteo_dcn2":               25.0,  
            "criteo_dlrm":               23.0,  
            "beauty_bert4rec":           22.0,  
            "alidisplay_dmr":            18.0,  
            "kdd_dpin":                  15.0,  
            "amazon_dien":                7.2,  
            "amazon_bst":                 6.8,  
            "criteo_dcn":                 5.9,  
            "avazu_flen":                   5,
            "amazon_din":                 4.8,  
            "criteo_deepfm":             4.47,  
        }
        self.high_arin_type_set = { 
                                    'criteo_difm', 
                                    'criteo_dcn2', 
                                    'criteo_widedeep', 
                                    'criteo_dlrm', 
                                    'alidisplay_dmr',
                                    'beauty_bert4rec',
                                    'kdd_dpin',
                                    'avazu_widedeep'}
        self.high_memo_type_set = { 'amazon_bst',   
                                    'amazon_dien',
                                    'avazu_flen'}
        self.both_low_type_set  = { 'amazon_din', 
                                    'criteo_deepfm',   
                                    'criteo_dcn'}
        
        self.model_name = model_name
        self.config = config
        self.finish = False 
        self.assign = False
        self.memory = self.get_model_memory()
        # 1 : high_arin    0 : high_memo     -1 : both_low
        if model_name in self.high_arin_type_set:
            self.type = 1
        elif model_name in self.high_memo_type_set:
            self.type = 0
        else:
            self.type = -1

    def get_model_memory(self):
        dygraph_model = importlib.import_module('.' + self.model_name + '.dygraph_model', package="models")
        dy_model_class = dygraph_model.DygraphModel()
        total_mem = dy_model_class.calc_mem(self.config)
        return total_mem