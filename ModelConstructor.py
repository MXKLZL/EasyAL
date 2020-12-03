from BaseModel import *
from MEBaseModel import *
from TEModel import *
from LossPredictBaseModel import *


def get_model_class(dataset,model_name,train_configs,model_type = 'Basic',test_ds=None):
    if model_type == 'Basic':
        return BaseModel(dataset,model_name,train_configs)
    elif model_type == 'MeanTeacher':
        return MEBaseModel(dataset, model_name, train_configs, test_ds=test_ds)
    elif model_type == 'TemporalEnsembling':
        return TEModel(dataset, model_name, train_configs, test_ds=test_ds)
    elif model_type == 'NoisyStudent':
        return BaseModel(dataset,model_name,train_configs,semi = True)
    elif model_type == 'Loss':
        return LossPredictBaseModel(dataset,model_name,train_configs)
    else:
        print('Please Enter A Valid Model_Type. Choose from [Basic,MeanTeacher,TemporalEnsembling,NoisyStudent,Loss]')

