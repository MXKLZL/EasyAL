from BaseModel import *
from MEBaseModel import *
from TEBaseModel import *
from LossPredictBaseModel import *
from NoisyStudentBaseModel import *


def get_model(dataset,model_name,train_configs,model_type = 'Basic',test_ds=None):
    if model_type == 'Basic':
        return BaseModel(dataset,model_name,train_configs)
    elif model_type == 'MeanTeacher':
        return MEBaseModel(dataset, model_name, train_configs, test_ds=test_ds)
    elif model_type == 'TemporalEnsembling':
        return TEBaseModel(dataset, model_name, train_configs, test_ds=test_ds)
    elif model_type == 'NoisyStudent':
        return NoisyStudentBaseModel(dataset,model_name,train_configs)
    elif model_type == 'Loss':
        return LossPredictBaseModel(dataset,model_name,train_configs)
    else:
        print('Please Enter A Valid Model_Type. Choose from [Basic,MeanTeacher,TemporalEnsembling,NoisyStudent,Loss]')

