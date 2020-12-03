from BaseModel import *
from MEBaseModel import *
from TEModel import *
from LossPredictBaseModel import *


def get_model_class(dataset,model_name,train_configs,semi = False,test_ds=None,weight=True)