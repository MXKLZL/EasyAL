import torch.nn as nn

'''
Got inspiration from the work of CuriousAI, the original author of Mean Teacher
Github Link: https://github.com/CuriousAI/mean-teacher
Paper: https://arxiv.org/abs/1703.01780
'''

class TwoOutputClassifier(nn.Module):
    def __init__(self, in_features, num_classes):
        super(TwoOutputClassifier, self).__init__()
        self.fc1 = nn.Linear(in_features, num_classes,bias=True)
        self.fc2 = nn.Linear(in_features, num_classes,bias=True)
        self.dropout = nn.Dropout(p=0.2, inplace=False)
    
    def forward(self, x):
        x = self.dropout(x)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        return x1,x2