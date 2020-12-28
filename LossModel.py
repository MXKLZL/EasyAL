import torch
import torch.nn as nn 
import torch.nn.functional as F 

"""
Got inspiration from the original paper, 'Learning Loss for Active Learning'
http://arxiv.org/abs/1905.03677
"""

class LossModel(nn.Module):
    def __init__(self,device,num_filters=[160,160,160,320], out_dim=128):
        super().__init__()

        self.device = device
        self.gap = {}
        self.fc = {}
        for i in range(len(num_filters)):
          self.gap[f"gap{i}"] = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        for j in range(len(num_filters)):
          self.fc[f"fc{j}"] = nn.Linear(num_filters[j], out_dim)

        for key, value in self.gap.items():
          self.gap[key] = self.gap[key].to(self.device)
        
        for key, value in self.fc.items():
          self.fc[key] = self.fc[key].to(self.device)

        self.linear = nn.Linear(len(num_filters) * out_dim, 1)
        



    def forward(self, inputs):
        out = []
        for i in range(len(inputs)):
          tmp = self.gap[f"gap{i}"](inputs[i])
          tmp = tmp.view(tmp.shape[0],-1)
          tmp = F.relu(self.fc[f"fc{i}"](tmp))
          out.append(tmp)

        output = self.linear(torch.cat(out, 1))
        return output