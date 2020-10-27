import torch
import torch.nn as nn 
import torch.nn.functional as F 


class LossNet(nn.Module):
    def __init__(self, num_filters=[160,160,160,320], out_dim=128):
        super(LossNet, self).__init__()

        self.gap = {}
        self.fc = {}
        for i in range(len(num_filters)):
          self.gap["gap{0}".format(i)] = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        for j in range(len(num_filters)):
          self.fc["fc{0}".format(j)] = nn.Linear(num_filters[j], out_dim)
      

        self.linear = nn.Linear(len(num_filters) * out_dim, 1)
    
    def forward(self, inputs):
        out = []
        for i in range(len(inputs)):
          tmp = self.gap["gap{0}".format(i)](inputs[i])
          tmp = tmp.view(tmp.shape[0],-1)
          tmp = F.relu(self.fc["fc{0}".format(i)](tmp))
          out.append(tmp)

        output = self.linear(torch.cat(out, 1))
        return output