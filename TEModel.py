import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, models
import torch.nn.functional as F
from tqdm.notebook import tqdm
from BaseModel import BaseModel

class TEModel(BaseModel):
    def __init__(self, dataset, model_name, labeled_index, configs):
        super().__init__(dataset, model_name, labeled_index, configs)
        self.data_loader = torch.utils.data.DataLoader(dataset, batch_size = configs['batch_size'])
        
    def fit(self):
        self.dataset.set_mode(0)
        batch_size = self.configs['batch_size']
        alpha = configs['alpha']
        ramp_length = configs['ramp_length']

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.002, betas=(0.9, 0.99))

        Z = torch.zeros(self.num_train, self.num_class).float().to(self.device)
        z = torch.zeros(self.num_train, self.num_class).float().to(self.device)
        outputs = torch.zeros(self.num_train, self.num_class).float().to(self.device)

        loss_weights = None
        if self.configs['weighted_loss']:
            loss_weights = []
            for class_name in self.dataset.classes:
                class_id = self.dataset.class_name_map[class_name]
                if class_id in self.class_counts:
                    loss_weights.append(self.class_counts[class_id])
                else:
                    loss_weights.append(1e-4)

            loss_weights=sum(loss_weights)/torch.FloatTensor(loss_weights)/len(self.dataset.classes)

            if torch.cuda.is_available():
                loss_weights = loss_weights.cuda()
        
        num_epochs = self.configs['epoch']
        self.model.train()

        for epoch in tqdm(range(num_epochs)):
            
            labeled_mask = np.array(np.zeros(self.num_train), dtype=bool)
            labeled_mask[self.labeled_index] = True
            num_labeled = len(self.labeled_index)

            running_loss = 0.0
            running_corrects = 0

            weight = weight_scheduler(epoch, ramp_length, 30 , num_labeled, self.num_train)

            for i, (inputs, labels) in self.data_loader:
                labels = labels.to(self.device)
                inputs = inputs.to(self.device)

                optimizer.zero_grad()
                outputs_batch = self.model(inputs)
                _, preds = torch.max(outputs_batch, 1)

                z_batch = z[i*batch_size: (i+1)*batch_size].detach()
                loss = total_loss(outputs_batch, z_batch,labels,labeled_mask,weight,loss_weights)

                outputs[i * batch_size: (i + 1) * batch_size] = outputs_batch.detach().clone()

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            Z = alpha * Z + (1. - alpha) * outputs
            z = Z * (1. / (1. - alpha ** (epoch + 1)))

            epoch_loss = running_loss / len(self.labeled_index)
            epoch_acc = running_corrects.double() / len(self.labeled_index)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format('Train',epoch_loss, epoch_acc.item()))



def unsup_loss(output,ensemble,unlabel_weight):
  loss = nn.MSELoss(reduction = 'sum')
  return unlabel_weight*(loss(F.softmax(output, dim=1),F.softmax(ensemble, dim=1))/output.numel())


def sup_loss(output,labels,indicator,weights = None):
  label_output = output[indicator]
  ground_truth = labels[indicator]

  if len(label_output) > 0:
    loss = nn.CrossEntropyLoss(weight = weights)
    label_loss = loss(label_output,ground_truth)

    return label_loss
  return torch.tensor(0.0,requires_grad=False)



def total_loss(output, ensemble,labels,indicator,unlabel_weight,class_weight):
  ul = unsup_loss(output,ensemble,unlabel_weight)
  sl = sup_loss(output,labels,indicator,weights = class_weight)

  return ul+sl


def rampup(epoch,ramp_length):
  if epoch == 0:
    return 0
  elif epoch < ramp_length:
    p = float(epoch) / float(ramp_length)
    p = 1.0 - p
    return math.exp(-p*p*5.0)
  else:
      return 1.0

def weight_scheduler(epoch, ramp_length, weight_max,num_labeled, num_samples):
  weight_max = weight_max * (float(num_labeled) / num_samples)
  rampup_val = rampup(epoch,ramp_length)
  return weight_max*rampup_val



