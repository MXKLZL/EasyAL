import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, models
import torch.nn.functional as F
from tqdm.notebook import tqdm
from BaseModel import BaseModel
from TwoOutputClassifier import TwoOutputClassifier

class MEBaseModel(BaseModel):
    def __init__(self, dataset, model_name, labeled_index, configs):
        super().__init__(dataset, model_name, labeled_index, configs)

        self.model = self.__get_model(model_name)
        self.ema_model = self.__get_model(model_name, ema = True)


    
    def __get_model(self, model_name,ema = False):

        if model_name == 'mobilenet':
            model = models.mobilenet_v2(pretrained=True)
            num_ftrs = model.classifier[1].in_features
            model.classifier = TwoOutputClassifier(num_ftrs, self.num_class)
            model = model.to(self.device)
            children = list(list(model.children())[0].children())
        
        for child in children[:len(children) - self.configs['num_ft_layers']]:
            for param in child.parameters():
                param.require_grad = False
        
        if ema:
            for param in model.parameters():
                param.detach_()


        return model
        
    def fit(self):
        global_step = 0
        self.dataset.set_mode(2)
        batch_size = self.configs['batch_size']
        alpha = self.configs['alpha']
        ramp_length = self.configs['ramp_length']
        logit_distance_cost = 1e-2

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.0001)

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
            
            
            num_labeled = len(self.labeled_index)

            running_loss = 0.0
            running_sl = 0.0
            running_ul = 0.0
            running_corrects_lb = 0
            running_corrects_ulb = 0

            weight = weight_scheduler(epoch, ramp_length, 3000 , num_labeled, self.num_train)

            labeled_loader_iter = iter(self.data_loader_labeled)

            for i, ((input_u, input_u_ema), labels_u) in self.data_loader_unlabeled:
                try:
                    (input_l, input_l_ema), labels_l = next(labeled_loader_iter)
                except StopIteration:
                    labeled_loader_iter = iter(self.data_loader_labeled)
                    (input_l, input_l_ema), labels_l = next(labeled_loader_iter)

                # adjust learning rate missed

                optimizer.zero_grad()
                minibatch_size = self.configs['labeled_batch_size'] + self.configs['unlabeled_batch_size']
                labeled_minibatch_size = self.configs['labeled_batch_size']

                ema_input = torch.cat(input_l_ema, input_u_ema)
                model_input = torch.cat(input_l, input_u)
                labels = torch.cat(labels_l, labels_u)

                labeled_mask_batch = np.array(np.zeros(minibatch_size), dtype=bool)
                labeled_mask_batch[:labeled_minibatch_size] = True

                ema_model_out = self.ema_model(ema_input)
                model_out = self.model(model_input)

                assert len(ema_model_out) == 2
                assert len(model_out) == 2

                logit1, logit2 = model_out
                ema_logit, _ = ema_model_out

                if logit_distance_cost >= 0:
                    class_logit, cons_logit = logit1, logit2
                else:
                    class_logit, cons_logit = logit1, logit1
                    res_loss = 0

                ema_logit = ema_logit.detch()

                _, preds = torch.max(class_logit, 1)

                loss, ul, sl, rl = total_loss()
                loss.backward()
                optimizer.step()
                global_step += 1
                update_ema_variables(self.model, self.ema_model,alpha, global_step)

                running_loss += loss.item() * model_input.size(0)
                running_sl += sl.item() * model_input.size(0)
                running_ul += ul.item() * model_input.size(0)
                running_corrects_lb += torch.sum(preds[labeled_mask_batch] == labels[labeled_mask_batch].data)
                running_corrects_ulb += torch.sum(preds[~labeled_mask_batch] == labels[~labeled_mask_batch].data)

            epoch_loss = running_loss / self.num_train
            epoch_acc_lb = running_corrects_lb.double() / len(self.labeled_index)
            epoch_acc_ulb = running_corrects_ulb.double() / (self.num_train - len(self.labeled_index))
            epoch_sl = running_sl / self.num_train
            epoch_ul = running_ul / self.num_train


            print(f'Loss {epoch_loss : .4f} SL {epoch_sl : .4f} UL {epoch_ul: .4f} Acc Lb {epoch_acc_lb: .4f} Acc Ulb {epoch_acc_ulb : .4f}')


def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def unsup_loss(output,ensemble,unlabel_weight):
  loss = nn.MSELoss(reduction = 'sum')
  return unlabel_weight*(loss(F.softmax(output, dim=1),F.softmax(ensemble, dim=1))/output.numel())


def sup_loss(output,labels,indicator,weights = None):
  label_output = output[indicator]
  ground_truth = labels[indicator]

  if len(label_output) > 0:
    loss = nn.CrossEntropyLoss(weight = weights,reduction = 'sum')
    label_loss = loss(label_output,ground_truth)

    return label_loss
  return torch.tensor(0.0,requires_grad=False)



def total_loss(class_output,cons_output,ensemble,labels,indicator,unlabel_weight,class_weight,res_weight):
  ul = unsup_loss(cons_output,ensemble,unlabel_weight)
  sl = sup_loss(class_output,labels,indicator,weights = class_weight)/len(class_output)
  rl = symmetric_mse_loss(class_output,cons_output)

  return ul+sl+res_weight*rl, ul, sl, rl


def rampup(epoch,ramp_length):
  if epoch == 0:
    return 0
  elif epoch < ramp_length:
    p = float(epoch) / float(ramp_length)
    p = 1.0 - p
    return np.exp(-p*p*5.0)
  else:
      return 1.0

def weight_scheduler(epoch, ramp_length, weight_max,num_labeled, num_samples):
  weight_max = weight_max * (float(num_labeled) / num_samples)
  rampup_val = rampup(epoch,ramp_length)
  return weight_max*rampup_val



def symmetric_mse_loss(input1, input2):
    num_classes = input1.size()[1]
    return torch.sum((input1 - input2)**2) / num_classes




def create_model(model_name,ema=False):


        if ema:
            for param in model.parameters():
                param.detach_()

        return model






