import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, models
import torch.nn.functional as F
from tqdm.notebook import tqdm
from collections import Counter
from LossModel import LossModel


class LossPredictBaseModel(BaseModel):
    def __init__(self, dataset, model_name, labeled_index, configs):
        super().__init__(dataset, model_name, labeled_index, configs)
        self.loss_model = self.__get_loss_model()
        #self.loss_feat_layers = configs['loss_feat_layers']
        self.loss_feat_layers = [14,15,16,17]

    def __get_loss_model(self):
        loss_model = LossModel()
        loss_model = loss_model.to(self.device)
        return loss_model
        
    
    def fit(self):
        self.dataset.set_mode(0)

        activation = {}
        hook_handles = self.__add_hook(self.loss_feat_layers, activation)

        optimizer_cf = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.0001)
        optimizer_lm = torch.optim.Adam(filter(lambda p: p.requires_grad, self.loss_model.parameters()), lr=0.0001)

        criterion = nn.CrossEntropyLoss(reduction='none')
        num_epochs = self.configs['epoch']
        epoch_loss = self.configs['epoch_loss']

        self.model.train()
        self.loss_model.train()

        for epoch in tqdm(range(num_epochs)):
            running_loss = 0.0
            running_corrects = 0
            running_lm_loss = 0.0

            for inputs, labels in self.data_loader_labeled():
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                optimizer_cf.zero_grad()
                optimizer_lm.zero_grad()

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

                features = []
                for layer in self.loss_feat_layers:
                    features.append(activation[layer])
                
                if epoch >= epoch_loss:
                    for i in range(len(features)):
                        features[i] = features[i].detach()
                
                loss_preds = self.loss_model(features)
                loss_preds = loss_preds.view(loss_preds.size(0))

                cf_loss = criterion(outputs, labels)
                lm_loss = pair_comparison_loss(loss_preds, cf_loss, margin = configs['margin'])

                loss =  torch.sum(cf_loss)/cf_loss.size(0) + configs['lambda']*lm_loss

                loss.backward()
                optimizer_cf.step()
                optimizer_lm.step()

                running_loss += torch.sum(cf_loss).item()
                running_corrects += torch.sum(preds == labels.data)
                running_lm_loss += lm_loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(self.labeled_index)
            epoch_acc = running_corrects.double() / len(self.labeled_index)
            epoch_lm_loss = running_lm_loss / len(self.labeled_index)

            print('{} Loss: {:.4f} Acc: {:.4f} LM Loss {:.4f}'.format('Train',epoch_loss, epoch_acc.item(), epoch_lm_loss))

    
    def __add_hook(self, loss_feat_layers, activation):
        # add hook to given network, so that intermediate output will be recorded
        hook_handles = []
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output
            return hook

        for layer_num in loss_feat_layers:
            handle = self.model.features[layer_num].register_forward_hook(get_activation(layer_num))
            hook_handles.append(handle)
        
        return hook_handles

    def predict_loss(self, test_data_loader):
        # predict loss for given samples
        self.dataset.set_mode(1)
        self.model.eval()
        self.loss_model.eval()
        
        activation = {}
        self.__add_hook(self.loss_feat_layers, activation)

        predicted_loss = None

        with torch.no_grad():
            for inputs, labels in test_data_loader:
                inputs = inputs.to(self.device)
                preds =  self.model(inputs)
                features = []
                for layer in self.loss_feat_layers:
                    features.append(activation[layer])
                loss_preds = self.loss_model(features)
                loss_preds = loss_preds.view(loss_preds.size(0))

                if predicted_loss is None:
                    predicted_loss = loss_preds
                else:
                    predicted_loss = torch.cat((predicted_loss, loss_preds))
        
        return predicted_loss.cpu()

    def predict_unlabeled_loss(self):
        return self.predict_loss(self.data_loader_unlabeled)

def pair_comparison_loss(predicted_loss, true_loss, margin):
    pass
