import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, models
import torch.nn.functional as F
from tqdm.notebook import tqdm
from BaseModel import BaseModel
from Evaluation import classification_evaluation

'''
Got inspiration from the Johan Ferret's pytorch implementation of Temporal Ensembling
Github Link: https://github.com/ferretj/temporal-ensembling

and the original paper of Temporal Ensembling, 'Temporal Ensembling for Semi-Supervised Learning'
Paper: https://arxiv.org/abs/1610.02242
'''

class TEBaseModel(BaseModel):
    def __init__(self, dataset, model_name, configs, test_ds=None, weight=True):
        super().__init__(dataset, model_name, configs)
        self.data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=configs['batch_size'])
        self.use_weight = weight
        self.query_schedule = iter(configs['query_schedule'])

        if test_ds:
            self.testloader = torch.utils.data.DataLoader(
                test_ds, batch_size=32)
            self.test_target = test_ds.target_list
        
        self.__init_model_set_up()

    def update(self):
        self.labeled_index = self.get_labeled_index(self.dataset)
        self.init_data_loaders()
        self.init_class_weights()

    def pred_acc(self, testloader, test_target, criterion='f1'):
        _, pred = torch.max(self.predict(testloader), 1)
        return classification_evaluation(pred, test_target, criterion, 'weighted')

    def __init_model_set_up(self):
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.0001)
        self.Z = torch.zeros(self.num_train, self.num_class).float().to(self.device)
        self.z = torch.zeros(self.num_train, self.num_class).float().to(self.device)
        self.start_epoch = 0

    def fit(self):
        self.dataset.set_mode(0)
        batch_size = self.configs['batch_size']
        alpha = self.configs['alpha']
        ramp_length = self.configs['ramp_length']
        outputs = torch.zeros(
            self.num_train, self.num_class).float().to(self.device)

        loss_weights = None
        start_epoch = self.start_epoch

        if self.configs['weighted_loss']:
            loss_weights = self.get_loss_weights()

        num_epochs = self.configs['epoch']

        try:
            # get next epoch to stop to wait for query
            stop_epoch = next(self.query_schedule)
        except StopIteration: 
            stop_epoch  = num_epochs

        self.model.train()

        for epoch in tqdm(range(start_epoch, stop_epoch)):

            labeled_mask = np.array(np.zeros(self.num_train), dtype=bool)
            labeled_mask[self.labeled_index] = True
            num_labeled = len(self.labeled_index)

            running_loss = 0.0
            running_sl = 0.0
            running_ul = 0.0
            running_corrects_lb = 0


            if self.use_weight:
                weight = weight_scheduler(
                    epoch, ramp_length, 3000, num_labeled, self.num_train)
            else:
                weight = 0

            for i, (inputs, labels) in enumerate(self.data_loader):
                labels = labels.to(self.device)
                inputs = inputs.to(self.device)

                self.optimizer.zero_grad()
                outputs_batch = self.model(inputs)
                _, preds = torch.max(outputs_batch, 1)

                z_batch = self.z[i*batch_size: (i+1)*batch_size].detach()
                labeled_mask_batch = labeled_mask[i *
                                                  batch_size: (i+1)*batch_size]
                loss, ul, sl = total_loss(
                    outputs_batch, z_batch, labels, labeled_mask_batch, weight, loss_weights)

                outputs[i * batch_size: (i + 1) *
                        batch_size] = outputs_batch.detach().clone()

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_sl += sl.item() * inputs.size(0)
                running_ul += ul.item() * inputs.size(0)
                running_corrects_lb += torch.sum(
                    preds[labeled_mask_batch] == labels[labeled_mask_batch].data)
                

            self.Z = alpha * self.Z + (1. - alpha) * outputs
            self.z = self.Z * (1. / (1. - alpha ** (epoch + 1)))

            epoch_loss = running_loss / self.num_train
            epoch_acc_lb = running_corrects_lb.double() / len(self.labeled_index)


            epoch_sl = running_sl / self.num_train
            epoch_ul = running_ul / self.num_train
            self.start_epoch += 1
            print(f'{epoch} Loss {epoch_loss : .4f} SL {epoch_sl : .4f} UL {epoch_ul: .4f} Acc Lb {epoch_acc_lb: .4f}')
            if self.testloader:
                test_acc = self.pred_acc(self.testloader, self.test_target)
                print(f'Test_Acc {test_acc : .4f} ')


def unsup_loss(output, ensemble, unlabel_weight):
    loss = nn.MSELoss(reduction='sum')
    return unlabel_weight*(loss(F.softmax(output, dim=1), F.softmax(ensemble, dim=1))/output.numel())


def sup_loss(output, labels, indicator, weights=None):
    label_output = output[indicator]
    ground_truth = labels[indicator]

    if len(label_output) > 0:
        loss = nn.CrossEntropyLoss(weight=weights, reduction='sum')
        label_loss = loss(label_output, ground_truth)

        return label_loss
    return torch.tensor(0.0, requires_grad=False)


def total_loss(output, ensemble, labels, indicator, unlabel_weight, class_weight):
    ul = unsup_loss(output, ensemble, unlabel_weight)
    sl = sup_loss(output, labels, indicator, weights=class_weight)/len(output)

    return ul+sl, ul, sl


def rampup(epoch, ramp_length):
    if epoch == 0:
        return 0
    elif epoch < ramp_length:
        p = float(epoch) / float(ramp_length)
        p = 1.0 - p
        return np.exp(-p*p*5.0)
    else:
        return 1.0


def weight_scheduler(epoch, ramp_length, weight_max, num_labeled, num_samples):
    weight_max = weight_max * (float(num_labeled) / num_samples)
    rampup_val = rampup(epoch, ramp_length)
    return weight_max*rampup_val
