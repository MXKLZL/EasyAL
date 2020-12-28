import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, models
import torch.nn.functional as F
from tqdm.notebook import tqdm
from BaseModel import BaseModel
from TwoOutputClassifier import TwoOutputClassifier
from Evaluation import classification_evaluation

'''
Got inspiration from the work of CuriousAI, the original author of Mean Teacher
Github Link: https://github.com/CuriousAI/mean-teacher
Paper: https://arxiv.org/abs/1703.01780
'''

class MEBaseModel(BaseModel):
    def __init__(self, dataset, model_name, configs, test_ds=None, weight=True, test_mode=False):
        super().__init__(dataset, model_name, configs)

        self.model = self.__get_model(model_name)
        self.ema_model = self.__get_model(model_name, ema=True)
        self.query_schedule = iter(configs['query_schedule'])
        self.use_weight = weight
        self.test_mode = test_mode
        if test_ds:
            self.testloader = torch.utils.data.DataLoader(
                test_ds, batch_size=32)
            self.test_target = test_ds.target_list

        self.__init_model_set_up()

    def update(self):
        self.labeled_index = self.get_labeled_index(self.dataset)
        self.init_data_loaders()
        self.init_class_weights()

    def __get_model(self, model_name, ema=False):
        model = None

        if model_name == 'mobilenet':
            model = models.mobilenet_v2(pretrained=self.configs['pretrained'])
            num_ftrs = model.classifier[1].in_features
            model.classifier = TwoOutputClassifier(num_ftrs, self.num_class)
            model = model.to(self.device)
            children = list(list(model.children())[0].children())

        if self.configs['pretrained']:
            for child in children[:len(children) - self.configs['num_ft_layers']]:
                for param in child.parameters():
                    param.require_grad = False

        assert model is not None, "Invalid model_name"

        if ema:
            for param in model.parameters():
                param.detach_()

        return model

    def __init_model_set_up(self):
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.00003)
        self.global_step = 0
        self.start_epoch = 0

    def pred_acc(self, testloader, test_target, criterion='f1'):
        _, pred = torch.max(self.predict(testloader), 1)
        return classification_evaluation(pred, test_target, criterion, 'weighted')

    def fit(self):
        self.dataset.set_mode(2)
        alpha = self.configs['alpha']
        ramp_length = self.configs['ramp_length']
        logit_distance_cost = 1e-2
        optimizer = self.optimizer
        num_epochs = self.configs['epoch']
        self.model.train()
        start_epoch = self.start_epoch

        loss_weights = None
        if self.configs['weighted_loss']:
            loss_weights = self.get_loss_weights()

        try:
            # get next epoch to stop to wait for query
            stop_epoch = next(self.query_schedule)
        except StopIteration: 
            stop_epoch  = num_epochs

        for epoch in tqdm(range(start_epoch, stop_epoch)):

            num_labeled = len(self.labeled_index)

            running_loss = 0.0
            running_sl = 0.0
            running_ul = 0.0
            running_rl = 0.0
            running_corrects_lb = 0
            running_corrects_ulb = 0

            if self.use_weight:
                weight = weight_scheduler(
                    epoch, ramp_length, 3000, num_labeled, self.num_train)
            else:
                weight = 0

            labeled_loader_iter = iter(self.data_loader_labeled)

            label_count = 0

            for i, ((input_u, input_u_ema), labels_u) in enumerate(self.data_loader_unlabeled):
                try:
                    (input_l, input_l_ema), labels_l = next(labeled_loader_iter)
                except StopIteration:
                    labeled_loader_iter = iter(self.data_loader_labeled)
                    (input_l, input_l_ema), labels_l = next(labeled_loader_iter)

                # adjust learning rate missed

                optimizer.zero_grad()
                # minibatch_size = self.configs['labeled_batch_size'] + self.configs['unlabeled_batch_size']
                # labeled_minibatch_size = self.configs['labeled_batch_size']
                labeled_minibatch_size = input_l.size(0)
                minibatch_size = input_u.size(0) + labeled_minibatch_size
                label_count += labeled_minibatch_size

                ema_input = torch.cat((input_l_ema, input_u_ema), 0)
                model_input = torch.cat((input_l, input_u), 0)
                labels = torch.cat((labels_l, labels_u))

                ema_input = ema_input.to(self.device)
                model_input = model_input.to(self.device)
                labels = labels.to(self.device)

                labeled_mask_batch = np.array(
                    np.zeros(minibatch_size), dtype=bool)
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

                ema_logit = ema_logit.detach()

                _, preds = torch.max(class_logit, 1)

                loss, ul, sl, rl = total_loss(
                    class_logit, cons_logit, ema_logit, labels, labeled_mask_batch, weight, loss_weights, logit_distance_cost)
                loss.backward()
                optimizer.step()
                self.global_step += 1
                update_ema_variables(
                    self.model, self.ema_model, alpha, self.global_step)

                running_loss += loss.item() * model_input.size(0)
                running_sl += sl.item() * model_input.size(0)
                running_ul += ul.item() * model_input.size(0)
                running_rl += rl.item() * model_input.size(0)
                running_corrects_lb += torch.sum(
                    preds[labeled_mask_batch] == labels[labeled_mask_batch].data)
                if self.test_mode:
                    running_corrects_ulb += torch.sum(
                        preds[~labeled_mask_batch] == labels[~labeled_mask_batch].data)

            epoch_loss = running_loss / self.num_train
            epoch_acc_lb = running_corrects_lb.double() / label_count
            if self.test_mode:
                epoch_acc_ulb = running_corrects_ulb.double(
                ) / (self.num_train - len(self.labeled_index))
            else:
                epoch_acc_ulb = -1
            epoch_sl = running_sl / self.num_train
            epoch_ul = running_ul / self.num_train
            epoch_rl = running_rl / self.num_train
            self.start_epoch += 1

            print(f'{epoch} Loss {epoch_loss : .4f} SL {epoch_sl : .4f} UL {epoch_ul: .4f} RL {epoch_rl: .4f} Acc Lb {epoch_acc_lb: .4f} Acc Ulb {epoch_acc_ulb : .4f}')

            if self.testloader:
                test_acc = self.pred_acc(self.testloader, self.test_target)
                print(f'Test_Acc {test_acc : .4f} ')

            # set mode back to two ouput
            # self.dataset.set_mode(2)


def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


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


def total_loss(class_output, cons_output, ensemble, labels, indicator, unlabel_weight, class_weight, res_weight):
    ul = unsup_loss(cons_output, ensemble, unlabel_weight)
    sl = sup_loss(class_output, labels, indicator,
                  weights=class_weight)/len(class_output)
    rl = symmetric_mse_loss(class_output, cons_output)/len(class_output)

    return ul+sl+res_weight*rl, ul, sl, rl


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


def symmetric_mse_loss(input1, input2):
    num_classes = input1.size()[1]
    return torch.sum((input1 - input2)**2) / num_classes
