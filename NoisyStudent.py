import gc
import math
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, models
import torch.nn.functional as F
from tqdm.notebook import tqdm
from BaseModel import BaseModel
from Evaluation import classification_evaluation

class NoisyStudent(BaseModel):
    def __init__(self, dataset, model_name, configs, test_ds=None, teacher_list = None):
        super().__init__(dataset, model_name, configs)
        self.teacher_list = teacher_list
        self.teacher_target = None
        self.student_epoch = configs['student_epoch']
        self.dropout = 0

        if test_ds:
            self.testloader = torch.utils.data.DataLoader(
                test_ds, batch_size=32)
            self.test_target = test_ds.target_list

    def init_data_loaders(self):
        unlabeled_index = self.get_unlabeled_index()
        dataset_labeled = torch.utils.data.Subset(self.dataset, self.labeled_index)
        dataset_unlabeled = torch.utils.data.Subset(self.dataset, unlabeled_index)

        self.dataset_unlabeled = dataset_unlabeled
        self.data_loader_labeled = torch.utils.data.DataLoader(dataset_labeled,
                                                               batch_size=self.configs['label_batch_size'])
        self.data_loader_unlabeled = torch.utils.data.DataLoader(dataset_unlabeled,
                                                                 batch_size=self.configs['label_batch_size'])

    def pred_acc(self, testloader, test_target, criterion='f1'):
        _, pred = torch.max(self.predict(testloader), 1)
        return classification_evaluation(pred, test_target, criterion, 'weighted')

    def __get_model(self, model_name):

        if model_name == 'resnet18':
            model = models.resnet18(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(self.dropout),
                nn.Linear(num_ftrs, self.num_class)
            )
            model = model.to(self.device)
            children = list(model.children())

        if model_name == 'resnet34':
            model = models.resnet18(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(self.dropout),
                nn.Linear(num_ftrs, self.num_class)
            )
            model = model.to(self.device)
            children = list(model.children())

        if model_name == 'resnet50':
            model = models.resnet50(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(self.dropout),
                nn.Linear(num_ftrs, self.num_class)
            )
            model = model.to(self.device)
            children = list(model.children())

        if model_name == 'resnet101':
            model = models.resnet101(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(self.dropout),
                nn.Linear(num_ftrs, self.num_class)
            )
            model = model.to(self.device)
            children = list(model.children())

        if model_name == 'mobilenet':
            model = models.mobilenet_v2(pretrained=True)
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, self.num_class)
            model = model.to(self.device)
            children = list(list(model.children())[0].children())

        for child in children[:len(children) - self.configs['num_ft_layers']]:
            for param in child.parameters():
                param.require_grad = False

        return model

    def fit(self, train_epoch = None):
        self.dataset.set_mode(0)

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.0001)

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
            criterion = self.configs['loss_function'](weight=loss_weights)
        else:
            criterion = self.configs['loss_function']()

        if train_epoch:
            num_epochs = train_epoch
        else:
            num_epochs = self.configs['epoch']

        self.model.train()

        for epoch in tqdm(range(num_epochs)):
            #print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            #print('-' * 10)

            running_loss = 0.0
            running_corrects = 0

            if not self.teacher_target:
                for inputs, labels in self.data_loader_labeled:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    optimizer.zero_grad()

                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

            else:
                targets = iter(self.teacher_target[1])

                data_loader_unlabeled = self.get_new_unlabeled_loader()

                for inputs, labels in self.data_loader_labeled:
                    try:
                        input_u, label_u = next(data_loader_unlabeled)
                    except StopIteration:
                        data_loader_unlabeled = self.get_new_unlabeled_loader()
                        input_u, label_u = next(data_loader_unlabeled)
                        targets = iter(self.teacher_target[1])

                    input_u = input_u.to(self.device)
                    predict_label = torch.from_numpy(np.array([next(targets) for i in range(label_u.shape[0])]))
                    predict_label = predict_label.to(self.device)

                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    optimizer.zero_grad()

                    outputs1 = self.model(inputs)
                    _, preds1 = torch.max(outputs1, 1)
                    loss1 = criterion(outputs1, labels)

                    outputs2 = self.model(input_u)
                    _, preds2 = torch.max(outputs2, 1)
                    loss2 = criterion(outputs2, predict_label)

                    loss = loss1 + loss2

                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds1 == labels.data)

            epoch_loss = running_loss / len(self.labeled_index)
            epoch_acc = running_corrects.double() / len(self.labeled_index)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format('Train',epoch_loss, epoch_acc.item()))

    def filter(self):
        probabilities, labels = torch.max(self.predict_unlabeled(), 1)
        probabilities, labels = np.array(probabilities), np.array(labels)

        # filter work
        idx1 = []
        idx2 = []
        for ii, each in enumerate(probabilities):
            if each > 0.5:
                idx1.append(ii)
                idx2.append(labels[ii])
        self.teacher_target = [idx1, idx2]

    def update(self):
        self.labeled_index = self.get_labeled_index(self.dataset)
        self.init_data_loaders()

        self.filter()

        torch.cuda.empty_cache()
        gc.collect()

        self.dropout = self.configs['dropout']
        
        new_model = self.teacher_list[0]
        self.model_name = new_model
        self.model = self.__get_model(new_model)
        self.teacher_list = self.teacher_list[1:]

        self.data_loader_unlabeled = torch.utils.data.DataLoader(self.dataset_unlabeled,
                                                                 batch_size=self.configs['labeled_batch_size'])
        self.init_class_weights()

    def test_train(self):
        self.fit(self.configs['initial_epoch'])
        print('teacher evaluation-' + self.model_name + ':  ' + str(self.pred_acc(self.testloader, self.test_target)))

        while len(self.teacher_list) > 0:
            self.update()

            new_epoch = self.student_epoch[0]
            self.fit(new_epoch)
            self.student_epoch = self.student_epoch[1:]
            print('student evaluation-' + self.model_name + ':  ' + str(self.pred_acc(self.testloader, self.test_target)))

    def get_new_unlabeled_loader(self):

        unlabel_count = len(self.teacher_target[0])
        label_count = len(self.labeled_index)
        rounds = math.ceil(label_count / self.configs['labeled_batch_size'])
        unlabel_batchsize = min(64, int(unlabel_count / rounds))

        unlabeled_index = self.get_unlabeled_index()

        dataset_unlabeled = torch.utils.data.Subset(self.dataset, unlabeled_index[self.teacher_target[0]])
        return iter(torch.utils.data.DataLoader(dataset_unlabeled, batch_size=unlabel_batchsize))
