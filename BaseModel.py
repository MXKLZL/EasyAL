import math
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, models
import torch.nn.functional as F
from tqdm.notebook import tqdm
from collections import Counter

class BaseModel():

    def __init__(self, dataset, model_name, labeled_index, configs, semi = False, teacher_target = None):
        self.configs = configs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_class = len(dataset.classes)
        self.labeled_index = labeled_index
        self.num_train = len(dataset)
        self.dataset = dataset
        self.semi = semi
        self.teacher_target = teacher_target
        self.__init_data_loaders()
        
        self.model_name = model_name
        self.model = self.__get_model(model_name)
        
        class_counts = dict(Counter(sample_tup[1] for sample_tup in self.data_loader_labeled.dataset))
        self.class_counts = dict(sorted(class_counts.items()))
        self.weights = {}
        for class_name in self.dataset.classes:
          class_id = dataset.class_name_map[class_name]
          if class_id not in class_counts:
            self.weights[class_id] = 1
          else:
            self.weights[class_id] = 1/class_counts[class_id]

    def __init_data_loaders(self):
        unlabeled_index = self.get_unlabeled_index()
        dataset_labeled = torch.utils.data.Subset(self.dataset, self.labeled_index)
        dataset_unlabeled = torch.utils.data.Subset(self.dataset, unlabeled_index)

        self.dataset_unlabeled = dataset_unlabeled
        self.data_loader_labeled = torch.utils.data.DataLoader(dataset_labeled, batch_size = self.configs['labeled_batch_size'])
        self.data_loader_unlabeled = torch.utils.data.DataLoader(dataset_unlabeled, batch_size = self.configs['unlabeled_batch_size'])

    def query_cost(self, query_idx, weights=None):
        if weights is None:
          weights = self.weights
        dataset_query = torch.utils.data.Subset(self.dataset, query_idx)
        class_counts = dict(Counter(sample_tup[1] for sample_tup in dataset_query))
        class_counts = dict(sorted(class_counts.items()))
        res = 0
        #for class_name in self.dataset.classes:
        for class_id in class_counts:
          #class_id = self.dataset.class_name_map[class_name]
          res += weights[class_id] * class_counts[class_id]
        return res
    
    def get_unlabeled_index(self):
        return np.setdiff1d(np.arange(self.num_train), self.labeled_index)

    def __get_model(self, model_name):
        if model_name == 'resnet18':
            model = models.resnet18(pretrained=self.configs['pretrained'])
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, self.num_class)
            model = model.to(self.device)
            children = list(model.children())

        if model_name == 'resnet34':
            model = models.resnet18(pretrained=self.configs['pretrained'])
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, self.num_class)
            model = model.to(self.device)
            children = list(model.children())

        if model_name == 'resnet50':
            model = models.resnet50(pretrained=self.configs['pretrained'])
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, self.num_class)
            model = model.to(self.device)
            children = list(model.children())

        if model_name == 'mobilenet':
            model = models.mobilenet_v2(pretrained=self.configs['pretrained'])
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, self.num_class)
            model = model.to(self.device)
            children = list(list(model.children())[0].children())
        
        if self.configs['pretrained']:
            for child in children[:len(children) - self.configs['num_ft_layers']]:
                for param in child.parameters():
                    param.require_grad = False

        return model

    def fit(self):
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

        num_epochs = self.configs['epoch']

        self.model.train()

        for epoch in tqdm(range(num_epochs)):
            #print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            #print('-' * 10)

            running_loss = 0.0
            running_corrects = 0

            if not self.semi:
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
                targets = iter(self.teacher_target)

                data_loader_unlabeled = self.get_new_unlabeled_loader()

                for inputs, labels in self.data_loader_labeled:
                    try:
                        input_u, label_u = next(data_loader_unlabeled)
                    except StopIteration:
                        data_loader_unlabeled = self.get_new_unlabeled_loader()
                        input_u, label_u = next(data_loader_unlabeled)
                        targets = iter(self.teacher_target)

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

    def predict(self, test_data_loader):
        cur_mode = self.dataset.mode
        self.dataset.set_mode(1)
        self.model.eval()
        preds = None
        with torch.no_grad():
            for inputs, labels in test_data_loader:
                
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)


                output = self.model(inputs)

                if isinstance(output, tuple):
                    output = F.softmax(output[0], dim=1)
                output = output.cpu()
                if preds is not None:
                    preds = torch.cat((preds, output))
                else:
                    preds = output

        self.dataset.set_mode(cur_mode)
        return preds

    def predict_unlabeled(self):
        return self.predict(self.data_loader_unlabeled)

    def get_embedding(self, test_data_loader):
        cur_mode = self.dataset.mode
        self.dataset.set_mode(1)

        if self.model_name in ['resnet18', 'resnet34', 'resnet50']:
            backup_layer = self.model.fc
            self.model.fc = nn.Sequential()
        if self.model_name == 'mobilenet':
            backup_layer = self.model.classifier[1]
            self.model.classifier[1] = nn.Sequential()

        embeddings = None
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in test_data_loader:

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                output = self.model(inputs)
                output = output.cpu()
                if embeddings is not None:
                    embeddings = torch.cat((embeddings, output))
                else:
                    embeddings = output
        
        if self.model_name in ['resnet18', 'resnet34', 'resnet50']:
            self.model.fc = backup_layer
        if self.model_name == 'mobilenet':
            self.model.classifier[1] = backup_layer
        

        self.dataset.set_mode(cur_mode)
        return embeddings

    def get_embedding_unlabeled(self):
        return self.get_embedding(self.data_loader_unlabeled)

    def get_new_unlabeled_loader(self):

        unlabel_count = len(self.teacher_target)
        label_count = len(self.labeled_index)
        rounds = math.ceil(label_count / self.configs['batch_size'])
        unlabel_batchsize = int(unlabel_count / rounds)

        return iter(torch.utils.data.DataLoader(self.dataset_unlabeled, batch_size=unlabel_batchsize))