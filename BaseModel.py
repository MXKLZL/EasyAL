import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, models
import torch.nn.functional as F
from tqdm.notebook import tqdm

class BaseModel():

    def __init__(self, dataset, model_name, labeled_index, configs):
        self.configs = configs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_class = len(dataset.classes)
        self.labeled_index = labeled_index
        self.num_train = len(dataset)
        self.dataset = dataset
        unlabeled_index = self.get_unlabeled_index()
        dataset_labeled = torch.utils.data.Subset(dataset, labeled_index)
        dataset_unlabeled = torch.utils.data.Subset(dataset, unlabeled_index)

        self.data_loader_labeled = torch.utils.data.DataLoader(dataset_labeled, batch_size = configs['batch_size'])
        self.data_loader_unlabeled = torch.utils.data.DataLoader(dataset_unlabeled, batch_size = configs['batch_size'])
        self.model = self.__get_model(model_name)
        
        class_counts = dict(Counter(sample_tup[1] for sample_tup in self.data_loader_labeled.dataset))
        class_counts = dict(sorted(class_counts.items()))
        self.weights = {}
        for class_name in self.dataset.classes:
          class_id = dataset.class_name_map[class_name]
          self.weights[class_id] = [1/class_counts[class_id]]
    
    def get_unlabeled_index(self):
        return np.setdiff1d(np.arange(self.num_train), self.labeled_index)

    def __get_model(self, model_name):
        if model_name == 'resnet18':
            model = models.resnet18(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, self.num_class)
        
        model = model.to(self.device)
        children = list(model.children())
        for child in children[:len(children) - self.configs['num_ft_layers']]:
            for param in child.parameters():
                param.require_grad = False

        return model

    def fit(self):
        self.dataset.set_mode(0)

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.0001)
        criterion = self.configs['loss_function']
        num_epochs = self.configs['epoch']

        self.model.train()

        for epoch in tqdm(range(num_epochs)):
            #print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            #print('-' * 10)

            running_loss = 0.0
            running_corrects = 0

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

            epoch_loss = running_loss / len(self.labeled_index)
            epoch_acc = running_corrects.double() / len(self.labeled_index)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format('Train',epoch_loss, epoch_acc.item()))

    def predict(self, test_data_loader):
        self.dataset.set_mode(1)
        self.model.eval()
        preds = None
        with torch.no_grad():
            for inputs, labels in test_data_loader:
                
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                output = self.model(inputs)
                output = F.softmax(output, dim=1)
                output = output.cpu()
                if preds is not None:
                    preds = torch.cat((preds, output))
                else:
                    preds = output
        
        return preds

    def predict_unlabeled(self):
        return self.predict(self.data_loader_unlabeled)

    def get_embedding(self, test_data_loader):
        self.dataset.set_mode(1)
        backup_layer = self.model.fc
        self.model.fc = nn.Sequential()
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
        
        self.model.fc = backup_layer
        return embeddings

    def get_embedding_unlabeled(self):
        return self.get_embedding(self.data_loader_unlabeled)

    
