import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from encoders import StyleInvariantEncoder
from classifiers import FeatureSpaceClassifier
from utils import GradManip

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class StyleDisentangleModel(nn.Module):
    def __init__(self, config):
        super(StyleDisentangleModel, self).__init__()
        self.config = config

        self.encoder = StyleInvariantEncoder(self.config['meta_config']['vocab_size'],
                                             self.config['meta_config']['embedding_dim'],
                                             config=self.config['model_config'])


        self.f_classifier = FeatureSpaceClassifier(self.config['model_config']['hid_size'],
                                                   self.config['meta_config']['num_domains'])

        self.reverse_grad = GradManip()

    def forward(self, x):
        z, c = self.encoder(x)
        #print('z shape is ', z.shape)
        z = z[:,-1]
        #print('mean z shape is ', z.shape)
        #rev_z = self.reverse_grad(z)
        #print('rev_z shape is ', rev_z.shape)
        output = self.f_classifier(z)
        #print('output shape is ', output.shape)

        return output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {'meta_config': {'vocab_size': 33161, 'num_domains': 2, 'embedding_dim': 100},
          'model_config': {
                       'model_type': 'LSTM', 'hid_size': 100, 'num_layers': 1,
                       'batch_first': True, 'dropout': 0.0, 'bidirectional': False}}

opt = {'lr': 1e-5, 'momentum': 0.5, 'batch_size': 32}

# Prepare dataloader
data = np.load('political.npz')['data']
target = np.load('political.npz')['target']


trainset = torch.utils.data.TensorDataset(torch.from_numpy(data[:1000]).long(), torch.from_numpy(target[:1000]).long())
valset = torch.utils.data.TensorDataset(torch.from_numpy(data[1000:1200]).long(), torch.from_numpy(target[1000:1200]).long())

trainloader = torch.utils.data.DataLoader(dataset = trainset, batch_size = opt['batch_size'], shuffle = True, num_workers = 2)
validloader = torch.utils.data.DataLoader(dataset = valset, batch_size = opt['batch_size'], shuffle = False, num_workers = 2)

model = StyleDisentangleModel(config)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

model.to(device)

optimizer = optim.SGD(model.parameters(), lr=opt['lr'], momentum=opt['momentum'],
                      weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

num_epoch = 50

interval = 100

for epoch in range(1, num_epoch):
    print('| - at epoch {}'.format(epoch))
    model.train()
    for i, (data, target) in enumerate(trainloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        pred = torch.argmax(output, dim=1)
        acc = np.where((pred - target).cpu().data.numpy() == 0)[0].shape[0] / pred.size(0)
        print('Training Accuracy is ', acc)

        #print(output.shape)
        #print(target.shape)
        loss = criterion(output, target)
        if i % interval == 0:
          print('step {}, loss {}'.format(i, loss.data))

        loss.backward()
        optimizer.step()
