from __future__ import print_function

import sys
sys.insert(0, '.')
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms
import torchvision.models as models

from vgg import VGG

learning_rate = 1e-3
batch_size = 16
epoches = 30
data_dir = 'dataset'
save_path = 'final.pth'
train = 'train'
test = 'val'

def trainval(dataloders, model, optimizer, scheduler, criterion, dataset_sizes, phase='train'):

    for epoch in range(epoches):
        print('Epoch {}/{}'.format(epoch, epoches - 1))
        print('-' * 10)

        if phase == 'train':
            model.train()
        elif phase == val:
            model.eval()
        else:
            print ('eeeeeeeee')
            return

        running_loss = 0.0
        running_corrects = 0.0

        for data in dataloders[phase]:
            imgs, labels = data
            if use_gpu:
                imgs = imgs.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()

            outputs = model(imgs)
            preds = torch.argmax(outputs.data, 1)
            loss = criterion(outputs, labels)

            if phase == 'train':
                loss.backward()
                optimizer.step()
                scheduler.step()

            with torch.no_grad():
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels).item()

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects / dataset_sizes[phase]

        print('{} Loss: {:.5f} Acc: {:.5f}'.format(phase, epoch_loss, epoch_acc))

    if phase == 'train':
        if use_gpu:
            model = model.cpu()
        torch.save(model.state_dict(), save_path)

if __name__ == '__main__':
	data_transforms = {
        'train': transforms.Compose([
            #transforms.ToTensor(),
            #transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        val: transforms.Compose([
            #transforms.ToTensor(),
            #transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in [train, val]}

    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                 batch_size=8,
                                                 shuffle=True,
                                                 num_workers=4) for x in [train, val]}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', val]}

    use_gpu = torch.cuda.is_available()

    model = VGG(2)

    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))

    if use_gpu:
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.95)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    print ('*' * 10)
    print ('start training')
    trainval(dataloders, model, optimizer, scheduler, criterion, dataset_sizes, phase='train')
