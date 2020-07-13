#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, TensorDataset
# import torch.optim as optim
import torch.functional as F
from sklearn.model_selection import train_test_split
import torchvision.models as models
import torch_optimizer  as optim

from hashimoto_lib.network.tanhexp import Tanhexp
from packages.dataset import TensorDataset_Aug

# In[ ]:
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

train_data = pd.read_csv('../DATASET/mnist/digit-recognizer/train.csv')
test_data = pd.read_csv('../DATASET/mnist/digit-recognizer/test.csv')
sample_submission = pd.read_csv('../DATASET/mnist/digit-recognizer/sample_submission.csv')


# In[ ]:

# test

x_train, x_val, y_train, y_val = train_test_split(train_data.values[:, 1:], train_data.values[:, 0], test_size=0.2)


batch_size = 64
num_epochs = 30

# In[ ]:

x_train_tensor = torch.from_numpy(x_train.astype(np.float32)/255).view(-1, 1, 28, 28)
y_train_tensor = torch.from_numpy(y_train)
x_val_tensor = torch.from_numpy(x_val.astype(np.float32)/255).view(-1, 1, 28, 28)
y_val_tensor = torch.from_numpy(y_val)
test_tensor = torch.from_numpy(test_data.values[:,:].astype(np.float32)/255).view(-1, 1, 28, 28)

aug = transforms.Compose([
    #transforms.Grayscale(),
    transforms.ToPILImage(),
    transforms.RandomApply([
    #transforms.RandomAffine(15),#20
    transforms.RandomRotation(10),#20
    transforms.ToTensor(),
    #transforms.RandomErasing(scale=(0.02, 0.05), ratio=(0.3, 0.5),),
    transforms.ToPILImage(),
    ], p=0.5),

    transforms.ToTensor(),
 # [0,1] => [-1,1]
])

train_dataset = TensorDataset_Aug(x_train_tensor, y_train_tensor,transforms=aug)
val_dataset = TensorDataset_Aug(x_val_tensor, y_val_tensor)
test_dataset = TensorDataset_Aug(test_tensor)

# train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
# val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
# test_dataset = TensorDataset(test_tensor)


# In[ ]:
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)


# In[ ]:


class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size = 3, stride = 2, padding = 1, bias = False), # 14 * 14 * 96
            nn.BatchNorm2d(96),
            #nn.ReLU(True),
            Tanhexp(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 7 * 7 * 96
            nn.Conv2d(96, 256, kernel_size=3, stride=1, padding=1,bias=False),  # 7*7*256
            nn.BatchNorm2d(256),
            #nn.ReLU(True),
            Tanhexp(),
            nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=1, bias=False),     # 4*4*384
            nn.BatchNorm2d(384),
            #nn.ReLU(True),
            Tanhexp(),
            nn.Dropout(0.25),

            nn.Conv2d(384, 384, kernel_size=3, stride=2, padding=1, bias=False),    # 2*2*384
            nn.BatchNorm2d(384),
            #nn.ReLU(True),
            Tanhexp(),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1, bias=False),    # 2*2*384
            nn.BatchNorm2d(384),
            #nn.ReLU(True),
            Tanhexp(),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, bias=False),    # 2*2*256
            nn.BatchNorm2d(256),
            #nn.ReLU(True),
            Tanhexp(),
            nn.MaxPool2d(kernel_size=2, stride=2) ,
            nn.Dropout(0.25),# 1*1*256
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            #nn.ReLU(),
            Tanhexp(),
            nn.Linear(128, 128),
            #nn.ReLU(),
            Tanhexp(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        out = self.feature(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


# In[ ]:

# alexnet = models.alexnet()
# pretrain_alexnet = models.alexnet(pretrained=True)
# net = resnet = models.resnet50(pretrained=False).cuda()
# # print(resnet)

# net = models.resnet18(pretrained=False).cuda()
# print(net)
# net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).cuda()
# net.fc = nn.Linear(512, 10).cuda()

net = Net().cuda()
#optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
# optimizer = optim.Adam(net.parameters(), lr=1.0e-4, betas=(0.5,0.9999),)#momentum=0.9, weight_decay=0.0001)

optimizer = optim.AdaBound(net.parameters(),lr=1.0e-4, betas=(0.5,0.9999))
criterion = nn.CrossEntropyLoss()
f_loss = FocalLoss


# In[ ]:


def test():
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            outputs_test = net(images)
            _, predicted = outputs_test.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        acc = correct / total
        return acc


# In[ ]:


best_acc = 0.0
for epoch in range(num_epochs):
    net.train()
    sum_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = f_loss(outputs, labels)
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()
        if i % 100 == 99:
            print('[%d %d] loss:%.03f' % (epoch+1, i+1, sum_loss / 100))
            sum_loss = 0.0
acc = test()
print(acc)

with open('./save/results.csv', 'a') as f:
    print(acc, file=f)

# In[ ]:


net.eval()
preds = []
for i, data in enumerate(test_loader):
    image = data[0]
    image = image.cuda()
    output = net(image)
    _, prediction = output.max(1)
    preds.extend(prediction.tolist())


# In[ ]:


sample_submission['Label'] = preds
sample_submission.to_csv('./submission/submission.csv', index = False)


# In[ ]:




