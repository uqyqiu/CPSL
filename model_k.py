import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18


class Model(nn.Module):
    def __init__(self, feature_dim=128):
        super(Model, self).__init__()
        self.f = []
        for name, module in resnet18().named_children():
            if name == 'conv1':
                module = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(512, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


class SupModel(nn.Module):
    def __init__(self, num_classes=2, temp=0.5):
        super(SupModel, self).__init__()

        self.f = []
        for name, module in resnet18().named_children():
            if name == 'conv1':
                module = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        self.g = nn.Sequential(nn.Linear(512, 512, bias=False), nn.BatchNorm1d(512),
                                nn.ReLU(inplace=True), nn.Linear(512, 64, bias=True))
        # projection head
        self.classifier = nn.Linear(64, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)
        self.temp = temp

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        feature = self.g(feature)
        out_cl = self.classifier(feature)
        return F.normalize(feature, dim=-1), self.softmax(out_cl) / self.temp