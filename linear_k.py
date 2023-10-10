import torch, warnings
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from train_sup import train_val
from model_k import Model
from train_kd import train_kd
import utils
import dataset_k as dataset
warnings.filterwarnings('ignore')

class SupNet(nn.Module):
    def __init__(self, num_class, pretrained_path, temp=0.07):
        super(SupNet, self).__init__()

        # encoder
        self.f = Model().f
        # classifier
        self.fc = nn.Linear(64, num_class, bias=True)
        self.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)
        self.g = nn.Sequential(nn.Linear(512, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, 64, bias=True))
        self.softmax = nn.Softmax(dim=1)
        self.temp = temp

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        feature = self.g(feature)
        out = self.fc(feature)
        return F.normalize(feature, dim=-1), self.softmax(out) / self.temp

def linear_ft(args, path, pre_model, record_path, g_epoch):
    traindata_dir= path + '/traindata.pt'
    valdata_dir = path + '/valdata.pt'
    batch_size = 128
    
    trainData = (torch.load(traindata_dir)['samples'], torch.load(traindata_dir)['labels'])
    valData = (torch.load(valdata_dir)['samples'], torch.load(valdata_dir)['labels'])

    train_loader = dataset.get_trainDataloader(trainData, batch_size, shuffle=True, num_workers=8, drop_last=True)
    val_loader = dataset.get_testDataloader(valData, batch_size, shuffle=False, num_workers=8)
    model_path = path + '/model.pth'
        
    num_cls = np.unique(trainData[1]).shape[0]
    model = SupNet(num_class=num_cls, pretrained_path=model_path).cuda()

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-7)
    loss_criterion = nn.CrossEntropyLoss()

    results = {'train_loss': [], 'test_loss': [],'test_acc': [], 'recall': [], 'precision': [], 'f1': []}
    utils.update_result_dic(results, num_cls)

    best_acc = 0
    for epoch in range(1, args.linear_epochs + 1):
        train_loss, _ = train_kd(args, pre_model, model, train_loader, epoch, g_epoch=g_epoch, train_optimizer=optimizer)
        _, test_results = train_val(args, model, val_loader, loss_criterion, epoch)

        if test_results[0] > best_acc:
            best_acc = test_results[0]
            torch.save(model.state_dict(), path + '/model_best.pth')
    # evaluate
    model.load_state_dict(torch.load(path + '/model_best.pth'))

    return model