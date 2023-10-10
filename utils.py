import pandas as pd
import torch
import random
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
import torch.nn as nn

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def update_result_dic(results, cls_num):
    for i in range(cls_num):
        results.update({'recall_{}'.format(i): [], 'precision_{}'.format(i): [], 'f1_{}'.format(i): []})

def save_results(results, save_path, filename):
    data_frame = pd.DataFrame.from_dict(data=results, orient='index')
    data_frame.to_csv('{}/statistics_{}.csv'.format(save_path, filename))

def checkpoints(results, train_loss, test_results, save_path, cls_num, filename=None, test_loss=None):
    results['train_loss'].append(train_loss)
    if 'test_loss' in results.keys():
        results['test_loss'].append(test_loss)
    results['test_acc'].append(test_results[0]* 100)
    results['recall'].append(test_results[1]* 100)
    results['precision'].append(test_results[3]* 100)
    results['f1'].append(test_results[5]* 100)
    for i in range(cls_num):
        results['recall_{}'.format(i)].append(test_results[2][i])
        results['precision_{}'.format(i)].append(test_results[4][i])
        results['f1_{}'.format(i)].append(test_results[6][i])
    if filename is not None:
        save_results(results, save_path, filename)
        report = test_results[-1]
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv('{}/report_{}.csv'.format(save_path, filename))
    
    return results

def avg_results(dict):
    avg = {}
    for i in dict.keys():
        temp_dic = {i:np.mean(dict[i])}
        avg.update(temp_dic)
    return avg

def clear_dict(dict):
    for i in dict.keys():
        dict[i] = []
    return dict

def add_value_to_dict(dict_old, dict_new):
    for i in dict_new.keys():
        dict_old[i].append(dict_new[i])
    return dict_old

def criterias(y_true, y_pred, conf_mat:bool=False):
    Conf_Mat = confusion_matrix(y_true, y_pred)
    if conf_mat:
        print(Conf_Mat)
    report = classification_report(y_true, y_pred, output_dict=True)
    recall_global = recall_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average=None)

    precision_global = precision_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=None)
    
    f1_global = f1_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average=None)

    return recall_global, recall, precision_global, precision, f1_global, f1, report

def loss_func(out_1, out_2, temperature, batch_size, tau_plus=0.6, debiased=False):
        if isinstance(temperature,float):
            temperature = temperature
            pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        else:
            temperature = torch.cat([temperature, temperature], dim=0).cuda()
            pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature[:batch_size])
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

        # compute loss
        # pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature[:batch_size])
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        
        # estimator g()
        """
        Debaised loss function from:
        https://github.com/chingyaoc/DCL/blob/master/main.py
        """
        if debiased:
            N = batch_size * 2 - 2
            Ng = (-tau_plus * N * pos_sim + sim_matrix.sum(dim = -1)) / (1 - tau_plus)
            # constrain (optional)
            Ng = torch.clamp(Ng, min = N * np.e**(-1 / temperature))
            loss = (- torch.log(pos_sim / (sim_matrix.sum(dim=-1) + Ng))).mean()

        else:
            # Ng = sim_matrix.sum(dim=-1)
            loss = (- torch.log(pos_sim / (sim_matrix.sum(dim=-1)))).mean()

        return loss

def bank_collect(net, test_data_loader):
    net.eval()
    feature_bank, fea_target = [], []

    with torch.no_grad():
        for data, _, target, _ in test_data_loader: 
            feature, out = net(data.cuda(non_blocking=True))

            feature_bank.append(feature)
            fea_target.append(target)

    feature_bank = torch.cat(feature_bank, dim=0).contiguous()
    # feature_bank = feature_bank.topk(k, dim=-1)[0]
    fea_target = torch.cat(fea_target, dim=0).to(feature_bank.device).cpu().numpy()
    return feature_bank, fea_target

def plot_results(args, results, results_avg, sampling_results, save_path, file_name):
    fontsize = 20
    plt.figure(figsize=(30, 35), dpi=300, facecolor="white")
    plt.figure(1)
    plt.suptitle('Summary plot, seed={}'.format(args.seed), fontsize=fontsize*1.5)
    y_major_locator=MultipleLocator(5)
    plt.xticks(fontsize=fontsize * 0.8)
    plt.yticks(fontsize=fontsize * 0.8)

    ax1 = plt.subplot(5,2,1)
    if 'test_loss' in results.keys():
        ax1.plot(results['test_loss'], label='test_loss', color='red')
        ax1.plot(results['train_loss'], label='train_loss', color='blue')
        ax1.legend(fontsize=fontsize * 0.8)
    else: 
        ax1.plot(results['train_loss'])
    ax1.set_title('Loss', fontsize=fontsize)
    ax1.set_xlabel('Epochs', fontsize=fontsize * 0.8)
    ax1.set_ylabel('Loss', fontsize=fontsize * 0.8)
    ax1.yaxis.grid(True, linestyle='-.')
    ax1.autoscale()
    
    ax2 = plt.subplot(5,2,2)
    ax2.plot(results_avg['test_acc'])
    ax2.set_title('Test Accuracy', fontsize=fontsize)
    ax2.set_xlabel('Epochs', fontsize=fontsize * 0.8)
    ax2.set_ylabel('Accuracy', fontsize=fontsize * 0.8)
    # ax2.yaxis.set_major_locator(y_major_locator)
    ax2.yaxis.grid(True, linestyle='-.')
    ax2.autoscale()

    ax3 = plt.subplot(5,2,3)
    ax3.plot(results_avg['recall'])
    ax3.set_title('Recall', fontsize=fontsize)
    ax3.set_xlabel('Epochs', fontsize=fontsize * 0.8)
    ax3.set_ylabel('Recall', fontsize=fontsize * 0.8)
    # ax3.yaxis.set_major_locator(y_major_locator)
    ax3.yaxis.grid(True, linestyle='-.')
    ax3.autoscale()

    ax4 = plt.subplot(5,2,4)
    ax4.plot(results_avg['precision'])
    ax4.set_title('Precision', fontsize=fontsize)
    ax4.set_xlabel('Epochs', fontsize=fontsize * 0.8)
    ax4.set_ylabel('Precision', fontsize=fontsize * 0.8)
    # ax4.yaxis.set_major_locator(y_major_locator)
    ax4.yaxis.grid(True, linestyle='-.')
    ax4.autoscale()

    ax5 = plt.subplot(5,2,5)
    ax5.plot(results_avg['f1'])
    ax5.set_title('F1 Score', fontsize=fontsize)
    ax5.set_xlabel('Epochs', fontsize=fontsize * 0.8)
    ax5.set_ylabel('F1-Score', fontsize=fontsize * 0.8)
    # ax5.yaxis.set_major_locator(y_major_locator)
    ax5.yaxis.grid(True, linestyle='-.')
    ax5.autoscale()

    ax6 = plt.subplot(5,2,6)
    for i in range(2):
        ax6.plot(results_avg['recall_{}'.format(i)], label='class{}'.format(i))
    # ax6.plot(results['recall_0'], label='recall_majprity')
    # ax6.plot(results['recall_1'], label='recall_minority')
    ax6.set_title('Recall for each class', fontsize=fontsize)
    ax6.set_xlabel('Epochs', fontsize=fontsize * 0.8)
    ax6.set_ylabel('Recall', fontsize=fontsize * 0.8)
    # ax6.yaxis.set_major_locator(y_major_locator)
    ax6.yaxis.grid(True, linestyle='-.')
    ax6.legend(fontsize=fontsize * 0.8)
    ax6.autoscale()

    ax7 = plt.subplot(5,2,7)
    for i in range(2):
        ax7.plot(results_avg['precision_{}'.format(i)], label='class{}'.format(i))
    # ax7.plot(results['precision_0'], label='pre_majority')
    # ax7.plot(results['precision_1'], label='pre_minoroty')
    ax7.set_title('Precision for each class', fontsize=fontsize)
    ax7.set_xlabel('Epochs', fontsize=fontsize * 0.8)
    ax7.set_ylabel('Precision', fontsize=fontsize * 0.8)
    # ax7.yaxis.set_major_locator(y_major_locator)
    ax7.yaxis.grid(True, linestyle='-.')
    ax7.legend(fontsize=fontsize * 0.8)
    ax7.autoscale()

    ax8 = plt.subplot(5,2,8)
    for i in range(2):
        ax8.plot(results_avg['f1_{}'.format(i)], label='class{}'.format(i))
    # ax8.plot(results['f1_0'], label='f1_majority')
    # ax8.plot(results['f1_1'], label='f1_minority')
    ax8.set_title('F1-score for each class', fontsize=fontsize)
    ax8.set_xlabel('Epochs', fontsize=fontsize * 0.8)
    ax8.set_ylabel('F1-score', fontsize=fontsize * 0.8)
    # ax8.yaxis.set_major_locator(y_major_locator)
    ax8.yaxis.grid(True, linestyle='-.')
    ax8.legend(fontsize=fontsize * 0.8)
    ax8.autoscale()

    epoch = sampling_results['epoch']
    samplingA_list = sampling_results['samplingA']
    samplingB_list = sampling_results['samplingB']
    ax9 = plt.subplot(5,2,9)
    x = list(range(len(samplingA_list)))
    width = 0.3
    ax9.bar(x, samplingA_list, width=width, label="ClassA", fc = "b")

    for a,b in zip(x,samplingA_list):
        plt.text(a,b,'%d'%b,ha='center',va='top',fontsize=10)
    for i in range(len(x)):
        x[i] = x[i] + width

    ax9.bar(x, samplingB_list, width=width, label="ClassB", tick_label = epoch, fc ="r")
    ax9.set_xlabel("the epochs of sampling ",fontsize=fontsize * 0.8)
    ax9.set_ylabel("the numebr of each sampling operation",fontsize=fontsize * 0.8)
    ax9.set_title("Sampling distribtino for every sampling operation",fontsize=fontsize)
    for a,b in zip(x,samplingB_list):
        ax9.text(a,b,'%d'%b,ha='center',va='top',fontsize=10)
    ax9.legend()

    weights_dist = sampling_results['weights_dist']
    frac_main = sampling_results['frac_main']
    frac_ex = sampling_results['frac_ex']
    ax10 = plt.subplot(5,2,10)
    ax10.plot(weights_dist, label="avg weights' distance / 10", linestyle='-', marker='*')
    ax10.plot(frac_main, label="fraction of A", linestyle='-', marker='o')
    ax10.plot(frac_ex, label="fraction of B", linestyle=':', marker='x')
    ax10.yaxis.grid(True, linestyle='-.')
    ax10.legend(fontsize=fontsize * 0.8)

    plt.savefig(save_path + '/figs/results_' + file_name + '.png')
    plt.close()

    return

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def get_feats_loss_list_pu(model, ex_loader, feat_dim=128):
    ln_ce = nn.CrossEntropyLoss(reduction='none')
    model.eval()
    
    loss_list = []
    targets_list = []
    p = []
    feats_list = torch.zeros(
            (len(ex_loader.dataset), 64), dtype=torch.float32)
            
    with torch.no_grad():
        ptr = 0
        for images, _, target in ex_loader:
            images = images.cuda()
            labels = target.cuda()
            targets = torch.zeros(images.size(0)).long().cuda()

            feature, cl_out = model(images)
            loss_ce = ln_ce(cl_out, targets).cpu()

            p.append(torch.softmax(cl_out.detach()/0.2, dim=-1))
            loss_list.append(loss_ce)
            targets_list.append(labels)
            feats_list[ptr:ptr + images.size(0)] = feature.detach().cpu()
            ptr += images.size(0)

    loss_list = torch.cat(loss_list)
    loss_list = (loss_list - torch.min(loss_list))/ (torch.max(loss_list) - torch.min(loss_list))    # Normalization
    targets_list = torch.cat(targets_list).cpu()
    p = torch.cat(p).cpu()
    
    return loss_list, targets_list, feats_list, p


def shuffle_data(data, targets):
    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)
    data = data[idx]
    targets = targets[idx]
    return (data, targets)

def loss_fn_kd(outputs, labels, teacher_outputs, T=20, alpha=0.2):
    hard_loss = F.cross_entropy(outputs, labels) * (1. - alpha)
    soft_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                               F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T)
    return hard_loss + soft_loss