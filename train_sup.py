import torch
from tqdm import tqdm
import utils

# train or test for one epoch
def train_val(args, net, data_loader, loss_func, epoch, train_optimizer=None):
    is_train = train_optimizer is not None
    net.train() if is_train else net.eval()

    total_loss, total_correct, total_num, data_bar = 0.0, 0.0, 0, tqdm(data_loader)
    target_true, target_pred = [], []
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, _, target in data_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            _, out = net(data)
            loss = loss_func(out, target)

            if is_train:
                loss.backward()
                train_optimizer.step()
                train_optimizer.zero_grad()

            total_num += data.size(0)
            total_loss += loss.item() * data.size(0)
            pred_labels = torch.argsort(out, dim=-1, descending=True)
            total_correct += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            target_true += target.data.tolist()
            target_pred += pred_labels[:, :1].data.tolist()
            recall_global, recall, precision_global, precision, f1_global, f1, report = utils.criterias(target_true, target_pred)

            data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} ACC: {:.2f}% Re: {:.2f}% Pre: {:.2f}% f1: {:.2f}% '
                                     .format('Train' if is_train else 'Test', 
                                            epoch, args.linear_epochs, 
                                            total_loss / total_num,
                                            total_correct / total_num * 100,
                                            recall_global * 100,
                                            precision_global * 100,
                                            f1_global * 100,
                                            ))

    return total_loss / total_num, (total_correct / total_num, recall_global, recall, precision_global, precision, f1_global, f1, report)