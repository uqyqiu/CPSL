import torch
from tqdm import tqdm
import utils

# train for one epoch to learn unique features
def train(args, net, data_loader, train_optimizer, batch_size, epoch, g_epoch, dynamic_tau=False):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    
    for batch in train_bar:
        if args.selection_method != 'kmeans_select':
            pos_1, pos_2, _, = batch
        else:
            pos_1, pos_2, _, entropy = batch
        if dynamic_tau and g_epoch > 2:
            temperature = args.tau_max * torch.sigmoid(-entropy) + args.tau_min
        else:
            temperature = args.temperature
        pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
        _, out_1 = net(pos_1)
        _, out_2 = net(pos_2)
        # [2*B, D]
        loss = utils.loss_func(out_1, out_2, temperature, batch_size, debiased=False)
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, args.c_epochs, total_loss / total_num))

    return total_loss / total_num, net


