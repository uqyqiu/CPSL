import os, copy, torch
import torch.optim as optim
import numpy as np
import utils, dataset, train_contrs
from model import Model,SupModel
import config
from linear import linear_ft
import train_sup
import selection
import kmeans_selection

def main(args, main_trainData, ex_trainData, valData, testData, min_cls):

    sup_trainLoader = dataset.get_trainDataloader(main_trainData, args.sup_bs, shuffle=True, num_workers=8)
    ex_menory_loader = dataset.get_testDataloader(ex_trainData, args.batch_size, shuffle=False, num_workers=8)
    val_loader = dataset.get_testDataloader(valData, args.batch_size, shuffle=False, num_workers=8)
    test_loader = dataset.get_testDataloader(testData, args.sup_bs, shuffle=False, num_workers=8)
    # records settings
    num_cls = len(np.unique(main_trainData[1]))
    sampling_results = {'epoch': [], 'samplingA': [], 'samplingB': [], 'weights_dist': [], 'frac_main': [], 'frac_ex': []}
    results_sup = {'train_loss': [], 'test_loss': [],'test_acc': [], 'recall': [], 'precision': [], 'f1': []}
    results_main = {'train_loss': [], 'test_loss': [],'test_acc': [], 'recall': [], 'precision': [], 'f1': []}
    results_loss = {'train_loss': []}
    results_global = {'train_loss': [], 'test_loss': [], 'test_acc': [], 'recall': [], 'precision': [], 'f1': []}
    results_first_sup = {'train_loss': [], 'test_loss': [], 'test_acc': [], 'recall': [], 'precision': [], 'f1': []}
    results_ = {'train_loss': [], 'test_loss': [], 'test_acc': [], 'recall': [], 'precision': [], 'f1': []}
    utils.update_result_dic(results_sup, num_cls)
    utils.update_result_dic(results_main, num_cls)
    utils.update_result_dic(results_global, num_cls)
    utils.update_result_dic(results_first_sup, num_cls)
    results_avg = copy.deepcopy(results_global)
    results_temp = copy.deepcopy(results_global)
    utils.update_result_dic(results_main, len(np.unique(main_trainData[1])))

    save_name_pre = 'SEED:{}_TEMP:{}_BS:{}_IR:{}_GEPOCHS:{}_CEPOCHS:{}_MIN:{}_SAMNUM:{}_SIZEA:{}_SELEMETHOD:{}'.format(args.seed, 
                                                                                                            args.temperature, 
                                                                                                            args.batch_size,  
                                                                                                            args.im_ratio, 
                                                                                                            args.g_epochs,
                                                                                                            args.c_epochs,
                                                                                                            min_cls,
                                                                                                            args.sampling_num,
                                                                                                            len(main_trainData[0]),
                                                                                                            args.selection_method)

    save_path = '../results/' + args.reason4Exp + '/' + args.hyperpara +  '/' + save_name_pre
    A_records = save_path + '/A_records'
    B_records = save_path + '/B_records'

    if not os.path.exists('../results'):
        os.mkdir('../results')

    if not os.path.exists('../results/' + args.reason4Exp):
        os.mkdir('../results/' + args.reason4Exp) 

    if not os.path.exists('../results/' + args.reason4Exp + '/' + args.hyperpara):
        os.mkdir('../results/' + args.reason4Exp + '/' + args.hyperpara)
    
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    if not os.path.exists(A_records):
        os.mkdir(A_records)
        os.mkdir(A_records + '/figs')  

    traindata_dic = {'samples': main_trainData[0], 'labels': main_trainData[1]}
    val_dic = {'samples': valData[0], 'labels': valData[1]}
    torch.save(traindata_dic, save_path + '/traindata.pt')
    torch.save(val_dic, save_path + '/valdata.pt') 

    # Supervised model setup and optimizer config
    Sup_model = SupModel().cuda()
    Sup_optimizer = optim.Adam(Sup_model.parameters(), lr=1e-4, weight_decay=1e-7)
    loss_criterion = torch.nn.CrossEntropyLoss()

    # train a fully-supervised learning on A
    best_acc = 0
    for epoch in range(1, args.sup_epochs + 1):
        train_loss, _ = train_sup.train_val(args, Sup_model, sup_trainLoader, loss_criterion, epoch, Sup_optimizer)   
        val_loss, val_results = train_sup.train_val(args, Sup_model, val_loader, loss_criterion, epoch)

        results_sup = utils.checkpoints(results_sup, train_loss, val_results, A_records, cls_num=2,
                                        filename='first_train',test_loss=val_loss)
        results_temp = utils.checkpoints(results_temp, train_loss, val_results, A_records, cls_num=2, test_loss=val_loss)
        
        if epoch % 5 == 0:
            avg_ = utils.avg_results(results_temp)
            results_avg = utils.add_value_to_dict(results_first_sup, avg_)
            utils.plot_results(args, results_sup, results_avg, sampling_results, A_records, 'first_val_plot')
            utils.clear_dict(results_temp)

        if val_results[0] > best_acc:
            best_acc = val_results[0]
            torch.save(Sup_model.state_dict(), save_path + '/sup_model.pth')
    Sup_model.load_state_dict(torch.load(save_path + '/sup_model.pth'))
    # evaluate the model on test set
    _, test_results = train_sup.train_val(args, Sup_model, test_loader, loss_criterion, epoch)
    results_sup = utils.checkpoints(results_sup, train_loss, test_results, A_records, cls_num=2,
                                        filename='first_train_evaluation',test_loss=val_loss)
    

    Contra_model = Model().cuda()
    Contra_optimizer = optim.Adam(Contra_model.parameters(), lr=1e-3, weight_decay=1e-7)
    selected_index = []
    sampled_idxs = []
    best_acc = 0
    for g_epoch in range(1, args.g_epochs + 1):
        loss_list, _, feats_list, probabilities = utils.get_feats_loss_list_pu(Sup_model, ex_menory_loader, feat_dim=512)

        # select data samples from the external dataset
        if args.selection_method == 'radnom_selection':
            selected_Dataloader, select_num, _ = selection.randomly_select(args, ex_trainData, g_epoch, select_num=args.sampling_num)

        elif args.selection_method == 'loss_based':
            theta = 0.3 + (g_epoch-1) * 0.1 
            theta = theta if theta < 0.5 else 0.5
            selected_Dataloader, select_num, _ = selection.loss_select(args, ex_trainData, loss_list, probabilities, select_num=args.sampling_num, theta=theta)

        elif args.selection_method == 'informativeness':
            selected_Dataloader, select_num, _ = selection.informativeness(args, ex_trainData, probabilities)
        
        elif args.selection_method == 'diversity':
            selected_Dataloader, select_num, _, selected_index = selection.diversity_select(args, ex_trainData, feats_list, 
                                                                                            probabilities, epoch, selected_index)
        elif args.selection_method == 'loss_informative':
            theta = 0.3 + (g_epoch-1) * 0.1 
            theta = theta if theta < 0.5 else 0.5
            selected_Dataloader, select_num, _ = selection.loss_informative(args, ex_trainData, loss_list, probabilities, theta=theta)
        
        elif args.selection_method == 'loss_diversity':
            theta = 0.3 + (g_epoch-1) * 0.1 
            theta = theta if theta < 0.5 else 0.5
            selected_Dataloader, select_num, _, selected_index = selection.loss_diversity(args, ex_trainData, feats_list, loss_list, 
                                                                                            probabilities, epoch, selected_index, theta=theta)
        elif args.selection_method == 'infor_diversity':
            selected_Dataloader, select_num, _ = selection.infor_diversity(args, ex_trainData, feats_list, probabilities, epoch, selected_index)

        elif args.selection_method == 'kmeans_select':
            theta = 0.3 + (g_epoch-1) * 0.1 
            theta = theta if theta < 0.5 else 0.5
            entropy_list, select_num, selectData, selected_index = kmeans_selection.kmeans_select_data(args, ex_trainData, feats_list, loss_list, probabilities,
                                                                                                           g_epoch, selected_index, theta=theta)
            selected_Dataloader = dataset.get_trainDataloader(selectData, batch_size=args.batch_size, 
                                                    shuffle=True, num_workers=8, drop_last=True, entropy_list=entropy_list)
        
        sampling_results['samplingA'].append(select_num[0])
        sampling_results['samplingB'].append(select_num[1])
        sampling_results['epoch'].append(g_epoch)

        print('Select majority data: {}, \n minority data: {}'.format(select_num[0], select_num[1]))
        
        for epoch in range(1, args.c_epochs+1):
            train_loss, Contra_model = train_contrs.train(args, Contra_model, selected_Dataloader, Contra_optimizer, g_epoch=g_epoch,
                                                                batch_size=selected_Dataloader.batch_size, epoch=epoch,dynamic_tau=args.dynamic_tau)
            results_loss['train_loss'].append(train_loss)
            torch.save(Contra_model.state_dict(), '{}/model.pth'.format(save_path))

        # fine tuen the model on A dataset
        Sup_model =  linear_ft(args, save_path, Sup_model, A_records, g_epoch)
        _, eval_results = train_sup.train_val(args, Sup_model, val_loader, loss_criterion, epoch)
        if eval_results[0] > best_acc:
            best_acc = eval_results[0]
            torch.save(Sup_model.state_dict(), save_path + '/best_final_model.pth')
        results_global = utils.checkpoints(results_global, train_loss, eval_results, A_records, cls_num=2, 
                                        filename='global_evaluation') 
        utils.plot_results(args, results_global, results_global, sampling_results, A_records, 'global_evaluation')
        results_main = {key: [] for key in results_main} 
        results_loss = {key: [] for key in results_loss} 

    # evaluate the model on test set
    Sup_model.load_state_dict(torch.load(save_path + '/best_final_model.pth'))
    _, test_results = train_sup.train_val(args, Sup_model, test_loader, loss_criterion, epoch)
    results_sup = utils.checkpoints(results_sup, train_loss, test_results, A_records, cls_num=2,
                                        filename='final_train_evaluation')
    os.remove(save_path + '/model_best.pth')
    os.remove(save_path + '/best_final_model.pth')
    os.remove(save_path + '/sup_model.pth')
    os.remove(save_path + '/traindata.pt')
    os.remove(save_path + '/valdata.pt')
    os.remove(save_path + '/model.pth')


    print("========== Done! ==========")

if __name__ == '__main__':
    data_root = '../data/cifar10/'

    args, _ = config.get_config()
    utils.setup_seed(args.seed)

    for min_cls in range(10):
        task_name = 'min_cls:{}'.format(min_cls)
        print(task_name)

        main_trainData, ex_trainData, val_Data, testData = dataset.LOAD_CIFAR_BINARY_LT(args, data_root, min_cls=min_cls)
        # print('external data size: ',len(ex_trainData_rescale[1]))
        main(args, main_trainData, ex_trainData, val_Data, testData, min_cls=min_cls)

