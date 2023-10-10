import numpy as np
import dataset
import torch
from kmeans_selection import *


# randomly select
def randomly_select(args, ex_trainData, epoch, select_num=2000):
    np.random.seed(epoch)
    select_idx = np.random.choice(len(ex_trainData[0]), size=select_num, replace=False)
    selectSamples = ex_trainData[0][select_idx]
    select_targets = ex_trainData[1][select_idx]
    selectData = (selectSamples, select_targets)
    if select_num < args.batch_size:
        batch_size = select_num
    else:
        batch_size = args.batch_size
    
    num_majority = len(np.where(select_targets == 0)[0])
    num_minority = len(np.where(select_targets == 1)[0])

    SelectData_loader = dataset.get_trainDataloader(selectData, batch_size=batch_size, 
                                                            shuffle=True, num_workers=8, drop_last=True)
    return SelectData_loader, (num_majority, num_minority), selectData

# select the samples with the highest loss
def loss_select(args, ex_trainData, loss_list, probabilities, select_num=2000, theta=0.5):
    loss_list = torch.as_tensor(loss_list)
    feats_idx = np.arange(len(ex_trainData[0]))
    T = loss_list.mean() 
    min_budget = int(theta * args.sampling_num)
    maj_budget = args.sampling_num - min_budget
    feats_idx_with_large_loss = feats_idx[loss_list > T]
    feats_idx_maj = feats_idx[loss_list < T]

    if len(feats_idx_with_large_loss) < min_budget:
        min_budget = len(feats_idx_with_large_loss)
        maj_budget = args.sampling_num - min_budget

    min_select_idx = np.random.choice(feats_idx_with_large_loss, size=min_budget, replace=False)
    maj_select_idx = np.random.choice(feats_idx_maj, size=maj_budget, replace=False)

    select_idx = np.concatenate((min_select_idx, maj_select_idx), axis=0)
    selectSamples = ex_trainData[0][select_idx]
    selectLabels = ex_trainData[1][select_idx]
    selectData = (selectSamples, selectLabels)

    if select_num < args.batch_size:
        batch_size = select_num
    else:
        batch_size = args.batch_size

    num_majority = len(np.where(selectLabels == 0)[0])
    num_minority = len(np.where(selectLabels == 1)[0])
    entropy_list = norm(calculate_entropy(probabilities[select_idx]))

    SelectData_loader = dataset.get_trainDataloader(selectData, batch_size=batch_size,
                                                    shuffle=True, num_workers=8, drop_last=True, entropy_list=entropy_list)
    return SelectData_loader, (num_majority, num_minority), selectData


def informativeness(args, ex_trainData, probabilities):
    select_num = args.sampling_num
    entropy = calculate_entropy(probabilities)
    entropy_idx = np.argsort(entropy)
    select_idx = entropy_idx[-select_num:]
    selectSamples = ex_trainData[0][select_idx]
    selectLabels = ex_trainData[1][select_idx]
    selectData = (selectSamples, selectLabels)
    
    if select_num < args.batch_size:
        batch_size = select_num
    else:
        batch_size = args.batch_size
        
    num_majority = len(np.where(selectLabels == 0)[0])
    num_minority = len(np.where(selectLabels == 1)[0])

    SelectData_loader = dataset.get_trainDataloader(selectData, batch_size=batch_size,
                                                    shuffle=True, num_workers=8, drop_last=True, entropy_list=entropy[select_idx])
    
    return SelectData_loader, (num_majority, num_minority), selectData

def diversity_select(args, ex_trainData, feats_list, probabilities, epoch, old_feats_idx):
    old_selection_feats = feats_list[old_feats_idx]

    if epoch == 1:
        similarities = cal_similarity(feats_list, feats_list)
    else:
        similarities = cal_similarity(feats_list, old_selection_feats) 

    cl_loss_label, cl_loss_centoids = run_kMeans(feats_list, num_centroids=args.sampling_num, Niter=200, 
                                                                  seed=args.seed, force_no_lazy_tensor=False)
    select_idx = get_selection_with_reg(feats_list, similarities, cl_loss_label, num_centroids=args.sampling_num, 
                                                          final_sample_num=args.sampling_num, iters=20, w=args.w, momentum=0.9, alpha=args.alpha)
    selectSamples = ex_trainData[0][select_idx]
    selectLabels = ex_trainData[1][select_idx]
    selectData = (selectSamples, selectLabels)

    if args.sampling_num < args.batch_size:
        batch_size = args.sampling_num
    else:
        batch_size = args.batch_size
        
    num_majority = len(np.where(selectLabels == 0)[0])
    num_minority = len(np.where(selectLabels == 1)[0])

    entropy_list = norm(calculate_entropy(probabilities[select_idx]))

    SelectData_loader = dataset.get_trainDataloader(selectData, batch_size=batch_size,
                                                    shuffle=True, num_workers=8, drop_last=True, entropy_list=entropy_list)
    
    return SelectData_loader, (num_majority, num_minority), selectData, select_idx


def loss_informative(args, ex_trainData, loss_list, probabilities, theta=0.5):
    loss_list = torch.as_tensor(loss_list)
    entropy = calculate_entropy(probabilities)
    feats_idx = np.arange(len(ex_trainData[0]))
    T = loss_list.mean() 
    min_budget = int(theta * args.sampling_num)
    maj_budget = args.sampling_num - min_budget
    feats_idx_min = feats_idx[loss_list > T]
    feats_idx_maj = feats_idx[loss_list < T]

    entropy_min_idx = entropy[feats_idx_min].argsort()
    entropy_maj_idx = entropy[feats_idx_maj].argsort()

    if len(entropy_min_idx) < min_budget:
        min_budget = len(entropy_min_idx)
        maj_budget = args.sampling_num - min_budget

    min_select_idx = entropy_min_idx[-min_budget:]
    maj_select_idx = entropy_maj_idx[-maj_budget:]

    select_idx = np.concatenate((min_select_idx, maj_select_idx), axis=0)
    selectSamples = ex_trainData[0][select_idx]
    selectLabels = ex_trainData[1][select_idx]

    selectData = (selectSamples, selectLabels)

    if args.sampling_num < args.batch_size:
        batch_size = args.sampling_num
    else:
        batch_size = args.batch_size

    num_majority = len(np.where(selectLabels == 0)[0])
    num_minority = len(np.where(selectLabels == 1)[0])

    entropy_list = norm(calculate_entropy(probabilities[select_idx]))

    SelectData_loader = dataset.get_trainDataloader(selectData, batch_size=batch_size,
                                                    shuffle=True, num_workers=8, drop_last=True, entropy_list=entropy_list)
    
    return SelectData_loader, (num_majority, num_minority), selectData


def loss_diversity(args, ex_trainData, feats_list, loss_list, probabilities, epoch, old_feats_idx, theta=0.7):
    # prepare
    # feats_gpu = torch.tensor(feats_list).cuda()
    feats_idx = np.arange(len(feats_list))
    loss_list = torch.as_tensor(loss_list)
    old_selection_feats = feats_list[old_feats_idx]
    T = loss_list.mean() 
    loss_budget = int(theta * args.sampling_num)
    maj_budget = args.sampling_num - loss_budget
    # minority selection
    # find the index of the data with loss larger than theta*T
    feats_idx_with_large_loss = feats_idx[loss_list > T]
    feats_with_large_loss = feats_list[feats_idx_with_large_loss]
    if epoch == 1:
        q_score = torch.zeros(len(feats_with_large_loss)).cuda()
    else:
        similarities = cal_similarity(feats_with_large_loss, old_selection_feats)
        q_score = norm(similarities)

    num_centroids, final_sample_num = get_sample_info(loss_budget)
    cl_loss_label, cl_loss_centoids = run_kMeans(feats_with_large_loss, num_centroids=final_sample_num, Niter=200, seed=args.seed, force_no_lazy_tensor=False)
    S_loss_idxs = get_selection_with_reg(feats_with_large_loss, q_score, cl_loss_label, num_centroids=num_centroids, final_sample_num=final_sample_num, 
                                         iters=20, w=args.w, momentum=0.9, alpha=args.alpha)
    S_loss_idxs_in_all = feats_idx_with_large_loss[S_loss_idxs]

    # representative majority data
    feats_idx_maj = feats_idx[loss_list < T]
    feats_maj = feats_list[feats_idx_maj]
    if epoch == 1:
        q_score = cal_similarity(feats_maj, feats_maj)
    else:
        similarities = cal_similarity(feats_maj, old_selection_feats)
        q_score = similarities

    num_centroids, final_sample_num = get_sample_info(maj_budget) 
    cl_maj_label, cl_maj_centoids = run_kMeans(feats_maj.cpu(), num_centroids=final_sample_num, Niter=200, seed=args.seed, force_no_lazy_tensor=False)
    S_maj_idxs = get_selection_with_reg(feats_maj, q_score, cl_maj_label, num_centroids=num_centroids, final_sample_num=final_sample_num, 
                                        iters=20, w=args.w, momentum=0.9, alpha=args.alpha)
    S_maj_idxs_in_all = feats_idx_maj[S_maj_idxs]

    total_index = np.concatenate((S_loss_idxs_in_all, S_maj_idxs_in_all), axis=0)

    selectSamples = ex_trainData[0][total_index]
    selectLabels = ex_trainData[1][total_index]
    entropy_list = norm(calculate_entropy(probabilities[total_index]))

    num_majority = len(np.where(selectLabels == 0)[0])
    num_minority = len(np.where(selectLabels == 1)[0])

    selectData = (selectSamples, selectLabels)
    SelectData_loader = dataset.get_trainDataloader(selectData, batch_size=args.batch_size, 
                                                    shuffle=True, num_workers=8, drop_last=True, entropy_list=entropy_list)

    return SelectData_loader, (num_majority, num_minority), selectData, total_index

def infor_diversity(args, ex_trainData, feats_list, probabilities, epoch, old_feats_idx):
    entropy = calculate_entropy(probabilities)
    old_selection_feats = feats_list[old_feats_idx]

    if epoch == 1:
        q_score = entropy
    else:
        similarities = cal_similarity(feats_list, old_selection_feats)
        q_score = q(entropy, similarities, gamma=args.gamma) # 

    cl_loss_label, cl_loss_centoids = run_kMeans(feats_list, num_centroids=args.sampling_num, Niter=200, 
                                                                  seed=args.seed, force_no_lazy_tensor=False)
    select_idx = get_selection_with_reg(feats_list, q_score, cl_loss_label, num_centroids=args.sampling_num, 
                                                          final_sample_num=args.sampling_num, iters=20, w=args.w, momentum=0.9, alpha=args.alpha)
    selectSamples = ex_trainData[0][select_idx]
    selectLabels = ex_trainData[1][select_idx]
    selectData = (selectSamples, selectLabels)

    if args.sampling_num < args.batch_size:
        batch_size = args.sampling_num
    else:
        batch_size = args.batch_size
        
    num_majority = len(np.where(selectLabels == 0)[0])
    num_minority = len(np.where(selectLabels == 1)[0])

    entropy_list = norm(calculate_entropy(probabilities[select_idx]))

    SelectData_loader = dataset.get_trainDataloader(selectData, batch_size=batch_size,
                                                    shuffle=True, num_workers=8, drop_last=True, entropy_list=entropy_list)
    
    return SelectData_loader, (num_majority, num_minority), selectData



