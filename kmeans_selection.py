from pykeops.torch import LazyTensor
import torch
import time
from tqdm import tqdm 
import numpy as np

# Reference:
# [1] Unsupervised Selective Labeling for More Effective Semi-Supervised Learning 
#     https://github.com/TonyLianLong/UnsupervisedSelectiveLabeling
def KMeans(x, seed, K=10, Niter=10, verbose=True, force_no_lazy_tensor=False):
    """Implements Lloyd's algorithm for the Euclidean metric."""

    start = time.time()
    N, D = x.shape  # Number of samples, dimension of the ambient space
    c = x[:K, :].clone()
    c = torch.nn.functional.normalize(c, dim=1, p=2)

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    if force_no_lazy_tensor:  # For deterministic behavior
        print("No lazy tensor")
        def _LazyTensor(x): return x
        kwargs = {
        }
    else:
        _LazyTensor = LazyTensor
        kwargs = {
            "backend": "GPU"
        }
    x_i = _LazyTensor(x.view(N, 1, D))

    c_j = _LazyTensor(c.view(1, K, D))  # (1, K, D) centroids

    for _ in tqdm(range(Niter)):
        # E step: assign points to the closest cluster -------------------------
        # Perform this op in halves to avoid using up GPU memory:

        # (N, K) symbolic squared distances
        D_ij = ((x_i - c_j) ** 2).sum(-1, **kwargs)
        # D_ij = torch.mm(x_i, c_j.t())
        # Points -> Nearest cluster
        cl = D_ij.argmin(dim=1, **kwargs).long().view(-1)

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, D), x)

        # Divide by the number of points per cluster:
        Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
        c /= Ncl  # in-place division to compute the average
        # c[:] = torch.nn.functional.normalize(c, dim=1, p=2)
        

    if verbose:  # Fancy display -----------------------------------------------
        # if use_cuda:
        #     torch.cuda.synchronize()
        end = time.time()
        print(
            f"K-means for the Euclidean metric with {N:,} points in dimension {D:,}, K = {K:,}:"
        )
        print(
            "Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n".format(
                Niter, end - start, Niter, (end - start) / Niter
            )
        )

    return cl, c

def run_kMeans(feats_list, num_centroids, Niter, use_cuda=True, seed=None, force_no_lazy_tensor=False):
    # use_cuda: pre-copy data to GPU instead of getting data when running
    cluster_labels, centroids = KMeans(feats_list.cuda(), seed=seed,
                                        K=num_centroids, Niter=Niter, verbose=True, force_no_lazy_tensor=force_no_lazy_tensor)
    cluster_labels, centroids = cluster_labels.cpu(), centroids.cpu()

    return cluster_labels, centroids

def kNN(x_train, x_test, K=10):
    assert len(x_train.shape) == 2
    assert len(x_test.shape) == 2

    start = time.time()  # Benchmark:

    X_i = LazyTensor(x_test[:, None, :])  # (N1, 1, M) test set
    X_j = LazyTensor(x_train[None, :, :])  # (1, N2, M) train set
    D_ij = ((X_i - X_j) ** 2).sum(-1)  
    # (N1, N2) symbolic matrix of squared L2 distances

    # Samples <-> Dataset, (N_test, K)
    d_knn, ind_knn = D_ij.Kmin_argKmin(K, dim=1, backend="GPU")

    torch.cuda.synchronize()
    end = time.time()

    total_time = end - start
    print("Running time: {:.2f}s".format(total_time))

    return ind_knn, d_knn

def partitioned_kNN(feats_list, K=400, recompute=False, partitions_size=50, verify=False):
    suffix = "" if K == 20 else "_{}".format(K)
    if recompute:
        partitions = int(np.ceil(feats_list.shape[0] / partitions_size))

        print("Partitions:", partitions)

        # Assume the last partition has at least K elements

        ind_knns = torch.zeros(
            (feats_list.size(0), partitions * K), dtype=torch.long)
        d_knns = torch.zeros(
            (feats_list.size(0), partitions * K), dtype=torch.float)

        def get_sampled_data(ind):
            return feats_list[ind * partitions_size: (ind + 1) * partitions_size]

        for ind_i in range(partitions):  # ind_i: train dimension
            for ind_j in range(partitions):  # ind_j: test dimension
                print("Running with indices: {}, {}".format(ind_i, ind_j))
                x_train = get_sampled_data(ind_i).cuda()
                x_test = get_sampled_data(ind_j).cuda()

                ind_knn, d_knn = kNN(x_train, x_test, K=K)
                # ind_knn, d_knn: test dimension, K (indices: train dimension)
                ind_knns[ind_j * partitions_size: (ind_j + 1) * partitions_size, ind_i * K: (ind_i + 1) * K] = \
                    ind_i * partitions_size + ind_knn.cpu()
                d_knns[ind_j * partitions_size: (ind_j + 1) * partitions_size,
                       ind_i * K: (ind_i + 1) * K] = d_knn.cpu()

                del ind_knn, d_knn, x_train, x_test

        d_sorted_inds = d_knns.argsort(dim=1)
        d_selected_inds = d_sorted_inds[:, :K]
        ind_knns_selected = torch.gather(
            ind_knns, dim=1, index=d_selected_inds)
        d_knns_selected = torch.gather(d_knns, dim=1, index=d_selected_inds)
        d_knns = d_knns_selected
        ind_knns = ind_knns_selected

        del ind_knns_selected, d_knns_selected

        if verify:  # Verification
            ind_knns_target, d_knns_target = kNN(
                feats_list.cuda(), feats_list.cuda(), K=K)
            ind_knns_target = ind_knns_target.cpu()
            d_knns_target = d_knns_target.cpu()
            assert torch.all(d_knns == d_knns_target)
            # The ids may differ, but as long as the distance of the selected indices is correct:
            # assert torch.all(ind_knns != ind_knns_target)
            if not torch.all(ind_knns == ind_knns_target):
                def dist(a, b):
                    return torch.sum((a - b)**2)

                for dim1, dim2 in zip(*torch.where(ind_knns != ind_knns_target)):
                    dist1 = dist(feats_list[dim1],
                                 feats_list[ind_knns[dim1][dim2]])
                    dist2 = dist(
                        feats_list[dim1], feats_list[ind_knns_target[dim1][dim2]])
                    assert torch.isclose(
                        dist1, dist2), "{} != {}".format(dist1, dist2)
    
    return d_knns, ind_knns

def calculate_entropy(p, eps=1e-8):
    entropy = -p * torch.log(p + eps) - (1 - p) * torch.log((1 - p) + eps)
    entropy = torch.sum(entropy, dim=1)
    return entropy

def cal_similarity(feats_list, old_S):
    return torch.mm(feats_list, old_S.t()).sum(dim=1)

def norm(v):
    return (v - torch.mean(v)) / torch.std(v)

def q(importance, similarities, gamma=1):
    similarities_norm = norm(similarities)
    importance_norm = norm(importance)
    score = importance_norm - gamma * similarities_norm
    return norm(score)

def get_selection_with_reg(data, q, cluster_labels, num_centroids,
                  iters=20, final_sample_num=None, w=1, momentum=0.9, horizon_num=64, alpha=0.5, exclude_same_cluster=False, verbose=False):
    # Intuition: horizon_num = dimension * 2

    cluster_labels_cuda = cluster_labels.cuda()
    q_cuda = q.cuda()
    selection_regularizer = torch.zeros_like(q_cuda)

    data = data.cuda()
    N, D = data.shape  # Number of samples, dimension of the ambient space

    data_expanded_lazy = LazyTensor(data.view(N, 1, D))

    for iter_ind in tqdm(range(iters)):
        selected_inds = []
        if verbose:
            print("Computing selected ids")
            print("selection_regularizer", selection_regularizer)
        while len(selected_inds) != final_sample_num:
            for cls_ind in range(num_centroids):
                if len(selected_inds) == final_sample_num:
                    break
                match_arr = cluster_labels_cuda == cls_ind
                match = torch.where(match_arr)[0]
                if len(match) == 0:
                    continue

                # scores in the selection process

                # No prior:
                # scores = 1 / neighbors_dist[match_arr]

                scores = (1-w) * q_cuda[match_arr] - w * \
                    selection_regularizer[match_arr]
                if iter_ind != 0 and cls_ind == 0 and verbose:
                    print("original score:", (1 / q_cuda[match_arr]).mean(),
                        "regularizer adjustment:", (w * selection_regularizer[match_arr]).mean())
                min_dist_ind = scores.argmax()
                selected_inds.append(match[min_dist_ind].item())

        selected_inds = torch.tensor(selected_inds)

        if iter_ind < iters - 1:  # Not the last iteration
            if verbose:
                print("Updating selection regularizer")

            selected_data = data[selected_inds]

            if not exclude_same_cluster:
                # This is square distances: (N_full, N_selected)
                # data: (N_full, 1, dim)
                # selected_data: (1, N_selected, dim)
                new_selection_regularizer = (
                    (data_expanded_lazy - selected_data[None, :, :]) ** 2).sum(dim=-1)
                new_selection_regularizer = new_selection_regularizer.Kmin(
                    horizon_num, dim=1)

                if verbose:
                    print("new_selection_regularizer shape:",
                        new_selection_regularizer.shape)
                    print("Max:", new_selection_regularizer.max())
                    print("Mean:", new_selection_regularizer.mean())

                # Distance to oneself should be ignored
                new_selection_regularizer[new_selection_regularizer == 0] = 1e10
            else:
                # This is square distances: (N_full, N_selected)
                # data: (N_full, 1, dim)
                # selected_data: (1, N_selected, dim)

                # We take the horizon_num samples with the min distance to the other centroids
                new_selection_regularizer = (
                    (data_expanded_lazy - selected_data[None, :, :]) ** 2).sum(dim=-1)
                # indices within selected data
                new_selection_regularizer, selected_data_ind = new_selection_regularizer.Kmin_argKmin(horizon_num,
                                                                                                    dim=1, backend="GPU")

                if verbose:
                    print("new_selection_regularizer shape:",
                        new_selection_regularizer.shape)
                    print("Max:", new_selection_regularizer.max())
                    print("Mean:", new_selection_regularizer.mean())

                same_cluster_selected_data_ind_mask = (
                    selected_data_ind == cluster_labels_cuda.view((-1, 1))).float()
                # It's true that if cluster is not in the list, some instances will have one more regularizer item, but this is a small contribution.
                new_selection_regularizer = (1 - same_cluster_selected_data_ind_mask) * \
                    new_selection_regularizer + same_cluster_selected_data_ind_mask * 1e10
                selection_regularizer = norm(selection_regularizer)
            
                assert not torch.any(new_selection_regularizer == 0), "{}".format(
                    torch.where(new_selection_regularizer == 0))

            if verbose:
                print("Min:", new_selection_regularizer.min())

            # selection_regularizer: N_full
            if alpha != 1:
                new_selection_regularizer = (
                    1 / new_selection_regularizer ** alpha).sum(dim=1)
            else:
                new_selection_regularizer = (
                    1 / new_selection_regularizer).sum(dim=1)

            selection_regularizer = selection_regularizer * \
                momentum + new_selection_regularizer * (1 - momentum)
            selection_regularizer = norm(selection_regularizer)
    
    del cluster_labels_cuda
    del q_cuda
    del data

    assert len(selected_inds) == final_sample_num
    return selected_inds.numpy()

def get_sample_info(chosen_sample_num):
    num_centroids = chosen_sample_num
    final_sample_num = chosen_sample_num

    # We get one more centroid to take empty clusters into account
    if chosen_sample_num == 1500:
        num_centroids = 1501
        final_sample_num = 1500

    return num_centroids, final_sample_num

def kmeans_select_data(args, ex_trainData, feats_list, loss_list, probabilities, epoch, old_feats_idx, theta=0.7):
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
    importance_loss = calculate_entropy(probabilities[feats_idx_with_large_loss])
    if epoch == 1:
        q_score = importance_loss
    else:
        similarities = cal_similarity(feats_with_large_loss, old_selection_feats)
        q_score = q(importance_loss, similarities, gamma=args.gamma) # 

    num_centroids, final_sample_num = get_sample_info(loss_budget)   
    cl_loss_label, cl_loss_centoids = run_kMeans(feats_with_large_loss, num_centroids=final_sample_num, Niter=200, seed=args.seed, force_no_lazy_tensor=False)
    S_loss_idxs = get_selection_with_reg(feats_with_large_loss, q_score, cl_loss_label, num_centroids=num_centroids, final_sample_num=final_sample_num, 
                                         iters=20, w=args.w, momentum=0.9, alpha=args.alpha)
    S_loss_idxs_in_all = feats_idx_with_large_loss[S_loss_idxs]

    # representative majority data
    feats_idx_maj = feats_idx[loss_list < T]
    feats_maj = feats_list[feats_idx_maj]
    importance_maj = calculate_entropy(probabilities[feats_idx_maj])
    if epoch == 1:
        q_score = importance_maj
    else:
        similarities = cal_similarity(feats_maj, old_selection_feats)
        q_score = q(importance_maj, similarities, gamma=args.gamma)#

    num_centroids, final_sample_num = get_sample_info(maj_budget) 
    cl_maj_label, _ = run_kMeans(feats_maj.cpu(), num_centroids=final_sample_num, Niter=200, seed=args.seed, force_no_lazy_tensor=False)
    S_maj_idxs = get_selection_with_reg(feats_maj, q_score, cl_maj_label, num_centroids=num_centroids, final_sample_num=final_sample_num, 
                                        iters=20, w=args.w, momentum=0.9, alpha=args.alpha)
    S_maj_idxs_in_all = feats_idx_maj[S_maj_idxs]

    total_index = np.concatenate((S_loss_idxs_in_all, S_maj_idxs_in_all), axis=0)

    selectSamples = ex_trainData[0][total_index]
    selectLabels = ex_trainData[1][total_index]
    entropy_list = norm(calculate_entropy(probabilities[total_index]))
    torch.save(entropy_list, "entropy_list.pt")

    num_majority = len(np.where(selectLabels == 0)[0])
    num_minority = len(np.where(selectLabels == 1)[0])

    selectData = (selectSamples, selectLabels)
    # SelectData_loader = dataset.get_trainDataloader(selectData, batch_size=args.batch_size, 
    #                                                 shuffle=True, num_workers=8, drop_last=True, entropy_list=entropy_list)

    return entropy_list, (num_majority, num_minority), selectData, total_index

