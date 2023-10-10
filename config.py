import argparse

def str2bool(v):
    if isinstance(v,bool):
        return v
    if v == 'True':
        return True
    if v == 'False':
        return False

def get_config():
    parser = argparse.ArgumentParser(description='Binary Ctrastive learning')
    # dataset
    parser.add_argument('--im_ratio', default=0.1, type=float, help='imbalance ratio')
    parser.add_argument('--A_prop', default=0.1, type=float, help='dataseet of A proportion')
    parser.add_argument('--batch_size', default=512, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--sup_bs', default=128, type=int, help='Number of images in each mini-batch for supervised model training')
    parser.add_argument('--maj_cls', default=7, type=int, help='which majority class')
    parser.add_argument('--min_cls', default=0, type=int, help='which majority class')
    parser.add_argument('--cls_select_num', default=2, type=int, help='how many classes in the dataset')
    parser.add_argument('--test_imb', default=True, type=str2bool, help='if test is class imbalanced')

    # training
    parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--seed', default=42, type=int, help='Random seed setting')
    parser.add_argument('--temperature', default=0.3, type=float, help='Temperature used in softmax')
    parser.add_argument('--k', default=100, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--test_interval', default=50, type=int, help='Interval of testing or recording')
    parser.add_argument('--tau_max', default=0.7, type=float, help='maximum tau')
    parser.add_argument('--tau_min', default=0.2, type=float, help='minimum tau')

    #epochs
    parser.add_argument('--g_epochs', default=10, type=int, help='Number of global training epochs')
    parser.add_argument('--c_epochs', default=50, type=int, help='Contrastive learning epochs')
    parser.add_argument('--sup_epochs', default=50, type=int, help='Fully supervised learning epochs')
    parser.add_argument('--linear_epochs', default=50, type=int, help='Number of client training epochs')
    
    # Sampling
    parser.add_argument('--sampling_num', default=600, type=int, help='sampling numsber each time')
    parser.add_argument('--sampling', default=True, type=str2bool, help='if sampling')
    parser.add_argument('--selection_method', default='kmeans_select', type=str, 
                            help='samples selection methods including random selection and loss_based selection')
    parser.add_argument('--dynamic_tau', default=False, type=str2bool, help='dynamic tau')
    parser.add_argument('--alpha', default=0.5, type=float, help='kmeans alpha')
    parser.add_argument('--w', default=0.5, type=float, help='kmeans w')
    parser.add_argument('--gamma', default=1, type=float, help='kmeans gamma')
    parser.add_argument('--eta', default=0.1, type=float, help='kmeans eta')
    
    # Experiemnts
    parser.add_argument('--reason4Exp', default='debug', type=str, help='reason for Experiments')
    parser.add_argument('--hyperpara', default='contrast', type=str, help='which parameter to tune')
    parser.add_argument('--weight_scaler', default=0.7, type=float, help='the weight of old model')
    # args parse
    args = parser.parse_args()
    
    return args, parser