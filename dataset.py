import numpy as np
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as dsets
from PIL import Image
import torch

class get_Binary_Dataset(object):
    def __init__(self, traindata, testdata, min_cls, A_prop, seed):
        super(get_Binary_Dataset, self).__init__()
        np.random.seed(seed)
        self.traindata, self.traintargets  = traindata
        self.testdata, self.testtargets  = testdata
        self.A_prop = A_prop
        self.min_cls = min_cls

    def gen_min_dataset(self, data, target):
        targets_np = np.array(target, dtype=np.float32)
        min_idx = np.where(targets_np == self.min_cls)[0]
        np.random.shuffle(min_idx)
        min_data = data[min_idx]
        min_targets = np.ones(min_data.shape[0], dtype=np.float32)
        min_dataset = (min_data, min_targets)

        return min_dataset

    def gen_maj_dataset(self, data, target):
        targets_np = np.array(target, dtype=np.float32)
        maj_idx = np.where(targets_np != self.min_cls)[0]
        np.random.shuffle(maj_idx)
        maj_data = data[maj_idx]
        maj_targets = np.zeros(maj_data.shape[0], dtype=np.float32)
        maj_dataset = (maj_data, maj_targets)

        return maj_dataset

    def show_data_distribution(self, label):
        class_stat = [0 for _ in range(2)]
        train_label = label.astype(int)
        for lbl in train_label:
            class_stat[lbl] += 1
        return class_stat

    def shuffle(self, data, targets):
        idx = np.arange(data.shape[0])
        np.random.shuffle(idx)
        data = data[idx]
        targets = targets[idx]
        return data, targets


    def trainData(self, ):
        min_dataset = self.gen_min_dataset(self.traindata, self.traintargets)
        maj_dataset = self.gen_maj_dataset(self.traindata, self.traintargets)

        val_min_data, remain_min_data, val_min_y, remain_min_y = train_test_split(min_dataset[0], min_dataset[1], test_size=0.9)
        val_maj_data, remain_maj_data, val_maj_y, remain_maj_y = train_test_split(maj_dataset[0], maj_dataset[1], test_size=0.9)
        
        main_min_data, ex_min_data, main_min_y, ex_min_y = train_test_split(remain_min_data, remain_min_y, test_size=(1-self.A_prop))
        main_max_data, ex_max_data, main_max_y, ex_max_y = train_test_split(remain_maj_data, remain_maj_y, test_size=(1-self.A_prop))
        
        main_data = np.concatenate((main_min_data, main_max_data), axis=0)
        main_targets = np.concatenate((main_min_y, main_max_y), axis=0)
        ex_data = np.concatenate((ex_min_data, ex_max_data), axis=0)
        ex_targets = np.concatenate((ex_min_y, ex_max_y), axis=0)
        val_data = np.concatenate((val_min_data, val_maj_data), axis=0)
        val_targets = np.concatenate((val_min_y, val_maj_y), axis=0)

        main_data, main_targets = self.shuffle(main_data, main_targets)
        ex_data, ex_targets = self.shuffle(ex_data, ex_targets)
        val_data, val_targets = self.shuffle(val_data, val_targets)

        main_dataset = (main_data, main_targets)
        ex_dataset = (ex_data, ex_targets)
        val_dataset = (val_data, val_targets)

        print("class distribution of main dataset is {}".format(self.show_data_distribution(main_targets)))
        print("class distribution of external dataset is {}".format(self.show_data_distribution(ex_targets)))
        print("class distribution of validation dataset is {}".format(self.show_data_distribution(val_targets)))

        return main_dataset, ex_dataset, val_dataset


    def testData(self):
        min_dataset = self.gen_min_dataset(self.testdata, self.testtargets)
        maj_dataset = self.gen_maj_dataset(self.testdata, self.testtargets)
        test_data = np.concatenate((min_dataset[0], maj_dataset[0]), axis=0)
        test_targets = np.concatenate((min_dataset[1], maj_dataset[1]), axis=0)
        idx = np.arange(test_data.shape[0])
        np.random.shuffle(idx)
        test_data = test_data[idx]
        test_targets = test_targets[idx]
        testData = (test_data, test_targets)

        print("class distribution of test data is {}".format(self.show_data_distribution(test_targets)))

        return testData
    
def LOAD_CIFAR_BINARY_LT(args, root, min_cls):
    # get original data and labels
    temp_train = dsets.CIFAR10(root=root, train=True, download=True, transform=transforms.ToTensor())
    traindata, trainlabels = temp_train.data, torch.Tensor(temp_train.targets).long()
    trainData = (traindata, trainlabels)
    
    # set test dataloader
    temp_test = dsets.CIFAR10(root=root, train=False, transform=transforms.ToTensor())
    testdata, testlabels = temp_test.data, torch.Tensor(temp_test.targets).long()
    testData = (testdata, testlabels)

    Getting_dataset = get_Binary_Dataset(trainData, testData, min_cls, args.A_prop, seed=args.seed)
    
    main_trainData, ex_trainData, val_dataset = Getting_dataset.trainData()
    testData = Getting_dataset.testData()

    return main_trainData, ex_trainData, val_dataset, testData

def get_trainDataloader(data, batch_size, num_workers, drop_last=False, shuffle=True, entropy_list=None):
    imgs, labels = data
    dataset = CIFAR10_Augmentention(imgs, labels, entropy_list=entropy_list)
    ContructedDataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                                    num_workers=num_workers, pin_memory=True)
    return ContructedDataLoader

def get_testDataloader(data, batch_size, num_workers, drop_last=False, shuffle=False):
    imgs, labels = data
    dataset = CIFAR10_test(imgs,labels)
    ContructedDataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                                    num_workers=num_workers, pin_memory=True)
    return ContructedDataLoader

class CIFAR10_Augmentention(Dataset):
    def __init__(self,data, labels, entropy_list=None):
        self.data = data
        self.labels = torch.tensor(labels).long()
        self.entropy_list = entropy_list
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        # img = Image.fromarray(img.astype(np.uint8))

        imgs1= self.transform(img)
        imgs2 = self.transform(img)
        if self.entropy_list is not None:
            entropys = self.entropy_list[index]

            return imgs1, imgs2, label, entropys
        else:
            return imgs1, imgs2, label

    def __len__(self):
        return len(self.labels)

class CIFAR10_test(Dataset):
    def __init__(self,data, labels):
        self.data = data
        self.labels = torch.tensor(labels).long()
        self.test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        img = Image.fromarray(img.astype(np.uint8))
        imgs = self.test_transform(img)

        return imgs, imgs, label

    def __len__(self):
        return len(self.labels)