# A Progressive Sampling Method for Dual-Node Imbalanced Learning with Restricted Data Access
International Conference on Data Mining (ICDM), 2023 at Shanghai

## Requirements
Install the required packages:
```
python == 3.10
pip install -r requirements.txt
```


## Datasets
- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html): the dataset can be download into the folder `./data/cifar10` automatically when running the `main_cifar10` if the dataset is not existing. 
- [KMNIST](https://github.com/rois-codh/kmnist): run `./data/kmnist/download.sh` to download the dataset into folder `./data/kmnist`.

## Run codes
For CIFAR10 dataset,
```
cd ./command
bash kmeans_select.sh
```
For KMNIST dataset,
```
cd ./command
bash kmeans_select_k.sh
```

## Acknowledgements
This work uses code fragments from many projects. We acknowledge the authors of the following projects we referred in our codes. 
- [pykeops](https://www.kernel-operations.io/keops/_auto_tutorials/knn/plot_knn_torch.html) for kNN and k-Means.
- [USL](https://github.com/TonyLianLong/UnsupervisedSelectiveLabeling/tree/main) for parts of partitioned_kNN and get_selection_with_reg. 
