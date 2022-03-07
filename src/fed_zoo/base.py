import random

import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import transforms

from src.datasets.mnist import MnistLocalDataset, DatasetSplit
from src.utils.data import get_mnist_data
class FedBase:
    def create_cifar10_datasets(self,num_clients=100,
                              shard_size=300,
                              datadir="./data/cifar10",
                              iid=False):
        train_loader = datasets.CIFAR10(datadir, train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.Pad(4),
                                 transforms.RandomCrop(32),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.4914, 0.4824, 0.4467], [0.2471, 0.2435, 0.2616])
                             ]))
        val_loader = datasets.CIFAR10(datadir, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4824, 0.4467], [0.2471, 0.2435, 0.2616])
            ]))
        num_classes = 10
        input_pix = 32
        train_img, train_label, test_img, test_label=train_loader.data,np.array(train_loader.targets),val_loader.data,np.array(val_loader.targets)
        train_sorted_index = np.argsort(train_label)
        train_img = train_img[train_sorted_index]
        #train_label = train_label[train_sorted_index]

        if iid:
            random.shuffle(train_sorted_index)
            train_img = train_img[train_sorted_index]
            #train_label = train_label[train_sorted_index]

        shard_start_index = [i for i in range(0, len(train_img), shard_size)]
        random.shuffle(shard_start_index)
        print(
            f"divide data into {len(shard_start_index)} shards of size {shard_size}"
        )

        num_shards = len(shard_start_index) // num_clients
        #num_shards=1
        local_datasets = []

        testlocal_datasets = []
        test_sorted_index = np.argsort(test_label)
        test_img = test_img[test_sorted_index]
        #test_label = test_label[test_sorted_index]


        testId = []
        for i in range(10):
            testId.append(np.where(np.array(test_label)==i)[0])
        for client_id in range(num_clients):
            _index = num_shards * client_id
            img = np.concatenate([
                train_sorted_index[shard_start_index[_index +
                                            i]:shard_start_index[_index + i] +
                          shard_size] for i in range(num_shards)
            ],
                                 axis=0)

            dataSet=DatasetSplit(train_loader,img)
            local_datasets.append(dataSet)
            bug=train_label[img]
            labelSet=set(bug)
            id=np.array([], dtype='int64')
            for label in labelSet:
                iidxx=testId[label]
                id=np.concatenate((id,iidxx),axis=0)
            dataSet=DatasetSplit(val_loader,id)
            testlocal_datasets.append(dataSet)

        #test_dataset = Cifar10LocalDataset(test_img, test_label, client_id=-1)
        test_dataset=DatasetSplit(val_loader,test_sorted_index)
        return local_datasets, test_dataset,testlocal_datasets

    def create_cifar100_datasets(self, num_clients=100,
                                shard_size=300,
                                datadir="./data/cifar100",
                                iid=False):
        train_loader = datasets.CIFAR100(datadir, train=True, download=True,
                                        transform=transforms.Compose([
                                            transforms.Pad(4),
                                            transforms.RandomCrop(32),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.4914, 0.4824, 0.4467], [0.2471, 0.2435, 0.2616])
                                        ]))
        val_loader = datasets.CIFAR100(datadir, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4824, 0.4467], [0.2471, 0.2435, 0.2616])
        ]))
        num_classes = 10
        input_pix = 32
        train_img, train_label, test_img, test_label = train_loader.data, np.array(
            train_loader.targets), val_loader.data, np.array(val_loader.targets)
        train_sorted_index = np.argsort(train_label)
        train_img = train_img[train_sorted_index]
        # train_label = train_label[train_sorted_index]

        if iid:
            random.shuffle(train_sorted_index)
            train_img = train_img[train_sorted_index]
            # train_label = train_label[train_sorted_index]

        shard_start_index = [i for i in range(0, len(train_img), shard_size)]
        random.shuffle(shard_start_index)
        print(
            f"divide data into {len(shard_start_index)} shards of size {shard_size}"
        )

        num_shards = len(shard_start_index) // num_clients
        # num_shards=1
        local_datasets = []

        testlocal_datasets = []
        test_sorted_index = np.argsort(test_label)
        test_img = test_img[test_sorted_index]
        # test_label = test_label[test_sorted_index]

        testId = []
        for i in range(100):
            testId.append(np.where(np.array(test_label) == i)[0])
        for client_id in range(num_clients):
            _index = num_shards * client_id
            img = np.concatenate([
                train_sorted_index[shard_start_index[_index +
                                                     i]:shard_start_index[_index + i] +
                                                        shard_size] for i in range(num_shards)
            ],
                axis=0)

            dataSet = DatasetSplit(train_loader, img)
            local_datasets.append(dataSet)
            bug = train_label[img]
            labelSet = set(bug)
            id = np.array([], dtype='int64')
            for label in labelSet:
                iidxx = testId[label]
                id = np.concatenate((id, iidxx), axis=0)
            dataSet = DatasetSplit(val_loader, id)
            testlocal_datasets.append(dataSet)

        # test_dataset = Cifar10LocalDataset(test_img, test_label, client_id=-1)
        test_dataset = DatasetSplit(val_loader, test_sorted_index)
        return local_datasets, test_dataset, testlocal_datasets
    def create_mnist_datasets(self,num_clients=100,
                              shard_size=300,
                              datadir="./data/mnist",
                              iid=False):
        train_loader = datasets.MNIST(datadir, train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.Pad(4),
                                 transforms.RandomCrop(32),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.5], [0.5])
                             ]))
        val_loader = datasets.MNIST(datadir, train=False, transform=transforms.Compose([
            transforms.Pad(4),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ]))
        num_classes = 10
        input_pix = 32
        train_img, train_label, test_img, test_label=train_loader.data,np.array(train_loader.targets),val_loader.data,np.array(val_loader.targets)
        train_sorted_index = np.argsort(train_label)
        train_img = train_img[train_sorted_index]
        #train_label = train_label[train_sorted_index]

        if iid:
            random.shuffle(train_sorted_index)
            train_img = train_img[train_sorted_index]
            #train_label = train_label[train_sorted_index]

        shard_start_index = [i for i in range(0, len(train_img), shard_size)]
        random.shuffle(shard_start_index)
        print(
            f"divide data into {len(shard_start_index)} shards of size {shard_size}"
        )

        num_shards = len(shard_start_index) // num_clients
        #num_shards=1
        local_datasets = []

        testlocal_datasets = []
        test_sorted_index = np.argsort(test_label)
        test_img = test_img[test_sorted_index]
        #test_label = test_label[test_sorted_index]


        testId = []
        for i in range(10):
            testId.append(np.where(np.array(test_label)==i)[0])
        for client_id in range(num_clients):
            _index = num_shards * client_id
            img = np.concatenate([
                train_sorted_index[shard_start_index[_index +
                                            i]:shard_start_index[_index + i] +
                          shard_size] for i in range(num_shards)
            ],
                                 axis=0)

            dataSet=DatasetSplit(train_loader,img)
            local_datasets.append(dataSet)
            bug=train_label[img]
            labelSet=set(bug)
            id=np.array([], dtype='int64')
            for label in labelSet:
                iidxx=testId[label]
                id=np.concatenate((id,iidxx),axis=0)
            dataSet=DatasetSplit(val_loader,id)
            testlocal_datasets.append(dataSet)

        #test_dataset = Cifar10LocalDataset(test_img, test_label, client_id=-1)
        test_dataset=DatasetSplit(val_loader,test_sorted_index)
        return local_datasets, test_dataset,testlocal_datasets
    # def create_mnist_datasets(self,
    #                           num_clients=100,
    #                           shard_size=300,
    #                           datadir="./data/mnist",
    #                           iid=False):
    #     train_img, train_label, test_img, test_label = get_mnist_data(datadir)
    #
    #     train_sorted_index = np.argsort(train_label)
    #     train_img = train_img[train_sorted_index]
    #     train_label = train_label[train_sorted_index]
    #
    #     if iid:
    #         random.shuffle(train_sorted_index)
    #         train_img = train_img[train_sorted_index]
    #         train_label = train_label[train_sorted_index]
    #
    #     shard_start_index = [i for i in range(0, len(train_img), shard_size)]
    #     random.shuffle(shard_start_index)
    #     print(
    #         f"divide data into {len(shard_start_index)} shards of size {shard_size}"
    #     )
    #
    #     num_shards = len(shard_start_index) // num_clients
    #     local_datasets = []
    #     testlocal_datasets=[]
    #     test_sorted_index = np.argsort(test_label)
    #     test_img = test_img[test_sorted_index]
    #     test_label = test_label[test_sorted_index]
    #
    #     test_dataset = MnistLocalDataset(test_img, test_label, client_id=-1)
    #
    #     testId=[]
    #     for i in range(10):
    #         testId.append(np.where(np.array(test_label)==i)[0])
    #
    #     for client_id in range(num_clients):
    #         _index = num_shards * client_id
    #         img = np.concatenate([
    #             train_img[shard_start_index[_index +
    #                                         i]:shard_start_index[_index + i] +
    #                       shard_size] for i in range(num_shards)
    #         ],
    #                              axis=0)
    #
    #         label = np.concatenate([
    #             train_label[shard_start_index[_index +
    #                                           i]:shard_start_index[_index +
    #                                                                i] +
    #                         shard_size] for i in range(num_shards)
    #         ],
    #                                axis=0)
    #
    #         local_datasets.append(MnistLocalDataset(img, label, client_id))
    #
    #         labelSet=set(label)
    #         id=np.array([])
    #         for label in labelSet:
    #             iidxx=testId[label]
    #             id=np.concatenate((id,iidxx),axis=0)
    #
    #
    #
    #     return local_datasets, test_dataset,testlocal_datasets

    def train_step(self):
        raise NotImplementedError

    def validation_step(self):
        raise NotImplementedError

    def fit(self, num_round):
        raise NotImplementedError
