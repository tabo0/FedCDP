import copy
import logging
import os
import pickle

from torchvision import datasets
from torchvision.transforms import transforms

from src.prune.prune import adjust_learning_rate, save_checkpoint

log = logging.getLogger(__name__)

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from src.fed_zoo.base import FedBase
from src.fed_zoo.client import FedAvgClient as Client
from src.fed_zoo.center_server import FedAvgCenterServer as CenterServer


class FedCDP(FedBase):
    def __init__(self,
                 model,
                 optimizer,
                 optimizer_args,
                 num_clients=200,
                 batchsize=50,
                 fraction=1,
                 local_epoch=1,
                 iid=False,
                 device="cpu",
                 writer=None,
                 dataSet="cifar10",params_before=0,flops_before=0,resumeEpoch=0,dir='',layer_flops=[]):
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args
        self.resumeEpoch=resumeEpoch
        self.num_clients = num_clients  # K
        self.batchsize = batchsize  # B
        self.fraction = fraction  # C, 0 < C <= 1
        self.local_epoch = local_epoch  # E
        self.lr=optimizer_args.lr
        self.iid=iid
        self.lambda21=20
        self.dir=dir
        if(dataSet=='mnist'):local_datasets, test_dataset ,testlocal_datasets = self.create_mnist_datasets(
            num_clients, shard_size=250, iid=iid)
        elif(dataSet=='cifar10'):local_datasets, test_dataset ,testlocal_datasets= self.create_cifar10_datasets(
            num_clients, shard_size=250, iid=iid)
        elif(dataSet=='cifar100'):local_datasets, test_dataset ,testlocal_datasets= self.create_cifar100_datasets(
            num_clients, shard_size=250, iid=iid)
        local_dataloaders = [
            DataLoader(dataset,
                       num_workers=0,
                       batch_size=batchsize,
                       shuffle=True) for dataset in local_datasets
        ]
        testlocal_dataloaders = [
            DataLoader(dataset,
                       num_workers=0,
                       batch_size=batchsize,
                       shuffle=False) for dataset in testlocal_datasets
        ]
        model.to('cpu')
        self.clients = [
            Client(k, local_dataloaders[k],testlocal_dataloaders[k], device,params_before,flops_before) for k in range(num_clients)
        ]
        self.total_data_size = sum([len(client) for client in self.clients])
        self.aggregation_weights = [
            len(client) / self.total_data_size for client in self.clients
        ]

        test_dataloader = DataLoader(test_dataset,
                                     num_workers=0,
                                     batch_size=batchsize)
        self.center_server = CenterServer(model, test_dataloader, device,params_before,flops_before,layer_flops)

        self.loss_fn = CrossEntropyLoss()

        self.writer = writer

        self._round = 0
        self.result = None
        self.bestAcc=np.array([0.0 for i in range(num_clients)])
        self.Acc = np.array([0.0 for i in range(num_clients)])
    def fit(self, num_round):
        #os.chdir('D:\code\ecnu\py\\fed')
        print(os.path)
        self._round = 0
        self.result = {'loss': [], 'accuracy': [],'pruned_params':[],'pruned_flops':[],'avgAcc':[],'bestAvgAcc':[]}
        self.flag=0
        #self.validation_step()
        #self.send_model([0])

        if(self.resumeEpoch>90) :
            #self.lambda21*=0.5
            self.lr*=0.1
        for t in range(num_round):
            epoch_step = [90, 0.75 * 200,250]
            # epoch_step = [120, 160]

            if t+1+self.resumeEpoch in epoch_step:
                # args.lr *= 0.2
                self.lr *= 0.1
                #self.lambda21*=0.5
            self.center_server.lambda21=self.lambda21
            for i in range(self.num_clients):
                self.clients[i].epoch=t+self.resumeEpoch
                self.clients[i].lr=self.lr
                self.clients[i].lambda21=self.lambda21
            self._round = t + 1
            sample=self.train_step()
            if(self.iid):self.validation_step()
            else :self.Noniidvalidation_step(sample)
            if (t == 80):
                save_checkpoint(self.center_server.model, os.path.join(self.dir, 'epoch80'), is_best=False)
            if(t%10==0):
                save_checkpoint(self.center_server.model, os.path.join(self.dir, 'epoch'), is_best=False)
                with open(os.path.join(self.dir, "result.pkl"), "wb") as f:
                    pickle.dump(self.result, f)
        #self.test(1)
    def test(self,num_round):
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data/cifar10', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.Pad(4),
                                 transforms.RandomCrop(32),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.4914, 0.4824, 0.4467], [0.2471, 0.2435, 0.2616])
                             ])),
        batch_size = 256, shuffle = True)
        self.clients[0].dataloader=train_loader
        for i in range(100):
            self.clients[0].client_update(self.optimizer, self.optimizer_args,
                                              1, self.loss_fn)
            test_loss, accuracy=self.center_server.validation(self.loss_fn,self.clients[0].model)
            log.info(
                f"[client: {0: 04}] Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%"
            )
    def train_step(self):

        n_sample = max(int(self.fraction * self.num_clients), 1)
        sample_set = np.random.randint(0, self.num_clients, n_sample)
        self.send_model(sample_set)
        #sample_set=[i for i in range(self.num_clients)]
        #self.center_server.aggregation(self.clients, self.aggregation_weights)
        for k in iter(sample_set):
            self.clients[k].client_update(self.optimizer, self.optimizer_args,
                                          self.local_epoch, self.loss_fn)
            if(self.flag==0 and self.iid):
                test_loss, accuracy=self.center_server.validation(self.loss_fn,self.clients[k].model)
                log.info(
                    f"[client: {k: 04}] Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%"
                )
                if(accuracy>11):self.flag=1
        #self.center_server.Similarity(self.clients,sample_set)
        pruned_params,pruned_flops=self.center_server.aggregation(self.clients, self.aggregation_weights,self._round+self.resumeEpoch,sample_set)
        self.result['pruned_params'].append(pruned_params)
        self.result['pruned_flops'].append(pruned_flops)
        if self.writer is not None:
            self.writer.add_scalar("val/pruned_params", pruned_params, self._round)
            self.writer.add_scalar("val/pruned_flops", pruned_flops, self._round)
        return sample_set
    def send_model(self,clients):
        self.center_server.model.to('cpu')
        for client in clients:
            self.clients[client].model = self.center_server.send_model()
            if(self._round>1):self.clients[client].pruneLayer=self.center_server.pruneLayer
            self.clients[client].flag = self.center_server.flag
        for i in range(self.num_clients):
            if(i not in clients):
                self.clients[i].model=None
    def validation_step(self):
        test_loss, accuracy = self.center_server.validation(self.loss_fn,self.center_server.model)
        log.info(
            f"[Round: {self._round: 04}] Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%"
        )
        if self.writer is not None:
            self.writer.add_scalar("val/loss", test_loss, self._round)
            self.writer.add_scalar("val/accuracy", accuracy, self._round)

        self.result['loss'].append(test_loss)
        self.result['accuracy'].append(accuracy)
    def Noniidvalidation_step(self,sample):
        test_loss, accuracy = 0.0,0.0
        loss=[]
        a=[]
        for i in sample:
            l,acc=self.clients[i].validation(self.loss_fn)
            test_loss+=l
            accuracy+=acc
            loss.append(l)
            a.append(acc)
            if(self.bestAcc[i]<acc):self.bestAcc[i]=acc
            if(self.center_server.flag==0):self.Acc[i]=acc
            else:
                if(self.Acc[i]<acc): self.Acc[i]=acc
        test_loss/=len(sample)
        accuracy/=len(sample)
        # print("loss:",loss)
        # print(" acc:",a)
        log.info(
            f"[Round: {self._round: 04}] Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%,bestAvgAcc:{self.bestAcc.mean():.4f},avgAcc:{self.Acc.mean():.4f}"
        )
        if self.writer is not None:
            self.writer.add_scalar("val/loss", test_loss, self._round)
            self.writer.add_scalar("val/accuracy", accuracy, self._round)
            self.writer.add_scalar("val/avgAcc", self.Acc.mean(), self._round)
            self.writer.add_scalar("val/bestAvgAcc", self.bestAcc.mean(), self._round)
        self.result['loss'].append(test_loss)
        self.result['accuracy'].append(accuracy)
        self.result['avgAcc'].append(self.bestAcc.mean())
        self.result['bestAvgAcc'].append(self.bestAcc.mean())
        with open("./result.pkl", "wb") as f:
            pickle.dump(self.result, f)