import copy

import numpy as np
import torch
from torch import nn

from src.prune.prune import DPSS, sparse_coefficent, adjust_learning_rate
from src.utils import utils
from src.utils.compute_flops import print_model_param_nums, count_model_param_flops

class Client:
    def __init__(self, client_id, dataloader,testloader, device='cpu',params_before=0,flops_before=0):
        self.client_id = client_id
        self.dataloader = dataloader
        self.testloader=testloader
        self.device = device
        self.pr=0.74
        self.epoch=200
        self.epochs=200
        self.lambda21=1
        self.params_before=582602
        self.flops_before=41402368
        self.flag=0
        self.params_before=params_before
        self.flops_before=flops_before
        self.lr=3e-2
        self.SimilarityArray=np.array([])
        self.pruneLayer=[]
    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, model):
        self.__model = model

    def client_update(self, optimizer, optimizer_args, local_epoch, loss_fn):
        raise NotImplementedError

    def __len__(self):
        return len(self.dataloader.dataset)


class FedAvgClient(Client):
    def client_update(self, optimizer, optimizer_args, local_epoch, loss_fn):
        self.model.train()
        self.model.to(self.device)
        optimizer = optimizer(self.model.parameters(), **optimizer_args)

        method = DPSS(self.model, self.lambda21, self.pr,"vggsmall")
        s_e=1
        s_e = sparse_coefficent(self.epoch,self.epochs)
        method.sparse_coefficent_value(s_e)
        method.flag=self.flag

        adjust_learning_rate(optimizer, self.epoch,self.lr)
        train_loss=0

        for i in range(local_epoch):
            for img, target in self.dataloader:
                img=img.to(torch.float32)
                img = img.to(self.device)
                target = target.to(self.device)
                self.model.zero_grad()
                optimizer.zero_grad()
                logits = self.model(img)
                loss = loss_fn(logits, target)
                train_loss+=loss.item()
                loss.backward()
                #nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5, norm_type=2)
                optimizer.step()
                #a = self.model.state_dict()
                # if(self.epoch>=1and len(self.pruneLayer)>0and self.flag==0):
                #     method.pruneLayer=self.pruneLayer
                #     method.model = self.model
                #     method.SimilarityArray=np.array([])
                #     method.model_weight_update()
                #     self.model = method.model
        self.model.to("cpu")
        self.SimilarityArray=method.SimilarityArray
        a=self.model.state_dict()
        # params_pruning = print_model_param_nums(method.model.module)
        # flops_pruning =count_model_param_flops(method.model.module, 32)
        # pruned_params = 1 - params_pruning / self.params_before
        # pruned_flops = 1 - flops_pruning / self.flops_before
        #print("pruned_params:",pruned_params," pruned_flops:",pruned_flops,"loss:",train_loss/(local_epoch*len(self.dataloader)),"s_e:",s_e)
        print( "loss:",train_loss / (local_epoch * len(self.dataloader)),"lr:",self.lr)
        # if(pruned_params>=0.74):self.flag=1
        # model_pruning = copy.deepcopy(self.model)
        # method1 = DPSS(model_pruning, self.lambda21, self.pr)
        # # method1.adjust_scale_coe(next_ratio)
        # method1.channel_prune()
        # print(str(method1.layer_sparsity_ratio))
    def validation(self, loss_fn):
        self.model.to(self.device)
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for img, target in self.testloader:
                img=img.to(torch.float32)
                img = img.to(self.device)
                target = target.to(self.device)
                logits = self.model(img)
                test_loss += loss_fn(logits, target.long()).item()
                pred = logits.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        self.model.to("cpu")
        test_loss = test_loss / len(self.testloader)
        accuracy = 100. * correct / len(self.testloader.dataset)

        return test_loss, accuracy