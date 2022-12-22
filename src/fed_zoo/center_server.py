import copy
import math
from collections import OrderedDict

import numpy
import numpy as np
import torch
from torch import tensor

from src.prune.prune import DPSS, sparse_coefficent
from src.utils.compute_flops import print_model_param_nums, count_model_param_flops


class CenterServer:

    def __init__(self, model, dataloader, device="cpu",params_before=0,flops_before=0,layer_flops=[]):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.flag = 0
        self.pr=0.74
        self.epoch=200
        self.epochs=200
        self.lambda21=20
        self.params_before=582602
        self.flops_before=41402368
        self.params_before=params_before
        self.flops_before=flops_before
        self.lr=0.001
        self.SimilarityArray=np.array([])
        self.sameArray=np.array([])
        self.pruneLayer=[]
        self.pruned_params=0
        self.pruned_flops=0
        self.layer_flops=layer_flops
        self.pruneRatio = []
        self.weightMean=[]
        self.importanceMean = []
    def aggregation(self):
        raise NotImplementedError

    def send_model(self):
        return copy.deepcopy(self.model)

    def validation(self):
        raise NotImplementedError


class FedAvgCenterServer(CenterServer):
    def __init__(self, model, dataloader, device="cpu",params_before=0,flops_before=0,layer_flops=[]):
        super().__init__(model, dataloader, device,params_before,flops_before,layer_flops)


    def aggregation(self, clients, aggregation_weights,epoch,select):
        update_state = OrderedDict()
        for key in self.model.state_dict().keys():
            for k, i in enumerate(select):
                #client.model.to('cpu')
                client=clients[i]
                local_state = client.model.state_dict()

                if k == 0:

                    update_state[key] = local_state[key]/(len(select))
                else:
                    update_state[key] += local_state[key]/(len(select))

        self.model.load_state_dict(update_state)


        if(self.flag==1):
            return  self.pruned_params,self.pruned_flops
        self.model.to(self.device)

        method = DPSS(self.model, self.lambda21, 0.74,'vggsmall',self.layer_flops)

        method.flag=self.flag

        grad=None

        for i, layer_index in enumerate(method.list1):
            for k, i in enumerate(select):
                client = clients[i]
                if(k==0):
                    grad=client.model.module._modules['feature'][layer_index].weight.grad.data.clone().detach().to(self.device)
                else:
                    grad+=client.model.module._modules['feature'][layer_index].weight.grad.data.clone().detach().to(self.device)
            method.gradList.append(grad)
        for k, i in enumerate(select):
            client = clients[i]
            if (k == 0):
                grad = client.model.module._modules['classifier1'].weight.grad.data.clone().detach().to(self.device)
            else:
                grad += client.model.module._modules['classifier1'].weight.grad.data.clone().detach().to(self.device)
        method.gradList.append(grad)

        s_e = sparse_coefficent(epoch,self.epochs)
        method.sparse_coefficent_value(s_e)
        method.serverModel_weight_update()
        self.pruneLayer=method.pruneLayer
        # if(epoch==1):
        #     self.SimilarityArray=method.SimilarityArray
        #     self.sameArray=method.SimilarityArray
        # else:
        #     similar = self.getSimilar(self.SimilarityArray, method.SimilarityArray)
        #
        #     self.SimilarityArray=method.SimilarityArray
        #     self.sameArray*=method.SimilarityArray
        #     print("serverSimilar:",similar,"same:",self.sameArray.sum()/len(self.sameArray))
        if((epoch-1)%10==0):
            self.pruneRatio.append(method.printf)
            file=np.array(self.pruneRatio)
            print(file)
            np.save('D:\code\ecnu\py\\fed-main\output\plot\PruneRatio{}'.format(epoch), file)
            self.weightMean.append(method.weightMean)
            file = np.array(self.weightMean)
            print(file)
            np.save('D:\code\ecnu\py\\fed-main\output\plot\WeightMean{}'.format(epoch), file)
            self.importanceMean.append(method.importanceMean)
            file = np.array(self.importanceMean)
            print(file)
            np.save('D:\code\ecnu\py\\fed-main\output\plot\ImportanceMean{}'.format(epoch), file)
        self.model = method.model
        a=self.model.state_dict()
        model_pruning = copy.deepcopy(self.model)




        method1 = DPSS(model_pruning, self.lambda21, self.pr,'vggsmall')

        method1.channel_prune()
        params_pruning = print_model_param_nums(method1.model.module)
        flops_pruning,npf =count_model_param_flops(method1.model.module, 32)
        pruned_params = 1 - params_pruning / self.params_before
        pruned_flops = 1 - flops_pruning / self.flops_before
        print("pruned_params:", pruned_params, " pruned_flops:", pruned_flops,"fi:",method.s_c_v)
        if(pruned_params>=0.95):
            self.flag=1
            method.channel_prune()
            self.model=method.model
        print(str(method1.layer_sparsity_ratio))
        return pruned_params,pruned_flops
    def validation(self, loss_fn,model):
        model.to(self.device)
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for img, target in self.dataloader:
                img=img.to(torch.float32)
                img = img.to(self.device)
                target = target.to(self.device)
                logits = model(img)
                test_loss += loss_fn(logits, target.long()).item()
                pred = logits.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        model.to("cpu")
        test_loss = test_loss / len(self.dataloader)
        accuracy = 100. * correct / len(self.dataloader.dataset)

        return test_loss, accuracy
    def Similarity(self, clients,select):
        similar=numpy.ones((10,10))
        s=np.ones(clients[select[0]].SimilarityArray.shape)
        #sum=1.0
        for i in range(len(select)):
            s*=clients[select[i]].SimilarityArray
            #sum*=clients[select[i]].SimilarityArray.sum()
            for j in range(len(select)):
                similar[i][j]=self.getSimilar(clients[select[i]].SimilarityArray,clients[select[j]].SimilarityArray)
        print(similar)
        print(s.sum()/len(s))
    def getSimilar(self,a,b):
        c = a * b
        return c.sum() / math.sqrt(a.sum() * (b.sum()))