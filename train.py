import copy
import os
import logging
import pickle

from src.utils.compute_flops import print_model_param_nums, count_model_param_flops

log = logging.getLogger(__name__)

import torch
from torch.utils.tensorboard import SummaryWriter
import hydra
from omegaconf import DictConfig
from torch.optim import *
from  src.models.vgg import VGG
from src.models import *
from src.fed_zoo import *
from src.utils import *
from src.utils.data import get_mnist_data


@hydra.main(config_path="./config/config.yaml", strict=True)
def main(cfg: DictConfig):
    #os.chdir('D:\code\ecnu\py\\fed')
    seed_everything(cfg.seed)
    log.info("\n" + cfg.pretty())
    model = eval(cfg.model.classname)(**cfg.model.args)
    cuda = torch.cuda.is_available()
    gpu_num = torch.cuda.device_count()
    if(cuda):model = torch.nn.DataParallel(model, list(range(gpu_num)))
    else :model = torch.nn.DataParallel(model, list(range(gpu_num)))

    if cfg.resume:
        if os.path.isfile(cfg.resume):
            print("=> loading checkpoint '{}'".format(cfg.resume))
            model.module.load_state_dict(torch.load(cfg.resume).module.state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(cfg.resume))
    print(model.module._modules['feature'])
    writer = SummaryWriter(log_dir=os.path.join(cfg.savedir, "tf"))
    t=copy.deepcopy(model)
    params_before = print_model_param_nums(t)
    flops_before,npf = count_model_param_flops(t, 32)
    if cfg.resume:
        if os.path.isfile(cfg.resume):
            print("=> loading checkpoint '{}'".format(cfg.resume))
            model.module.load_state_dict(torch.load(cfg.resume).module.state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(cfg.resume))

    federater = eval(cfg.fed.classname)(model=model,
                                        optimizer=eval(cfg.optim.classname),
                                        optimizer_args=cfg.optim.args,
                                        num_clients=cfg.K,
                                        batchsize=cfg.B,
                                        fraction=cfg.C,
                                        local_epoch=cfg.E,
                                        iid=cfg.iid,
                                        device=cfg.device,
                                        writer=writer,
                                        dataSet=cfg.dataSet,params_before=params_before,flops_before=flops_before,resumeEpoch=0,dir=cfg.savedir,layer_flops=npf)

    federater.fit(cfg.n_round)

    with open(os.path.join(cfg.savedir, "result.pkl"), "wb") as f:
        pickle.dump(federater.result, f)


if __name__ == "__main__":
    main()
