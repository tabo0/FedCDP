

## Installation

#### Requirements
- Python 3.6.8+
- Numpy
- Pytorch 1.4.0+
- torchvision
- hydra-core=0.9.0
- tensorboard
- omegaconf=1.3.0


## Train
```shell
# example
$ python train.py n_round=100 C=0.3
```

## Usages config\config.yaml
- `model`: vgg
- `fed`: fedavg
- `optim`: sgd

- device: 'cuda'

- n_round: 200
- E: 5
- K: 100
- C: 0.1
- B: 64
- iid: False
- dataSet: mnist
- seed: 1999
- resume: False
- env: ./input
- root: ./input
- savedir: ./output/${fed.type}/${now:%Y-%m-%d_%H-%M-%S}
