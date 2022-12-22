

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
- `model`: vgg # 模型
- `fed`: fedCDP # 算法
- `optim`: sgd # 优化

- `device`: 'cuda'

- `n_round`: 200 # 训练轮数
- `E`: 5 # Local epoch
- `K`: 100 # Number of clients
- `C`: 0.1 # Percentage of clients participating in training in each round
- `B`: 64  # Batchsize
- `iid`: False  # 数据是否独立同分布
- `dataSet`: mnist  # 数据集
