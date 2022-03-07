import copy
import math
import numpy
import numpy as np
import torch

from src.models import CNN
from src.models.vgg import VGG


def te():
    a=numpy.array([0.4, 0.81, 0.71, 0.9, 0.72, 0.61, 0.47, 0.15, 0.05, 0.05,0.05, 0.05, 0.05])
    l,r=copy.deepcopy(a),numpy.ones(13)
    while(r-l).mean()>0.002:
        mid=(l+r)/2
        print(mid)
        r=mid
def sparse_coefficent(epoch, epochs):
    epoch_step = 0.6*epochs
    s_e = 1.
    if epoch < epoch_step:
        s_e = 1 / (1 + math.exp(15 - 30 * (epoch + 1) / epoch_step))
    return s_e
if __name__=="__main__":
    bestAcc=4
    bestAcc/=2
    print(bestAcc)