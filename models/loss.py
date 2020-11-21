import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def MDNLoss(pi, mu, var, y):
    expterm = -((y.expand_as(mu) - mu)*torch.reciprocal(var)).pow(2)/2
    Nd = (1.0/math.sqrt(2*math.pi))*(torch.exp(expterm) * torch.reciprocal(var))
    mdn = -torch.log((Nd * pi + 1e-5).sum(dim=1))
    #print(expterm.sum(), Nd.sum(),mdn.sum())
    return mdn.mean()

def KL(p, pi ,coef=1e-5):
    p_bar = p + 1.0
    klterm = -p_bar*torch.log(p_bar/pi + 1e-5)
    return klterm.mean()*coef
