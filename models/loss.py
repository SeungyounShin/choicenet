import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def MDNLoss(pi, mu, var, y, coef=1):
    expterm = -((y.expand_as(mu) - mu)*torch.reciprocal(var)).pow(2)/2
    Nd = (1.0/math.sqrt(2*math.pi))*(torch.exp(expterm) * torch.reciprocal(var))
    mdn = -torch.log((Nd * pi + 1e-2).sum(dim=1))
    #print(expterm.sum(), Nd.sum(),mdn.sum())
    return coef*mdn.mean()

def KL(rho, pi ,coef=1e-5):
    _eps = 1e-2
    #rho = torch.clamp(rho,-1,1)
    rho_pos = rho + 1.0
    _kl_reg = coef*(-rho_pos*(torch.log(pi+_eps)-torch.log(rho_pos+_eps))) # (N)
    #klterm = -p_bar*torch.log(p_bar/pi + 1e-5)
    return _kl_reg.mean()
