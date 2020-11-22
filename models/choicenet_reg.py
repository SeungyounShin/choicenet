import torch
import torch.nn as nn
import torch.nn.functional as F
from models.loss import MDNLoss,KL
import numpy as np
from torch.autograd import Variable

class ChoiceNetRegression(nn.Module):
    def __init__(self,xdim=1, ydim=1,feat_dim=128, k_mix = 5,
                 hdims=[64,64], base = None):
        super(ChoiceNetRegression, self).__init__()
        self.feat_dim = feat_dim
        self.xdim= xdim
        self.ydim = ydim
        self.k_mix = k_mix #from abalation study
        # base network
        self.h = self.build_base(hdims, base=base)
        self.Q = hdims[-1]
        self.tau_inv = 1e-2
        # cholesky block
        self.rho = nn.Sequential(nn.Linear(in_features=self.Q, out_features=self.k_mix),
                                 nn.BatchNorm1d(self.k_mix),
                                 nn.Sigmoid())
        self.pi = nn.Sequential(nn.Linear(in_features=self.Q, out_features=self.k_mix),
                                nn.BatchNorm1d(self.k_mix),
                                nn.ReLU(inplace=True),
                                 nn.Softmax(dim=1))
        self.varOut = nn.Sequential(nn.Linear(in_features=self.Q, out_features=self.ydim),
                                    nn.BatchNorm1d(self.ydim),
                                    nn.ReLU(inplace=True),)
        self.muW = nn.Parameter(torch.zeros(self.Q,ydim))
        self.SigmaW = nn.Parameter(torch.ones(self.Q,ydim))
        self.muZ = torch.zeros(self.Q,ydim)
        self.SigmaZ = torch.zeros(self.Q,ydim)
        nn.init.normal_(self.muW, std = 0.1)

    def build_base(self, hdims, base=None):
        if base is None:
            layers = list()
            hdim = self.xdim
            for idx,_ in enumerate(hdims):
                layers.append(nn.Linear(hdim, _))
                layers.append(nn.BatchNorm1d(_))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(0.5))
                hdim = _
            return nn.Sequential(*layers)
        else:
            return base

    def forward(self,x):
        N = x.shape[0]
        #base network
        feat = self.h(x)
        #correlation
        p = self.rho(feat)
        ixs = torch.arange(self.k_mix, dtype=torch.int64)
        p = torch.where(ixs[None,:] == 0, torch.tensor(1.), p)
        #print(p)

        #p_slice = p_tmp[:,0:1]*Variable(torch.tensor([0.]),requires_grad=True) + Variable(torch.tensor([1.]),requires_grad=True)
        #p = torch.cat([p_slice, p_tmp[:,1:]],axis=1)

        #p = torch.cat([p_tmp[:,0:1]*0.0+1., p_tmp[:,1:]],axis=1)
        pi = self.pi(feat) # [N,K]
        #reparametrization
        muW_ = self.muW.repeat(N,1,1)
        sigmaW_ = self.SigmaW.repeat(N,1,1)
        muZ_ = self.muZ.repeat(N,1,1)
        sigmaZ_ = self.SigmaZ.repeat(N,1,1)
        sampler = list()
        for k in range(self.k_mix):
            pk = p[:, k:k+1].unsqueeze(-1).repeat(1,self.Q, self.ydim) # [N,Q,D]
            Wk = muW_ +  sigmaW_.sqrt()*torch.randn(N,self.Q,self.ydim) # [N,Q,D]
            Zk = muZ_ + sigmaZ_.sqrt()*torch.randn(N,self.Q, self.ydim) # [N,Q,D]
            tilda_Wk = pk*muW_ + (1-pk.pow(2))*(pk*torch.sqrt(sigmaZ_)/(torch.sqrt(sigmaW_)+1e-7)*(Wk-muW_) + Zk*torch.sqrt(1-pk.pow(2)))
            sampler.append(tilda_Wk)
        tilda_W = torch.stack(sampler) # [K x N x Q x D]
        tilda_W = tilda_W.permute(1,3,0,2) # [N x D x K x Q]
        #compute μ
        mu = torch.matmul(tilda_W.view(N,-1,self.Q), feat.view(N,self.Q,1)).permute(0,2,1)
        # K variance mixture
        varOut = self.varOut(feat).exp().repeat(1,1,self.k_mix)
        var = (1 - p.repeat(1,self.ydim,1).pow(2))*varOut + self.tau_inv
        return p,pi,mu.squeeze(),var.squeeze()

    def compute_loss(self, out , y):
        p,pi,mu,var = out
        select = pi.argmax(dim=1).unsqueeze(-1)
        reg_loss = (1e-5)*F.mse_loss(y, torch.gather(mu, 1, select))
        mdn_loss = MDNLoss(pi, mu, var, y)
        kl_loss = KL(p,pi)
        #print(reg_loss, mdn_loss,kl_loss )
        return reg_loss + mdn_loss + kl_loss

    def sampler(self, x, n_samples=1):
        with torch.no_grad():
            rho,pi,mu,var = self.forward(x)
        pi = pi.numpy()
        n_points = x.shape[0]
        y_sampled = torch.zeros([n_points, self.ydim, n_samples])
        for i in range(n_points):
            for j in range(n_samples):
                pi[i,:] /= pi[i,:].sum().astype(float)
                k = np.random.choice(self.k_mix, p = pi[i,:], replace=True)
            y_sampled[i,:,j] = mu[i,k]
        return y_sampled

if __name__=="__main__":
    x = torch.randn(2,1)
    y = torch.randn(2,1)
    model = ChoiceNetRegression(xdim=1, ydim=1)
    out = model(x)
    loss = model.compute_loss(out, y)
    loss.backward()
    print("output : ",out[0].shape, out[1].shape, out[2].shape,out[3].shape)
    print("loss : ", loss )
