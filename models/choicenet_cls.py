import torch
import torch.nn as nn

class ChoiceNetCLS(nn.Module):
    def __init__(self,inch=3, feat_dim=128, k_mix = 5, y_):
        super(ChoiceNetCLS, self).__init__()
        self.feat_dim = feat_dim
        self.k_mix = k_mix #from abalation study
        # base network
        self.h = torch.hub.load('pytorch/vision:v0.6.0', 'wide_resnet50_2', pretrained=False)
        if(inch==1):
            self.h.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.h.fc = nn.Linear(in_features=2048, out_features=self.feat_dim, bias=True)
        # cholesky block
        self.rho = nn.Sequential(nn.Linear(in_features=self.feat_dim, out_features=self.k_mix),
                                 nn.Tanh())


    def forward(self,x):
        #base network
        feat = self.h(x)
        #correlation
        p = self.rho(feat)
        p[:,0] = 1 #p1 = 1 (slice backward works in pytorch) [N*K]

        return feat

if __name__=="__main__":
    x = torch.randn(2,1,28,28)
    model = ChoiceNetCLS(inch=1)

    print(model(x).shape)
