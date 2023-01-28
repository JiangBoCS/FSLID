from torch.functional import Tensor
import torch.nn as nn
import torch
from functools import partial
import kornia
import math
import warnings
import torch.nn.functional as f

class MSFR(nn.Module):
    '''Multi-Scale Feature Recursive module'''
    def __init__(self, in_c, out_c, stride=1):
        super(MSFR, self).__init__()
        in_channels = out_c
        s = 0.5
        self.num_steps = 2
        self.two = int(in_channels * s)
        self.four = int(in_channels * (s ** 2))
        self.eight = int(in_channels * (s ** 3))
        self.sixteen = int(in_channels * (s ** 4))

        self.inputs_c0 = nn.Sequential(
            nn.Conv2d(in_c, in_channels, 3, 1, 1),
            nn.LeakyReLU(inplace=True)) #in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, valid_padding=True, padding=0
            
        self.c1 = nn.Sequential(
            nn.Conv2d(self.two, self.two, kernel_size=9, stride=stride, padding=4),
            nn.LeakyReLU(inplace=True))
            
        self.c2 = nn.Sequential(
            nn.Conv2d(self.four, self.four, kernel_size=7, stride=stride, padding=3),
            nn.LeakyReLU(inplace=True))
            
        self.c3 = nn.Sequential(
            nn.Conv2d(self.eight, self.eight, kernel_size=5, stride=stride, padding=2),
            nn.LeakyReLU(inplace=True))
            
        self.c4 = nn.Sequential(
            nn.Conv2d(self.sixteen, self.sixteen, kernel_size=3, stride=stride, padding=1),
            nn.LeakyReLU(inplace=True))

        self.out = nn.Conv2d(self.num_steps * in_channels, in_channels, kernel_size=2, stride=2, padding=0)

    def forward(self, inputs):
        feature = []
        for _ in range(self.num_steps):
            out_c0 = self.inputs_c0(inputs)
            distilled_c0, remaining_c0 = torch.split(out_c0, (self.two, self.two), dim=1)

            out_c1 = self.c1(remaining_c0)
            distilled_c1, remaining_c1 = torch.split(out_c1, (self.four, self.four), dim=1)

            out_c2 = self.c2(remaining_c1)
            distilled_c2, remaining_c2 = torch.split(out_c2, (self.eight, self.eight), dim=1)

            out_c3 = self.c3(remaining_c2)
            distilled_c3, remaining_c3 = torch.split(out_c3, (self.sixteen, self.sixteen), dim=1)

            out_c4 = self.c4(remaining_c3)

            out = torch.cat([distilled_c1, distilled_c2, distilled_c3, out_c4, distilled_c0], dim=1)

            inputs = out
            feature.append(out)

        out_fused = torch.cat(feature, dim=1)

        return self.out(out_fused)
    

class Encoder(nn.Module):
    def __init__(self, in_channels, n_feat):
        super(Encoder, self).__init__()
        
        self.f0 = nn.Sequential(
            nn.Conv2d(in_channels, n_feat, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(n_feat, n_feat, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(inplace=True))
        
        self.f1 = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(n_feat, n_feat, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(inplace=True))
            
        self.f2 = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(n_feat, n_feat, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(inplace=True))
            
        self.f3 = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(n_feat, n_feat, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(inplace=True))
            
        self.f4 = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(n_feat, n_feat, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(inplace=True))

    def forward(self, x):
        f0 = self.f0(x)
        f1 = self.f1(f0)
        f2 = self.f2(f1)
        f3 = self.f3(f2)
        f4 = self.f4(f3)

        return [x, f0, f1, f2, f3, f4]

class Noiser(nn.Module):
    def __init__(self, dim, in_channels=3, out_channels=3):
        super(Noiser, self).__init__()

        self.f0 = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True))
        self.F = MSFR(dim, dim, 1)

        self.f1 = MSFR(dim, dim, 1)
        self.f2 = MSFR(dim, dim, 1)
        self.f3 = MSFR(dim, dim, 1)
        self.f4 = MSFR(dim, dim, 1)

    def forward(self, x, F): #, F        
        
        f0 = self.F(self.f0(x))
        f1 = self.f1(f0+F[1])
        f2 = self.f2(f1+F[2])
        f3 = self.f3(f2+F[3])
        f4 = self.f4(f3+F[4])

        return [x, f0, f1, f2, f3, f4]


class Decoder(nn.Module):
    def __init__(self, dim, out_channels):
        super(Decoder, self).__init__()
        
        self.d0 = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True))
            
        self.d1 = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True)) #6,2,2
            
        self.d2 = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True))
            
        self.d3 = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True))
            
        self.d4 = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True))
        
        self.out = nn.Conv2d(dim, out_channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        d0 = self.d0(x[5])
        d1 = self.d1(d0 + x[4]) 
        d2 = self.d2(d1 + x[3]) 
        d3 = self.d3(d2 + x[2]) 
        d4 = self.d4(d3 + x[1])
        
        out = self.out(d4) + x[0]
       
        return out
        
class DN(nn.Module):
    def __init__(self, dim, in_channels=3, out_channels=3, few_shot=False):
        super(DN, self).__init__()
        
        self.few_shot = few_shot
        self.Encoder = Encoder(in_channels, dim)
        self.Decoder = Decoder(dim, out_channels)
        self.Noiser = Noiser(dim)
        
    def forward(self, x, few_shot=False):
    
        fea = self.Encoder(x)
        
        if few_shot:
            fea_N = self.Noiser(x, fea)
            out = self.Decoder(fea_N)
        else:
            out = self.Decoder(fea)
       
        return out
        
if __name__ == "__main__":

    import torch
    import torchvision
    import kornia as K
    from ptflops import get_model_complexity_info
    import time
    import math
    import torchvision
    
    import cv2
    from matplotlib import pyplot as plt
    import numpy as np
    import timm

#        a = torch.randn(1,32,64,64)
#        b = torch.randn(1,32,64,64)
#        c = torch.randn(1,32,64,64)
#        d = torch.randn(1,32,64,64)
#        e = torch.randn(1,32,64,64)
#        
#        F = [a,b,c,d,e]

    y = torch.randn(1,3,256,256)
    M = DN(in_channels=3, dim=64, out_channels=3)
    x = M(y)
    print(x.shape)
#    
    flops, params = get_model_complexity_info(M, (3, 256, 256), as_strings=True, print_per_layer_stat=False)
    print('Flops:  ' + flops)
    print('Params: ' + params)