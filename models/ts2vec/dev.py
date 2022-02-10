import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pdb

from itertools import chain

class SamePadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1, gamma=0.9):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups, bias=False
        )
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]),requires_grad=True)
        self.padding=padding
        self.dilation = dilation
        self.kernel_size= kernel_size
        
        self.grad_dim, self.shape = [], []
        for p in self.conv.parameters():
            self.grad_dim.append(p.numel())
            self.shape.append(p.size())
        self.dim = sum(self.grad_dim)
        
        self.n_chunks = in_channels
        self.chunk_in_d = self.dim // self.n_chunks
        self.chunk_out_d = int(in_channels*kernel_size// self.n_chunks)
        self.grads = torch.Tensor(sum(self.grad_dim)).fill_(0).cuda()
        nh=64
        self.controller = nn.Sequential(nn.Linear(self.chunk_in_d, nh), nn.SiLU())
        self.calib_w = nn.Linear(nh, self.chunk_out_d)
        self.calib_b = nn.Linear(nh, out_channels//in_channels)
        self.calib_f = nn.Linear(nh, out_channels//in_channels)
        
        #self.calib_w = torch.nn.Parameter(torch.ones(out_channels, in_channels,1), requires_grad = True)
        #self.calib_b = torch.nn.Parameter(torch.zeros([out_channels]), requires_grad = True)
        #self.calib_f = torch.nn.Parameter(torch.ones(1,out_channels,1), requires_grad = True)

        self.remove = 1 if self.receptive_field % 2 == 0 else 0
        self.gamma = gamma
    
    def ctrl_params(self):
        c_iter = chain(self.controller.parameters(), self.calib_w.parameters(), 
                self.calib_b.parameters(), self.calib_f.parameters())
        for p in c_iter:
            yield p

    def store_grad(self):
        #print('storing grad')
        grad = self.conv.weight.grad.data.clone()
        grad = nn.functional.normalize(grad)
        grad = grad.view(-1)
        
        self.grads = self.gamma * self.grads + (1-self.gamma) * grad

    def fw_chunks(self):
        x = self.grads.view(self.n_chunks, -1)
        rep = self.controller(x)
        w = self.calib_w(rep)
        b = self.calib_b(rep)
        f = self.calib_f(rep)
        f = f.view(-1).unsqueeze(0).unsqueeze(2)
       
        return w.unsqueeze(0) ,b.view(-1),f

    def forward(self, x):
        w,b,f = self.fw_chunks()
        d0, d1 = self.conv.weight.shape[1:]
        
        cw = self.conv.weight * w
        #cw = self.conv.weight
        try:
            conv_out = F.conv1d(x, cw, padding=self.padding, dilation=self.dilation, bias = self.bias * b)
            out = f * conv_out
        except: pdb.set_trace()
        return out

    def representation(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out

    def _forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False, gamma=0.9):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation, gamma=gamma)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation, gamma=gamma)
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None
    
    def ctrl_params(self):  
        c_iter = chain(self.conv1.controller.parameters(), self.conv1.calib_w.parameters(), 
                self.conv1.calib_b.parameters(), self.conv1.calib_f.parameters(),
                self.conv2.controller.parameters(), self.conv2.calib_w.parameters(), 
                self.conv2.calib_b.parameters(), self.conv2.calib_f.parameters())

        return c_iter 
       


    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual

class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size, gamma=0.9):
        super().__init__()
        self.net = nn.Sequential(*[
            ConvBlock(
                channels[i-1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2**i,
                final=(i == len(channels)-1), gamma=gamma
            )
            for i in range(len(channels))
        ])
    def ctrl_params(self):
        ctrl = []
        for l in self.net:
            ctrl.append(l.ctrl_params())
        c = chain(*ctrl)
        for p in c:
            yield p
    def forward(self, x):
        return self.net(x)
