#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import torch
from torch import nn
from torch.nn import functional as F
from watt import watt
# from watt_local_attention import watt as watt_cpu
from einops import rearrange

class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1 ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )
        return out

class InvertibleDepthwiseConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        kernel_size,
        style_dim,
        fdomain_size = 10,
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        
        self.padding = kernel_size // 2
        self.weight = nn.Parameter(
            torch.randn(1, in_channel, 1, fdomain_size,  fdomain_size//2 + 1,2)
        )


        idx = (fdomain_size - kernel_size)//2
        self.idx_row = (idx, idx + kernel_size)
        self.idx_col = (idx, idx + kernel_size//2 + 1)
        # print('kernel_size', kernel_size)
        
    def get_weight(self, inverse = False):

        weight =  torch.view_as_complex(self.weight)

        if inverse:
            weight = 1 / (weight + self.eps)
            mask = torch.isnan(weight)
            weight[mask] = 0

        weight = torch.fft.irfft2(weight)

        # print('weight.shape', weight.shape)
        weight_left = weight[...,self.idx_row[0]:self.idx_row[1],self.idx_col[0]:self.idx_col[1]]
        # print('weight_left.shape', weight_left.shape)

        weight_right = torch.flip(weight_left[...,:-1],[-1])
        weight = torch.cat((weight_left,weight_right),-1)

        weight = self.normalize(weight)

        return weight
        
    def normalize(self, x):
        demod = torch.rsqrt(x.pow(2).sum([-1,-2,-3], True) + self.eps)
        return x * demod
        
    def forward(self, input, inverse = False):
        batch, in_channel, height, width = input.shape
        
        weight = self.get_weight(inverse)

        weight = weight.view(
             self.in_channel, 1, self.kernel_size, self.kernel_size
        )
        
        out = F.conv2d(
            input, weight, bias = None, stride = 1, padding=self.padding, dilation = 1, groups= in_channel
        )
        
        return out


class InvertibleDepthwiseConv2dWrapper(nn.Module):
    def __init__(self,layer):
        super().__init__()

        weight   =  layer.get_weight().view(layer.in_channel,1,layer.kernel_size,layer.kernel_size)
        _,_,h,w = weight.size()
        weight_left = weight[:,:,:,:w//2+1]
        
        self.register_buffer("weight_left", weight_left)
        self.padding = layer.padding
        self.in_channel = layer.in_channel
        self.kernel_size = layer.kernel_size
        
    def forward(self, input, *args):
        
        weight = self.get_weight()
        weight = weight.view(
             self.in_channel, 1, self.kernel_size, self.kernel_size
        ) # Kernels to be used for deblur
        out = F.conv2d(
            input, weight, bias = None, stride = 1, padding=self.padding, dilation = 1, groups= self.in_channel
        ) # Deblur
        
        return out
    
    def get_weight(self,*args,**kwargs):
        weight_left = self.weight_left
        weight_right = torch.flip(weight_left[...,:-1],[-1])
        weight = torch.cat((weight_left,weight_right),-1)
        
        return weight.unsqueeze(0)


class InvertiblePointConv2d(nn.Module):
    def __init__(self, in_channel,style_dim,truncate):
        super().__init__()
        w_shape = [in_channel, in_channel]
        w_init = torch.qr(torch.randn(*w_shape))[0]

        p, lower, upper = torch.lu_unpack(*torch.lu(w_init))
        s = torch.diag(upper)
        sign_s = torch.sign(s)
        log_s = torch.log(torch.abs(s))
        upper = torch.triu(upper, 1)
        l_mask = torch.tril(torch.ones(w_shape), -1)
        eye = torch.eye(*w_shape)

        self.register_buffer("p", p)
        self.register_buffer("sign_s", sign_s)
        self.lower = nn.Parameter(lower)
        self.log_s = nn.Parameter(log_s)
        self.upper = nn.Parameter(upper)
        self.l_mask = l_mask
        self.eye = eye

        self.w_shape = w_shape
        
        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)
        
        self.eps = 1e-8
        self.truncate = truncate

    def get_weight(self, input,style, reverse):
        b, c, h, w = input.shape

        self.l_mask = self.l_mask.to(input.device)
        self.eye = self.eye.to(input.device)

        lower = self.lower * self.l_mask + self.eye

        u = self.upper * self.l_mask.transpose(0, 1).contiguous()
        u += torch.diag(self.sign_s * torch.exp(self.log_s))

        style = self.modulation(style)

        if reverse:
            u_inv = torch.inverse(u)
            l_inv = torch.inverse(lower)
            p_inv = torch.inverse(self.p)

            weight = torch.matmul(u_inv, torch.matmul(l_inv, p_inv))
        else:
            weight = torch.matmul(self.p, torch.matmul(lower, u))
            
        style = style.view(-1,self.w_shape[0],1,1)

        return style.softmax(1), weight.view(self.w_shape[0], self.w_shape[1], 1, 1)

    def forward(self, input, style, reverse=False):
        batch, in_channel, height, width = input.shape
        style, weight = self.get_weight(input, style, reverse)
        
        if reverse:
            style = torch.clip(torch.nan_to_num(1/(style + 1e-8),0),0, self.truncate) #10)
            demod = torch.rsqrt(style.pow(2).sum([-1,-2,-3], True) + self.eps)
            style=style*demod 
            out = F.conv2d(
                input, weight,bias = None,
            ) * style
            

        else:
            out = F.conv2d(
                input * style, weight,bias=None
            )
        return out
      
class Interaction(nn.Module):
    def __init__(self,in_channel,style_dim,kernel_size):
        super().__init__()
        self.to_w = nn.Parameter(torch.zeros(1,in_channel,1,kernel_size,kernel_size//2+1))
        self.to_r = nn.Linear(style_dim, in_channel)
        self.out = nn.Linear(in_channel, style_dim)
        self.relu  = nn.LeakyReLU(0.2,True)
        
        nn.init.trunc_normal_(self.to_r.weight,std=0.001)
        nn.init.zeros_(self.to_r.bias)
        nn.init.trunc_normal_(self.out.weight,std=0.001)
        nn.init.zeros_(self.out.bias)
        
    def forward(self,r,w):
        w = (self.to_w * w).sum([-1,-2,-3])
        return self.relu(self.out(self.to_r(r) * w)) + r
        
class PSFBlock(nn.Module):
    def __init__(self, in_channel,style_dim,truncate):
        super().__init__()
        self.conv1 = InvertibleDepthwiseConv2d(in_channel, 9, style_dim, 10)
        self.conv2 = InvertiblePointConv2d(in_channel, style_dim,truncate)
        self.relu  = nn.LeakyReLU(0.2,True)
        self.inter = Interaction(in_channel,style_dim,9)
    def forward(self, x, style, rev = False):
        skip = x
        w = self.conv1.get_weight(False)[:,:,:,:,:9//2+1] # Retrieving learned kernel
        style = self.inter(style,w)
        if rev:
            x = self.conv2(x,style,rev)
            x = self.relu(x)
            x = self.conv1(x,rev)
        else:
            x = self.conv1(x,rev)
            x = self.relu(x)
            x = self.conv2(x,style,rev)
        return self.relu(x) + skip
    
    
class SRPBlock(nn.Module):
    def __init__(self, new_channel,old_channel,psf_out_layer):
        super().__init__()
        self.conv_q = nn.Conv2d(new_channel+old_channel, new_channel+old_channel, 3, padding=1)
        self.conv_k = nn.Conv2d(new_channel+old_channel, new_channel+old_channel, 3, padding=1) 
        self.conv_v = nn.Conv2d(81, new_channel+old_channel, 3, padding=1)
        self.conv_o = nn.Sequential(
            nn.Conv2d(new_channel+old_channel, new_channel+old_channel, 3, padding=1),
            EBlock(new_channel + old_channel,psf_out_layer),
            nn.Conv2d(new_channel+old_channel, new_channel+old_channel, 3, padding = 1),
            )
        
    def forward(self,old,new):
        x=torch.cat((old,new),1)
        if x.is_cuda:
            att = watt.WATT(self.conv_q(x), self.conv_k(x))
        else:
            att = rearrange(watt_cpu.TorchLocalAttention.f_similar(self.conv_q(x),self.conv_k(x),9,9),"b h w c -> b c h w")
            
        x=self.conv_o(x*self.conv_v(att)) + x
        return x               
    
class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=False, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x
        
        
class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)        
        
