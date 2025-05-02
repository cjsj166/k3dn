from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from einops import rearrange


class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()
        self.start = nn.Sequential(
            BasicConv(3, 32, kernel_size=3, padding=1),
            BasicConv(32, 64, kernel_size=3, stride=1, padding=1))
        self.layer1 = nn.Sequential(
            BasicConv(64, 128, kernel_size=3, stride=1, padding=4, dilation=4),
            BasicConv(128, 128, False, kernel_size=3, stride=2, padding=8,dilation=8))

        self.branch1 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32,32)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))


        self.branch3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16,16)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.end = nn.Sequential(
            BasicConv(192, 96, kernel_size=3, padding=1),
            BasicConv(96, 32, kernel_size=1, bn=False, relu=False, padding=0))

    def forward(self, x):

        x = self.start(x)

        x = self.layer1(x)

        output_branch1 = self.branch1(x)
        output_branch1 = F.interpolate(output_branch1, (x.size()[2],x.size()[3]),mode='bilinear',align_corners=True)
        output_branch3 = self.branch3(x)
        output_branch3 = F.interpolate(output_branch3, (x.size()[2],x.size()[3]),mode='bilinear',align_corners=True)
              
        output_feature = torch.cat((output_branch1, output_branch3,  x), 1)
        output_feature = self.end(output_feature)

        return output_feature

def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation, dilation = dilation, bias=False), nn.BatchNorm2d(out_planes))

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, dilation=1, stride=1, downsample=None):
        super(ResBlock, self).__init__()

        padding = dilation

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride,padding=padding, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride,padding=padding, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.downsample = downsample
        self.stride = stride
        self.in_ch = in_channel
        self.out_ch = out_channel
        self.p = padding
        self.d = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual

        out = self.relu2(out)

        return out

class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()

        self.relu = relu
        self.use_bn = bn
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            if bn:
                self.bn = nn.BatchNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            if bn:
                self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x
    
class Matching(nn.Module):
    def __init__(self):
        super(Matching, self).__init__()
        self.start =  nn.Sequential(
            BasicConv(64, 32, is_3d=True, kernel_size=3, padding=1),
            BasicConv(32, 48, is_3d=True, kernel_size=3, stride=2, padding=1),
            BasicConv(48, 48, is_3d=True, kernel_size=3, stride=1,padding=1))
        self.conv1a = nn.Sequential(
            BasicConv(48, 64, is_3d=True, kernel_size=3, stride=2, padding=1),
            BasicConv(64, 64, is_3d=True, kernel_size=3, padding=1))
    def forward(self, x):
        x = self.start(x)
        x = self.conv1a(x)
        return x    
 
class YRStereonet_3D(nn.Module):
    def __init__(self, config):
        super(YRStereonet_3D, self).__init__()
        self.maxdisp = 12
        self.feature = Feature()
        self.matching = Matching()
        self.avgpool = nn.AvgPool2d(config["pool_size"])
        self.fc = nn.Sequential(
            nn.Linear(64, 256),
            nn.LeakyReLU(0.2,True)
            )
        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        self.config = config
    def forward(self, xl, yr):  
        
        _,_,h,w = xl.size()
        
        xl = F.avg_pool2d(xl,4)
        yr = F.avg_pool2d(yr,4)
        
        x = self.feature(xl) 
        y = self.feature(yr)
  
        cost = x.new().resize_(x.size()[0], x.size()[1]*2, int(self.maxdisp/3),  x.size()[2],  x.size()[3])  
        for i in range(int(self.maxdisp/3)):
            if i > 0 : 
                cost[:,:x.size()[1], i,:,i:] = x[:,:,:,i:]
                cost[:,x.size()[1]:, i,:,i:] = y[:,:,:,:-i]
            else:
                cost[:,:x.size()[1],i,:,i:] = x
                cost[:,x.size()[1]:,i,:,i:] = y
        
        cost = self.matching(cost).squeeze(2)
        cost = torch.nn.functional.interpolate(cost, (h//self.config["pad_scale"][0] + cost.size(2),w//self.config["pad_scale"][1] + cost.size(3)))
        cost = self.avgpool(cost)
        cost = rearrange(cost, "b c h w -> (b h w) c")
        cost = self.fc(cost)
        return cost
    
class SingleImage(nn.Module):
    def __init__(self,config):
        super().__init__()
        
       	self.model = nn.Sequential(
            BasicConv(3, 32, kernel_size=3, padding=1),
            BasicConv(32, 64, kernel_size=3, stride=1, padding=1),
            BasicConv(64, 128, kernel_size=3, stride=1, padding=4, dilation=4),
            BasicConv(128, 128, kernel_size=3, padding=1),
            BasicConv(128, 128, kernel_size=3, stride=2, padding=4,dilation=4),
            BasicConv(128, 256, kernel_size=1, padding=0),
            BasicConv(256, 128, kernel_size=3, padding=1),
            BasicConv(128, 64, kernel_size=3, stride=2, padding=1),
            BasicConv(64, 64, kernel_size=3, stride=1,padding=1),
            BasicConv(64, 64, kernel_size=3, stride=2, padding=1),
            BasicConv(64, 64, kernel_size=3, padding=1)
            )
        
       	self.avgpool = nn.AvgPool2d(config["pool_size"])
        self.fc = nn.Sequential(
            nn.Linear(64, 256),
            nn.LeakyReLU(0.2,True)
            )
        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        self.config = config
        
    def forward(self,x):
            
        _,_,h,w = x.size()
        x = F.avg_pool2d(x,4)
        cost = self.model(x)
        cost = torch.nn.functional.interpolate(cost, (h//self.config["pad_scale"][0] + cost.size(2),w//self.config["pad_scale"][1] + cost.size(3)))
        cost = self.avgpool(cost)
        cost = rearrange(cost, "b c h w -> (b h w) c")
        cost = self.fc(cost)
        return cost     
