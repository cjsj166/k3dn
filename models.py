import torch
from torch import nn
from layers import PSFBlock,BasicConv,InvertibleDepthwiseConv2dWrapper, EBlock, SRPBlock
from disp import YRStereonet_3D, SingleImage
from einops import rearrange

class Model(nn.Module):
    def __init__(self, model_config, dual_pixel = True):
        super().__init__()
        
        base_channel = model_config["base_channel"]
        ed_res = model_config["ed_res"]
        ld_res = model_config["ld_res"]
        truncate = model_config["truncate"]
        self.window_size = model_config["window_size"]
        
        in_channel = 6 if dual_pixel else 3
        self.dual_pixel = dual_pixel
        
        self.Encoder = nn.ModuleList([
            EBlock(base_channel, ed_res),
            EBlock(base_channel*2, ed_res),
            EBlock(base_channel*4, ed_res),
        ])
        
        self.feat_extract = nn.ModuleList([
            BasicConv(in_channel, base_channel, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*4, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2,base_channel, kernel_size=4, relu=True, stride=2, transpose=True)
        ])
        
        self.Decoder = nn.ModuleList([
            EBlock(base_channel * 4, ed_res),
            EBlock(base_channel * 2, ed_res),
            EBlock(base_channel + in_channel, ed_res)
        ])
        
        self.disp = YRStereonet_3D(model_config["disp"]) if dual_pixel else SingleImage(model_config["disp"])
        self.deblur = nn.ModuleList([PSFBlock(base_channel*4, 256,truncate) for _ in range(ld_res)])
        self.out = BasicConv(base_channel + in_channel, 3, kernel_size=3, relu=False, stride=1)
        
    def _forward_impl(self,x, disp_vector, rev = False):
    
        
        res_1 = self.Encoder[0](self.feat_extract[0](x))
        res_2 = self.Encoder[1](self.feat_extract[1](res_1))
        res_3 = self.Encoder[2](self.feat_extract[2](res_2))
        
        _,_, h,w = res_3.size()
        res_3 = rearrange(res_3, "b c (p1 w) (p2 h) -> (b p1 p2) c w h", p1 = h//self.window_size[0], p2 =w//self.window_size[1])
        
        for m in self.deblur:
            res_3 = m(res_3,disp_vector,rev)
        res_3 = rearrange(res_3, "(b p1 p2) c w h -> b c (p1 w) (p2 h)", p1 = h//self.window_size[0], p2 =w//self.window_size[1])
        z = self.Decoder[0](torch.cat((res_2, self.feat_extract[3](res_3)),1))
        z = self.Decoder[1](torch.cat((res_1, self.feat_extract[4](z)),1))
        x   = self.Decoder[2](torch.cat((x, self.feat_extract[5](z)),1))
        x = self.out(x)
        return torch.clip(x,-1,1)
    
    def set_inference_mode(self):
        
        deblur = []
        
        for m in self.deblur:
            m.conv1 = InvertibleDepthwiseConv2dWrapper(m.conv1)
            deblur.append(m)
            
        self.deblur = nn.ModuleList(deblur)
        if self.dual_pixel:
            self.forward = lambda *args: self.inference_dp(*args)
        else:
            self.forward = lambda *args: self.inference_sg(*args)
        
    def inference_dp(self,left,right):
        
        disp_vector = self.disp(left,right)
        deblur = self._forward_impl(torch.cat((left,right),1), disp_vector)
        
        return deblur 
        
    
class ModelFT(Model):
    def __init__(self, model_config, dual_pixel = True):
        super().__init__(model_config,dual_pixel)
        
        base_channel = model_config["base_channel"]
        in_channel = 6 if dual_pixel else 3
        psf_out_layer = model_config["psf_out_layer"]
         
        self.srp = nn.ModuleList(
            [SRPBlock(base_channel * 2,base_channel * 2, psf_out_layer),
             SRPBlock(base_channel, base_channel, psf_out_layer),
             SRPBlock(base_channel, in_channel, psf_out_layer),
             ]
            )
        
        for param in self.parameters():
            param.requires_grad = False
            
        for param in self.srp.parameters():
            param.requires_grad = True
        
    def _forward_impl(self,x, disp_vector, rev = False):
        
        res_1 = self.Encoder[0](self.feat_extract[0](x))
        res_2 = self.Encoder[1](self.feat_extract[1](res_1))
        res_3 = self.Encoder[2](self.feat_extract[2](res_2))
        
        _,_, h,w = res_3.size()
        res_3 = rearrange(res_3, "b c (p1 w) (p2 h) -> (b p1 p2) c w h", p1 = h//self.window_size[0], p2 =w//self.window_size[1])
        
        for m in self.deblur:
            res_3 = m(res_3,disp_vector,rev)
        res_3 = rearrange(res_3, "(b p1 p2) c w h -> b c (p1 w) (p2 h)", p1 = h//self.window_size[0], p2 =w//self.window_size[1])
        z = self.Decoder[0](self.srp[0](res_2, self.feat_extract[3](res_3)))
        z = self.Decoder[1](self.srp[1](res_1, self.feat_extract[4](z)))
        x  = self.Decoder[2](self.srp[2](x, self.feat_extract[5](z)))
        
        x = self.out(x)
        return torch.clip(x,-1,1)
        
        
        
    