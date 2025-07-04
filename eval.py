#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from models import ModelFT, Model
import torch
import torchvision
import os
from sklearn.metrics import mean_absolute_error
from skimage.metrics import structural_similarity
import cv2
import numpy as np
import math
import argparse
import glob
import yaml
from yaml import Loader
import lpips
from fvcore.nn import FlopCountAnalysis, parameter_count_table
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='')

parser.add_argument('--input_dir', default='datasets/dpdd_dataset/test_c/target', type=str, help='Directory of validation images')
parser.add_argument('--checkpoints', default='checkpoints/dp_2_ft/best.pt', type=str, help='Path to weights')
parser.add_argument('--cuda_device', default=0, type=int, help='device to use')
parser.add_argument("--exp_config", default = "config/config.yaml", type = str)
parser.add_argument("--network_type", default = 2, type = int)
parser.add_argument("--dual_pixel", default = 1, type = int)
parser.add_argument("--srp", default = 1, type = int)

args = parser.parse_args()

exp_config = yaml.load(open(args.exp_config, mode='r'), Loader=Loader)    
exp_config["dual_pixel"] = args.dual_pixel

device = "cuda" 
alex = lpips.LPIPS(net='alex').to(device)

network = "dp_network" if args.dual_pixel == 1 else "single_network"
if args.network_type == 0:
    model_config = exp_config[network]["network_S"]
elif args.network_type == 1:
    model_config = exp_config[network]["network_M"]
else:
    model_config = exp_config[network]["network_L"]
if args.srp == 1:
    model  = ModelFT(model_config,dual_pixel = exp_config["dual_pixel"])
else:
    model  = Model(model_config,dual_pixel = exp_config["dual_pixel"])

print(model)
model.load_state_dict(torch.load(args.checkpoints,map_location = "cpu"))
model.eval()
model.set_inference_mode()


x = torch.rand(1, 3, exp_config["datasets"]["trg_resolution"][0], exp_config["datasets"]["trg_resolution"][1]).cpu()

# with torch.no_grad():
#     x = (x,x) if args.dual_pixel else (x)
#     flops = FlopCountAnalysis(model.cpu(), x)
#     print("FLOPs: ", flops.total())
#     print(parameter_count_table(model))

model.to(device)


def norm_img(img, max_value):
     img = img / float(max_value)  
     return img

def get_ddblur_dataset(path):
    files = glob.glob(os.path.join(path,"*.png"))
    
    all_files = []
    for target_rgb in files:
        input_left = os.path.join(
            *target_rgb.replace("/target/", "/source/").replace("test_c", "test_l").split(os.sep))
        input_right = os.path.join(
            *target_rgb.replace("/target/", "/source/").replace("test_c", "test_r").split(os.sep))
        input_center = os.path.join(
            *target_rgb.replace("/target/", "/source/").split(os.sep))
        
        # breakpoint()
        assert os.path.exists(input_left)  and os.path.exists(input_right), f"input_left: {input_left} or input_right: {input_right} doesn't exist"
        assert os.path.exists(target_rgb)  and os.path.exists(input_center), f"target_rgb: {target_rgb} or input_center: {input_center} doesn't exist"
            
        all_files.append([
            input_left,
            input_right,
            input_center,
            target_rgb,
                ])
        
    flag = -1 if args.dual_pixel else cv2.IMREAD_COLOR  
    max_val = 2**16 -1 if args.dual_pixel else 2**8 - 1
    for  input_left, input_right, input_center, target_rgb in all_files:
        target_rgb = cv2.imread(target_rgb, flag)[...,::-1]
        input_right = cv2.imread(input_right, flag)[...,::-1]
        input_left = cv2.imread(input_left, flag)[...,::-1]
        input_center = cv2.imread(input_center, flag)[...,::-1]
        
        trg_resolution = exp_config["datasets"]["trg_resolution"]
        
        input_left,input_right,input_center, target_rgb = [cv2.resize(i, dsize=trg_resolution[::-1], interpolation=cv2.INTER_CUBIC) for i in [input_left,input_right,input_center]] + [target_rgb]
        
        input_left = norm_img(input_left, max_val)
        input_right = norm_img(input_right, max_val)
        input_center = norm_img(input_center, max_val)
        target_rgb = norm_img(target_rgb, max_val)
        
        
        input_left = torch.Tensor(input_left).permute(2,0,1).float() - 0.5
        input_right = torch.Tensor(input_right).permute(2,0,1).float() - 0.5
        input_center = torch.Tensor(input_center).permute(2,0,1).float() - 0.5
        target_rgb = torch.Tensor(target_rgb).permute(2,0,1).float() - 0.5
        
        yield {
               "input_left" : input_left,
               "input_right": input_right,
               "input_center" : input_center,
               "target_rgb" : target_rgb,
            }
    
dataset = get_ddblur_dataset(args.input_dir)

PSNR, SSIM, MAE, LPIPS = [], [], [], []


save_path_deblur = "results"
os.makedirs(save_path_deblur,exist_ok=True)
save_path_deblur += "dp_" if args.dual_pixel == 1 else "sg_"
save_path_deblur += str(args.network_type)
save_path_deblur += "_ft" if args.srp == 1 else ""
save_path_deblur += "_dpdd"
os.makedirs(save_path_deblur,exist_ok=True)

def mae(img1, img2):
    mae_0=mean_absolute_error(img1[:,:,0], img2[:,:,0],
                              multioutput='uniform_average')
    mae_1=mean_absolute_error(img1[:,:,1], img2[:,:,1],
                              multioutput='uniform_average')
    mae_2=mean_absolute_error(img1[:,:,2], img2[:,:,2],
                              multioutput='uniform_average')
    return np.mean([mae_0,mae_1,mae_2])

def ssim(img1, img2, PIXEL_MAX = 1.0):
    return structural_similarity(img1, img2, data_range=PIXEL_MAX, channel_axis=-1)

def psnr(img1, img2, PIXEL_MAX = 1.0):
    mse_ = np.mean( (img1 - img2) ** 2 )
    return 10 * math.log10(PIXEL_MAX / mse_)

with torch.no_grad():

    # breakpoint()
    for i,data in enumerate(dataset):
        
        data = {k:v.to(device).unsqueeze(0) for k,v in data.items()}
        left,right,input_center, target_rgb = data["input_left"],data["input_right"],data["input_center"], data.get("target_rgb",None)
        
        if args.dual_pixel:
            deblur = model(left,right)
        else:
            deblur = model(input_center)

        deblur = torch.nn.functional.interpolate(deblur,exp_config["datasets"]["src_resolution"], mode ="bicubic")

        deblur = torch.clip(deblur + 0.5,0,1) 
        
        target_rgb = torch.clip(target_rgb + 0.5,0,1)
            
        GT, out = deblur[0].permute(1,2,0).cpu().numpy(), target_rgb[0].permute(1,2,0).cpu().numpy()
            
        PSNR.append(psnr(GT, out))
        MAE.append(mae(GT, out))
        SSIM.append(ssim(GT, out))
        LPIPS.append(alex(target_rgb,deblur,normalize = True).item())
    
        iformat = "png"
        save_file_path_deblur = os.path.join(save_path_deblur, '{:02d}.{}'.format(i+1, iformat))
    
        torchvision.utils.save_image(deblur, '{}'.format(save_file_path_deblur), nrow=1, padding = 0, normalize = False)
    
        diff = np.round(cv2.cvtColor(np.abs(out - GT),cv2.COLOR_RGB2GRAY) * 255)
        diff = cv2.applyColorMap(diff.astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(save_path_deblur, 'e{:02d}.{}'.format(i+1, iformat)),diff)
    
        print('[{:02}] PSNR: {:.5f}, SSIM: {:.5f}, MAE: {:.5f}, LPIPS: {:.5f} '.format(i + 1, PSNR[-1], SSIM[-1], MAE[-1],LPIPS[-1]))
        #print("==============", PSNR,pnsr)
        with open(os.path.join(save_path_deblur, 'score.txt'), 'w' if i == 0 else 'a') as file:
            file.write('[{:02}] PSNR: {:.5f}, SSIM: {:.5f}, MAE: {:.5f}, LPIPS: {:.5f}\n'.format(i + 1, PSNR[-1], SSIM[-1], MAE[-1],LPIPS[-1]))
            file.close()
                

    PSNR_mean = sum(PSNR) / (i + 1)
    SSIM_mean = sum(SSIM)/ (i + 1)
    MAE_mean = sum(MAE) / (i + 1)
    LPIPS_mean = sum(LPIPS) / (i + 1)
    
    print('\nPSNR: {:.5f} SSIM: {:.5f} MAE: {:.5f} LPIPS: {:.5f} '.format( PSNR_mean, SSIM_mean, MAE_mean,LPIPS_mean))
    with open(os.path.join(save_path_deblur, 'score.txt'), 'a') as file:
        file.write('\nPSNR: {:.5f} SSIM: {:.5f} MAE: {:.5f} LPIPS: {:.5f} '.format(PSNR_mean, SSIM_mean, MAE_mean,LPIPS_mean))
        file.close()    
    
    
    
