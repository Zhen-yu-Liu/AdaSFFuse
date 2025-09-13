import argparse
import cv2
import glob
import numpy as np
from collections import OrderedDict
import os
import torch
import requests
import time
import sys

from models.network import MambaDFuse as net
from utils import utils_image as util
from data.dataloder import Dataset as D
from torch.utils.data import DataLoader
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"                                                                      # TODO:1 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='fusion', help='classical_sr, lightweight_sr, real_sr, '
                                                                     'gray_dn, color_dn, jpeg_car')
    parser.add_argument('--scale', type=int, default=1, help='scale factor: 1, 2, 3, 4, 8')  # 1 for dn and jpeg car
    parser.add_argument('--large_model', action='store_true', help='use large model, only provided for real image sr')
    parser.add_argument('--model_path', type=str,
                        default='/checkpoint/MEF/Multi_Exposure_Fusion/models')                               
    parser.add_argument('--iter_number', type=str,
                        default='64000')                                                                        # TODO:2
    parser.add_argument('--root_path', type=str, default='/imagefusion/test',                                   # TODO:3
                        help='input test image root folder')
    parser.add_argument('--dataset', type=str, default='MEF',                                                   # TODO:4 
                        help='input test image name')
    parser.add_argument('--A_dir', type=str, default='low',
                        help='input test image name')
    parser.add_argument('--B_dir', type=str, default='over',
                        help='input test image name')
    parser.add_argument('--tile', type=int, default=None, help='Tile size, None for no tile during testing (testing as a whole)')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
    parser.add_argument('--in_channel', type=int, default=1, help='3 means color image and 1 means gray image')
    args = parser.parse_args()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = define_model(args)
    model.eval()
    model = model.to(device)

    folder, save_dir, border, window_size = setup(args)
    a_dir = os.path.join(args.root_path, args.dataset, args.A_dir)
    b_dir = os.path.join(args.root_path, args.dataset, args.B_dir)
    print(a_dir)
    os.makedirs(save_dir, exist_ok=True)

    for img_name in os.listdir(a_dir):
        start = time.time()

        # 读取图像
        img_a = cv2.imread(os.path.join(a_dir, img_name), cv2.IMREAD_UNCHANGED)
        img_b = cv2.imread(os.path.join(b_dir, img_name), cv2.IMREAD_UNCHANGED)

        # 判断图像通道数并处理
        img_a_y, img_a_cr, img_a_cb = process_image(img_a)
        img_b_y, img_b_cr, img_b_cb = process_image(img_b)

        # 准备输入数据
        img_a_y = torch.FloatTensor(img_a_y).unsqueeze(0).unsqueeze(0).to(device) / 255.0
        img_b_y = torch.FloatTensor(img_b_y).unsqueeze(0).unsqueeze(0).to(device) / 255.0

        with torch.no_grad():
            # 填充图像
            _, _, h_old, w_old = img_a_y.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            img_a_y = torch.cat([img_a_y, torch.flip(img_a_y, [2])], 2)[:, :, :h_old + h_pad, :]
            img_a_y = torch.cat([img_a_y, torch.flip(img_a_y, [3])], 3)[:, :, :, :w_old + w_pad]
            img_b_y = torch.cat([img_b_y, torch.flip(img_b_y, [2])], 2)[:, :, :h_old + h_pad, :]
            img_b_y = torch.cat([img_b_y, torch.flip(img_b_y, [3])], 3)[:, :, :, :w_old + w_pad]
            
            output = test(img_a_y, img_b_y, model, args, window_size)
            output = output[..., :h_old * args.scale, :w_old * args.scale]

        # 处理融合结果
        output = (output - torch.min(output)) / (torch.max(output) - torch.min(output))
        fused_y = np.squeeze((output * 255).cpu().numpy()).astype(np.uint8)

        # 重建彩色图像
        if img_b_cr is not None and img_b_cb is not None:
            ycrcb_fused = np.dstack((fused_y, img_b_cr, img_b_cb))
            rgb_fused = cv2.cvtColor(ycrcb_fused, cv2.COLOR_YCrCb2BGR)
        else:
            rgb_fused = fused_y

        end = time.time()
        
        # 保存结果
        save_name = os.path.join(save_dir, img_name)
        cv2.imwrite(save_name, rgb_fused)
        print(f"[{img_name}] Saving fused image to: {save_name}, Processing time is {end - start:.4f} s")

# process_image: 可更改颜色空间或图像处理方式。
# define_model: 更改模型参数或更换模型。
# setup: 改变输入输出目录或其他设置。
# test: 可以改进合并逻辑和增强测试速度或内存管理。


def process_image(img):
    if len(img.shape) == 3:  # 彩色图像
        img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(img_ycrcb)
        return y, cr, cb
    else:  # 单通道图像
        return img, None, None

def define_model(args):
    model = net(upscale=args.scale, in_chans=args.in_channel, img_size=128, window_size=8,
                img_range=1., depths=[6, 6, 6, 6], embed_dim=64, num_heads=[6, 6, 6, 6],
                mlp_ratio=2, upsampler=None, resi_connection='1conv')
    param_key_g = 'params'
    model_path = os.path.join(args.model_path, args.iter_number + '_E.pth')
    pretrained_model = torch.load(model_path)
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
    
        
    return model

def setup(args):   
    save_dir = f'results/Visual/AdaSFFuse_{args.dataset}'
    folder = os.path.join(args.root_path, args.dataset, 'A_Y')
    print('folder:', folder)
    border = 0
    window_size = 8

    return folder, save_dir, border, window_size




def test(img_a, img_b, model, args, window_size):
    if args.tile is None:
        # test the image as a whole
        output = model(img_a, img_b)
    else:
        # test the image tile by tile
        b, c, h, w = img_a.size()
        tile = min(args.tile, h, w)
        assert tile % window_size == 0, "tile size should be a multiple of window_size"
        tile_overlap = args.tile_overlap
        sf = args.scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h*sf, w*sf).type_as(img_a)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_a[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
        output = E.div_(W)

    return output

if __name__ == '__main__':
    main()
