
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple, trunc_normal_
from einops import rearrange
import numbers
from einops import rearrange, repeat
import copy
from models.Mamba2_2d import Mamba2_2d
# from Mamba2_2d_Cross import Mamba2_2d_cross
from models.mywavelet_utils import MyDWT, MyIDWT 


import scipy.misc
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
plt.switch_backend('agg')
import numpy as np

def show_all_color_feature_map(feature_map,path, name):
    feature_map = feature_map.squeeze(0).detach().cpu().numpy()  #得到array的shape为(48, 544, 960)
    # index = feature_map.mean(0) # 求均值
    feature_map_num = feature_map.shape[0]#返回通道数
    for index in range(1, feature_map_num + 1):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.axis('off')  # 关闭坐标轴
        fn = '{}/{}_{}'.format(path, str(name),str(index))
        im = ax.imshow(feature_map[index - 1], cmap=plt.cm.get_cmap("viridis"))
        plt.tight_layout()  # 调整图像布局
        plt.savefig('{}.png'.format(fn), bbox_inches='tight', pad_inches=0, dpi=300)  # 保存并裁剪图像

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        if len(x.shape)==4:
            h, w = x.shape[-2:]
            return to_4d(self.body(to_3d(x)), h, w)
        else:
            return self.body(x)


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

class PatchUnEmbed(nn.Module):
    def __init__(self, embed_dim=96):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

class Feature_Ext(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.2):
        super(Feature_Ext, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(relu_slope, inplace=True))
            # nn.Conv2d(out_size // 2, out_size, kernel_size=3, padding=1, bias=True),
            # nn.LeakyReLU(relu_slope, inplace=True))

    def forward(self, x):
        # import pdb;pdb.set_trace()
        x = self.block(x)
        return x
    
#################### Mamba
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, drop=0.4):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 2 * hidden_features)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.fc1(x)
        x1, x2 = x12.chunk(2, dim=-1)
        gated_x = self.act(x1) * x2
        out = self.fc2(gated_x)
        out = self.drop(out)
        return out

class ConvMlp(nn.Module):
    """ MLP using 1x1 convs that keeps spatial dims
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.ReLU,
        norm_layer=None,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=True)
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=True)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

class SFMamba2D_Block(nn.Module):
    def __init__(self, dim):
        super(SFMamba2D_Block, self).__init__()
        self.norm1 = LayerNorm(dim,'with_bias')
        self.norm2 = LayerNorm(dim,'with_bias')
        self.mamba = Mamba2_2d(dim)
        self.mlp = Mlp(dim, dim * 2, dim)

    def forward(self, x, H, W):
        residual = x
        x1 = self.norm1(x)
        x_m = residual + self.mamba(x1,H,W)
        
        residual2 = x_m
        x2 = self.norm2(x_m)
        output = residual2 + self.mlp(x2)

        return output

class Freq_SFMamba2D(nn.Module):
    def __init__(self, embed_dim, num_blocks=2):
        super(Freq_SFMamba2D, self).__init__()
        self.blocks = nn.ModuleList([SFMamba2D_Block(embed_dim) for _ in range(num_blocks)])

    def forward(self, x, h, w):
        for block in self.blocks:
            x = block(x, h, w)
        return x
################## Network
class MambaDFuse(nn.Module):    ###########################################
    def __init__(self, img_size=64, patch_size=1, in_chans=1,
                 embed_dim=64, Ex_depths=[4], Fusion_depths=[2, 2], Re_depths=[4], 
                 Ex_num_heads=[6], Fusion_num_heads=[6, 6], Re_num_heads=[6],
                 window_size=7,qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 upscale=1, img_range=1., resi_connection='1conv',
                 **kwargs):
        super(MambaDFuse, self).__init__()
        num_out_ch = in_chans
        # num_feat = 64
        self.img_range = img_range
        embed_dim_temp = int(embed_dim / 2)
        print('in_chans: ', in_chans)
        if in_chans == 3 or in_chans == 6:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            rgbrgb_mean = (0.4488, 0.4371, 0.4040, 0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
            self.mean_in = torch.Tensor(rgbrgb_mean).view(1, 6, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        
        self.upscale = upscale
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
    

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.patch_unembed = PatchUnEmbed(embed_dim=embed_dim)
        self.softmax = nn.Softmax(dim=0)
        self.apply(self._init_weights)

        ################################### 1, feature extraction ###################################
        self.low_level_feature_extraction= Feature_Ext(in_chans, embed_dim)

        ################################### 2, DWT ######################################
        self.dwt = MyDWT(embed_dim)
        self.idwt = MyIDWT(embed_dim)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        ################################### 3, Freq Fusion ######################################
        self.HiFreqFusion = Freq_SFMamba2D(3 * self.embed_dim, num_blocks=2)
        self.LoFreqFusion = Freq_SFMamba2D(self.embed_dim, num_blocks=2)
        self.FreqFusion = Freq_SFMamba2D(self.embed_dim, num_blocks=4)
        #####################################################################################################
        self.conv_last1 = nn.Conv2d(embed_dim, embed_dim_temp, 3, 1, 1)
        self.conv_last2 = nn.Conv2d(embed_dim_temp, int(embed_dim_temp/2), 3, 1, 1)
        self.conv_last3 = nn.Conv2d(int(embed_dim_temp/2), num_out_ch, 3, 1, 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}
    ################## module start ###################################
    def dual_level_feature_extraction(self, x, y):  #torch.Size([1, 1, 264, 184])
        I1 = self.low_level_feature_extraction(x)
        I2 = self.low_level_feature_extraction(y)
        b,c,h,w = I2.shape
        return I1, I2, h, w

    def dual_dwtfuse(self, feature1, feature2):  #torch.Size([1, 1, 264, 184])
        high3_a, low_a =  self.dwt(feature1) # torch.Size([1, 192, 32, 32]) ; torch.Size([1, 64, 32, 32])
        high3_b, low_b =  self.dwt(feature2)
        #import pdb; pdb.set_trace()   #调试内容

        Hi = high3_a + high3_b
        Lo = low_a  + low_b 
        # Hi = self.down_hi(torch.concat([high3_a,high3_b],dim=1))
        # Lo = self.down_low(torch.concat([low_a,low_b],dim=1))
        b,c,h,w = Lo.shape
        return Hi, Lo, h, w
    

    def fused_img_recon(self, x,h,w):
        x_size = (h, w)
        # x = self.patch_embed(x)
        # -------------------mamba------------------ #
        x= self.FreqFusion(x, h,w)
  
        x = to_4d(x, h,w)
        
        # -------------------Convolution------------------- #
        x = self.lrelu(self.conv_last1(x))
        x = self.lrelu(self.conv_last2(x))
        x = self.conv_last3(x) 
        return x

    def forward(self, A, B):  # A: torch.Size([1, 1, 128, 128])
        # import pdb;pdb.set_trace()
        x = A
        y = B
        H, W = x.shape[2:]
        
        self.mean_A = self.mean.type_as(x)
        self.mean_B = self.mean.type_as(y)
        self.mean = (self.mean_A + self.mean_B) / 2

        x = (x - self.mean_A) * self.img_range
        y = (y - self.mean_B) * self.img_range

        ################################### 1, Dual_level_feature_extraction ###################################
        feature1, feature2, h0, w0 = self.dual_level_feature_extraction(x,y)

        ################################### 2, DWT Trans & Fuse  ###################################
        Hi, Lo, h, w = self.dual_dwtfuse(feature1, feature2)

        ################################### 3, High- Low- Freq Fusion ###################################
        Hi = to_3d(Hi)
        Lo = to_3d(Lo)

        Fusion_Hi = self.HiFreqFusion(Hi, h, w)
        Fusion_Lo = self.LoFreqFusion(Lo, h, w)

        Fusion_Hi = to_4d(Fusion_Hi, h,w)
        Fusion_Lo = to_4d(Fusion_Lo, h,w)

        # Hi: torch.Size([1, 192, 32, 32]);  Lo: torch.Size([1, 64, 32, 32])
        ################################### 4, IDWT ###################################
        fuse = self.idwt(Fusion_Hi, Fusion_Lo)
        # import pdb; pdb.set_trace()   # 调试内容

        # torch.Size([1, 64, 64, 64])x
        # import pdb;pdb.set_trace()
        b ,c ,H , W = fuse.shape
        # Fused_image_reconstruction
        fuse = to_3d(fuse)
        # import pdb;pdb.set_trace()
        x = self.fused_img_recon(fuse,H,W)                        
        # import pdb; pdb.set_trace()

        x = x / self.img_range + self.mean

        return x[:, :, :H*self.upscale, :W*self.upscale]
    
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    upscale = 4
    window_size = 8
    height = (1024 // upscale // window_size + 1) * window_size
    width = (720 // upscale // window_size + 1) * window_size
    model = MambaDFuse(upscale=2, img_size=(height, width),
                   window_size=window_size, img_range=1., depths=[6, 6, 6, 6],
                   embed_dim=48, num_heads=[6, 6, 6, 6]).to(device) 
    
    # model = FuseMLP(64).to(device) 
    # num_params = sum([p.numel() for p in model.parameters()]) / 1e6
    # print(f'Parameter number: {num_params:.2f} M')
    a = torch.randn((1, 1, 64, 64)).to(device) 
    b = torch.randn((1, 1, 64, 64)).to(device) 

    x = model(a,b)
    print(x.shape)

    import time
    from thop import profile 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda = True if torch.cuda.is_available() else False
    
    # Model initialization
    # model = ITFuse().to(device)
    input_data1 = torch.randn(1, 1, 512, 512).to(device)
    input_data2 = torch.randn(1, 1, 512, 512).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f'Learnable parameter number: {num_params:.2f} M')
    # Calculate FLOPs in gigaflops (G)
    flops, params = profile(model, inputs=(input_data1, input_data2))
    print(f'FLOPs: {flops / 1e9:.2f} G')
    print(f'Parameters: {params / 1e6:.2f} M')


    # Inference latency over 10 runs
    num_runs = 10
    total_time = 0
    
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = model(input_data1, input_data2)
            end_time = time.time()
            total_time += (end_time - start_time)
    
    avg_inference_time = total_time / num_runs
    print(f"\nAverage inference time: {avg_inference_time * 1000:.2f} ms")
   