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
    r""" Image to Patch Unembedding

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

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops

################################# model
class SCAM(nn.Module):
    '''
    Stereo Cross Attention Module (SCAM)
    '''
    def __init__(self, c):
        super().__init__()
        self.scale = c ** -0.5
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.l_proj1 = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
            nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1, groups=c, bias=True),
        )
        self.r_proj1 = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
            nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1, groups=c, bias=True),
        )
        self.l_proj2 = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
        )
        self.r_proj2 = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
        )
        self.se1 = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
            nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1, groups=c, bias=True),
        )
        self.se2 = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
            nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1, groups=c, bias=True),
        )

    def forward(self, x_l, x_r):
        Q_l = self.se1(x_l).permute(0, 2, 3, 1)         # torch.Size([4, 256, 256, 16])
        Q_r_T = self.se2(x_r).permute(0, 2, 1, 3)       # torch.Size([4, 256, 16, 256])
        V_l = self.l_proj1(x_l).permute(0, 2, 3, 1)  # B, H, W, c
        V_r = self.r_proj1(x_r).permute(0, 2, 3, 1)  # B, H, W, c

        # (B, H, W, c) x (B, H, c, W) -> (B, H, W, W)
        attention = torch.softmax(torch.matmul(Q_l, Q_r_T), dim=-1) * self.scale        # torch.matmul(Q_l, Q_r_T)： torch.Size([4, 256, 256, 256]) 
        F_r2l = torch.matmul(attention, V_r)  #B, H, W, c
        F_l2r = torch.matmul(attention.permute(0, 1, 3, 2), V_l) #B, H, W, c

        F_r2l = self.l_proj2(F_r2l.permute(0, 3, 1, 2)) * self.beta
        F_l2r = self.r_proj2(F_l2r.permute(0, 3, 1, 2)) * self.gamma

        return x_l + F_r2l, x_r + F_l2r



class ChannelExchange(nn.Module):
    def __init__(self, p=2):
        super().__init__()
        self.p = p
    def forward(self, x1, x2):
        x1 = x1.permute(0, 2, 1)
        x2 = x2.permute(0, 2, 1)

        B, C, N = x1.shape
        exchange_mask = torch.arange(C) % self.p == 0
        exchange_mask = exchange_mask.unsqueeze(0).expand((B, -1))
        out_x1, out_x2 = torch.zeros_like(x1), torch.zeros_like(x2)
        out_x1[~exchange_mask, ...] = x1[~exchange_mask, ...]
        out_x2[~exchange_mask, ...] = x2[~exchange_mask, ...]
        out_x1[exchange_mask, ...] = x2[exchange_mask, ...]
        out_x2[exchange_mask, ...] = x1[exchange_mask, ...]
        out_x1 = out_x1.permute(0, 2, 1)
        out_x2 = out_x2.permute(0, 2, 1)
        return out_x1, out_x2

class HighLevelFeatureExtraction(nn.Module):
    def __init__(self, embed_dim, num_blocks=8):
        super(HighLevelFeatureExtraction, self).__init__()
        self.blocks = nn.ModuleList([SingleMambaBlock(embed_dim) for _ in range(num_blocks)])

    def forward(self, x, residual, h, w):
        for block in self.blocks:
            x, residual = block(x, residual, h, w)
        return x, residual
    
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


class SingleMambaBlock(nn.Module):
    def __init__(self, dim):
        super(SingleMambaBlock, self).__init__()
        self.encoder = Mamba2_2d(dim)
        self.norm1 = LayerNorm(dim,'with_bias')
        self.norm2 = LayerNorm(dim,'with_bias')
        self.mlp = Mlp(dim, dim * 2, dim)

    def forward(self,x, residual, H, W):
        residual = x
        x1 = self.norm1(residual)
        # B,HW,C = x.shape
        x_m = residual + self.encoder(x1,H,W)
        
        residual2 = x_m
        x2 = self.norm2(residual2)
        x_m2 = residual2 + self.mlp(x2)
        return (x_m2,residual2)

class TokenSwapMamba(nn.Module):
    def __init__(self, dim):
        super(TokenSwapMamba, self).__init__()
        self.I1encoder = Mamba2_2d(dim)
        self.I2encoder = Mamba2_2d(dim)
        self.norm1 = LayerNorm(dim,'with_bias')
        self.norm2 = LayerNorm(dim,'with_bias')
        self.ChannelExchange = ChannelExchange(p=2)
    def forward(self, I1,I2
                ,I1_residual,I2_residual,H,W):

        I1_residual = I1+I1_residual
        I2_residual = I2+I2_residual
        I1 = self.norm1(I1_residual)
        I2 = self.norm2(I2_residual)
        # B,HW,C = I1.shape
 
        I1_swap, I2_swap = self.ChannelExchange(I1, I2) 
        I1_swap = self.I1encoder(I1_swap, H, W)
        I2_swap = self.I2encoder(I2_swap, H, W)
        return I1_swap,I2_swap,I1_residual,I2_residual


class M3(nn.Module):
    def __init__(self, dim):
        super(M3, self).__init__()
        self.multi_modal_mamba_block = Mamba2_2d(dim)
        self.norm1 = LayerNorm(dim,'with_bias')
        self.norm2 = LayerNorm(dim,'with_bias')
        self.norm3 = LayerNorm(dim,'with_bias')
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self,I1,fusion_resi,I2,fusion,test_h,test_w):
        fusion_resi = fusion + fusion_resi
        fusion = self.norm1(fusion_resi)
        I2 = self.norm2(I2)
        I1 = self.norm3(I1)
        B,L,C = I1.shape
        global_f = self.multi_modal_mamba_block(self.norm1(fusion), test_h, test_w )

        # B,HW,C = global_f.shape
        fusion = global_f.transpose(1, 2).view(B, C, test_h, test_w)
        fusion =  (self.dwconv(fusion)+fusion).flatten(2).transpose(1, 2)
        return fusion,fusion_resi


################## Network
class MambaDFuse(nn.Module):
   
    def __init__(self, img_size=64, patch_size=1, in_chans=1,
                 embed_dim=96, Ex_depths=[4], Fusion_depths=[2, 2], Re_depths=[4], 
                 Ex_num_heads=[6], Fusion_num_heads=[6, 6], Re_num_heads=[6],
                 window_size=7,qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 upscale=1, img_range=1., resi_connection='1conv',
                 **kwargs):
        super(MambaDFuse, self).__init__()
        num_out_ch = in_chans
        num_feat = 64
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
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
    
        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.softmax = nn.Softmax(dim=0)
        # absolute position embedding
        if self.ape: 
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.apply(self._init_weights)

        #####################################################################################################
        ################################### 1, low-level feature extraction ###################################
        self.low_level_feature_extraction1 = nn.Conv2d(in_chans, embed_dim_temp, 3, 1, 1)
        self.low_level_feature_extraction2 = nn.Conv2d(embed_dim_temp, embed_dim, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        #####################################################################################################
        ################################### 2, deep feature extraction ######################################
        self.high_level_feature_extraction1 = HighLevelFeatureExtraction(self.embed_dim)
        self.high_level_feature_extraction2 = HighLevelFeatureExtraction(self.embed_dim)
        
        #####################################################################################################
        ################################### 3, shallow feature fusion ######################################
        self.channel_exchange1 = TokenSwapMamba(self.embed_dim)
        self.channel_exchange2 = TokenSwapMamba(self.embed_dim)
        self.shallow_fusion1 = nn.Conv2d(self.embed_dim*2,self.embed_dim,3,1,1)
        self.shallow_fusion2 = nn.Conv2d(self.embed_dim*2,self.embed_dim,3,1,1)
        
        #####################################################################################################
        ################################### 4, deep feature fusion ######################################
        self.M3_block1= M3(self.embed_dim)
        self.M3_block2 = M3(self.embed_dim)
        self.M3_block3 = M3(self.embed_dim)
        self.M3_block4 = M3(self.embed_dim)
        self.M3_block5 = M3(self.embed_dim)
        #####################################################################################################
        ################################ 5, fused image reconstruction ################################
        self.feature_re = HighLevelFeatureExtraction(self.embed_dim)
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

    def dual_level_feature_extraction(self, x, y):
        I1 = self.lrelu(self.low_level_feature_extraction1(x))
        I1 = self.lrelu(self.low_level_feature_extraction2(I1))   # torch.Size([1, 60, 128, 128])


        I2 = self.lrelu(self.low_level_feature_extraction1(y))
        I2 = self.lrelu(self.low_level_feature_extraction2(I2))   
        b,c,h,w = I2.shape
        
        x_size = (I1.shape[2], I1.shape[3])
        I1 = self.patch_embed(I1)
        I2 = self.patch_embed(I2)
   
        if self.ape:
            I1 = I1 + self.absolute_pos_embed
            I2 = I2 + self.absolute_pos_embed
        I1 = self.pos_drop(I1)
        I2 = self.pos_drop(I2)

        residual_I1_f = 0
        residual_I2_f = 0
        I1,residual_I1_f = self.high_level_feature_extraction1(I1,residual_I1_f, h, w)
        I2,residual_I2_f = self.high_level_feature_extraction2(I2,residual_I2_f, h, w)
        
        return I1, residual_I1_f, I2, residual_I2_f, h, w
    
    def dual_phase_feature_fusion(self, x, x_residual, y, y_residual, h, w):
        # --------------------Shallow Fuse Module-------------------- #
        I1, I2, I1_residual, I2_residual = self.channel_exchange1(x, y, x_residual, y_residual,h,w)
        I1, I2, I1_residual, I2_residual = self.channel_exchange2(I1, I2, I1_residual, I2_residual,h,w)
        I1 = self.patch_unembed(I1, (h,w))
        I2 = self.patch_unembed(I2, (h,w))

        I1 = self.shallow_fusion1(torch.concat([I1,I2],dim=1)) + I1
        I2 = self.shallow_fusion2(torch.concat([I2,I1],dim=1)) + I2
        
        # add 
        fusion_f = (I1 + I2)/2

        test_h, test_w = I1.shape[2], I1.shape[3]

        I1 = self.patch_embed(I1)
        I2 = self.patch_embed(I2)
        fusion_f = self.patch_embed(fusion_f)
        
        # --------------------Deep Fuse Module-------------------- #
        residual_fusion_f = 0
        fusion_f,residual_fusion_f = self.M3_block1(I1,residual_fusion_f,I2,fusion_f,test_h,test_w)
        fusion_f,residual_fusion_f = self.M3_block2(I1,residual_fusion_f,I2,fusion_f,test_h,test_w)
        fusion_f,residual_fusion_f = self.M3_block3(I1,residual_fusion_f,I2,fusion_f,test_h,test_w)
        fusion_f,residual_fusion_f = self.M3_block4(I1,residual_fusion_f,I2,fusion_f,test_h,test_w)
        fusion_f,residual_fusion_f = self.M3_block5(I1,residual_fusion_f,I2,fusion_f,test_h,test_w)

        fusion_f = self.patch_unembed(fusion_f,(h,w))
        return fusion_f

    def fused_img_recon(self, x,h,w):

        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        # -------------------mamba------------------ #
        residual_re = 0
        x,residual_re = self.feature_re(x,residual_re,h,w)
  
        x = self.patch_unembed(x, x_size)
        
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
        # torch.Size([1, 1, 128, 128])
        # Dual_level_feature_extraction
        feature1, residual_feature1, feature2, residual_feature2, h, w = self.dual_level_feature_extraction(x,y)
        
        # Dual_phase_feature_fusion
        fusion_feature = self.dual_phase_feature_fusion(feature1, residual_feature1, feature2, residual_feature2, h, w)

        # Fused_image_reconstruction
        x = self.fused_img_recon(fusion_feature,H,W)                        
    
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
                   embed_dim=64, num_heads=[6, 6, 6, 6]).to(device) 
    

    a = torch.randn((1, 1, height, width)).to(device) 
    b = torch.randn((1, 1, height, width)).to(device) 

    x = model(a,b)
    print(x.shape)
   