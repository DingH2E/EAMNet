import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple
from timm.models.registry import register_model

from einops import rearrange, einsum

"""
    Graph Transformer Layer with edge features

"""


"""
    Single Attention Head
"""


def get_conv2d(in_channels,
               out_channels,
               kernel_size,
               stride,
               padding,
               dilation,
               groups,
               bias,
               attempt_use_lk_impl=True):
    kernel_size = to_2tuple(kernel_size)
    if padding is None:
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    else:
        padding = to_2tuple(padding)
    need_large_impl = kernel_size[0] == kernel_size[1] and kernel_size[0] > 5 and padding == (
    kernel_size[0] // 2, kernel_size[1] // 2)

    # if attempt_use_lk_impl and need_large_impl:
    #     print('---------------- trying to import iGEMM implementation for large-kernel conv')
    #     try:
    #         from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
    #         print('---------------- found iGEMM implementation ')
    #     except:
    #         DepthWiseConv2dImplicitGEMM = None
    #         print(
    #             '---------------- found no iGEMM. use original conv. follow https://github.com/AILab-CVC/UniRepLKNet to install it.')
    #     if DepthWiseConv2dImplicitGEMM is not None and need_large_impl and in_channels == out_channels \
    #             and out_channels == groups and stride == 1 and dilation == 1:
    #         print(f'===== iGEMM Efficient Conv Impl, channels {in_channels}, kernel size {kernel_size} =====')
    #         return DepthWiseConv2dImplicitGEMM(in_channels, kernel_size, bias=bias)

    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size,
                     stride=stride,
                     padding=padding,
                     dilation=dilation,
                     groups=groups,
                     bias=bias)


def get_bn(dim, use_sync_bn=False):
    if use_sync_bn:
        return nn.SyncBatchNorm(dim)
    else:
        return nn.BatchNorm2d(dim)


def fuse_bn(conv, bn):
    conv_bias = 0 if conv.bias is None else conv.bias
    std = (bn.running_var + bn.eps).sqrt()
    return conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1), bn.bias + (conv_bias - bn.running_mean) * bn.weight / std

def convert_dilated_to_nondilated(kernel, dilate_rate):
    identity_kernel = torch.ones((1, 1, 1, 1)).to(kernel.device)
    if kernel.size(1) == 1:
        #   This is a DW kernel
        dilated = F.conv_transpose2d(kernel, identity_kernel, stride=dilate_rate)
        return dilated
    else:
        #   This is a dense or group-wise (but not DW) kernel
        slices = []
        for i in range(kernel.size(1)):
            dilated = F.conv_transpose2d(kernel[:,i:i+1,:,:], identity_kernel, stride=dilate_rate)
            slices.append(dilated)
        return torch.cat(slices, dim=1)

def merge_dilated_into_large_kernel(large_kernel, dilated_kernel, dilated_r):
    large_k = large_kernel.size(2)
    dilated_k = dilated_kernel.size(2)
    equivalent_kernel_size = dilated_r * (dilated_k - 1) + 1
    equivalent_kernel = convert_dilated_to_nondilated(dilated_kernel, dilated_r)
    rows_to_pad = large_k // 2 - equivalent_kernel_size // 2
    merged_kernel = large_kernel + F.pad(equivalent_kernel, [rows_to_pad] * 4)
    return merged_kernel


class LayerNorm2d(nn.LayerNorm):
    def __init__(self, dim):
        super().__init__(normalized_shape=dim, eps=1e-6)

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c')
        x = super().forward(x)
        x = rearrange(x, 'b h w c -> b c h w')
        return x.contiguous()

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias):
        super().__init__()

        self.out_dim = out_dim
        self.num_heads = num_heads

        if use_bias:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.proj_e = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        else:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.proj_e = nn.Linear(in_dim, out_dim * num_heads, bias=False)



    def propagate_attention(self, q, k, v):
        # Compute attention score
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.out_dim ** 0.5)
        # R = self.relative_positional_encoding(q.size(1), q.size(2)) # [100, 128]
        # print("***** graph transformer *****")
        # print("scores: ", scores)
        # print("e: ", e)
        # e_scores = torch.matmul(e, e.transpose(-2, -1)) / (self.out_dim ** 0.5)     # [200, 8, 8]
        # scores = scores * e_scores
        # # print("scores: ", scores.shape)
        # # print("blocker: ", self.blocker)
        # scores = self.blocker(scores)
        # print("scores: ", scores)
        # print("***** graph transformer *****")
        # assert False
        # g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'))  # , edges)

        # scaling
        # g.apply_edges(scaling('score', np.sqrt(self.out_dim)))

        # Use available edge features to modify the scores
        # g.apply_edges(imp_exp_attn('score', 'proj_e'))

        # print("e: ", e.size())
        # scores = scores + e

        # Copy edge features as e_out to be passed to FFN_e
        # g.apply_edges(out_edge_features('score'))

        # softmax
        attention_weights = F.softmax(scores, dim=-1)   # [200, 8, 8]
        # print(kd_loss)
        # assert False

        # Send weighted values to target nodes
        # print("attention_weights: ", attention_weights.size())
        # print("v: ", v.size())
        # out = torch.matmul(attention_weights, v)
        # eids = g.edges()
        # g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
        # g.send_and_recv(eids, fn.copy_edge('score', 'score'), fn.sum('score', 'z'))

        out = torch.matmul(attention_weights, v).transpose(1, 2).contiguous()
        return out

    def forward(self, h, s):
        # print("***** MultiHeadAttentionLayer *****")
        # print("shape of h:", h.shape)
        # print("Q:", self.Q)

        # assert False
        h = h.view(h.size(0), h.size(1), -1).permute(0, 2, 1)  # [2, 128, 2500]
        s = s.view(s.size(0), s.size(1), -1).permute(0, 2, 1)  # [2, 128, 2500]
        # pe = self.positional_encoding(h.size(1), h.size(2)) # [100, 128]
        # pe = pe.unsqueeze(0).expand_as(h)
        # h = h + pe
        # e = e + pe
        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(s)

        # Reshaping into [num_nodes, num_heads, feat_dim] to
        # get projections for multi-head attention
        Q_h = Q_h.view(-1, self.num_heads, self.out_dim)    # [2*100, 8, 16]
        K_h = K_h.view(-1, self.num_heads, self.out_dim)
        V_h = V_h.view(-1, self.num_heads, self.out_dim)

        # print("shape of Q_h", Q_h.shape)
        # print("***** MultiHeadAttentionLayer *****")
        # assert False
        return self.propagate_attention(Q_h, K_h, V_h)


class DynamicTransformerLayer(nn.Module):
    """
        Param:
    """

    def __init__(self, in_dim, out_dim, num_heads, dropout=0.0, layer_norm=False, batch_norm=True, residual=True,
                 use_bias=False):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        self.attention = MultiHeadAttentionLayer(in_dim, out_dim // num_heads, num_heads, use_bias)

        self.O_h = nn.Linear(out_dim, out_dim)
        self.O_e = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
            self.layer_norm1_e = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim)
            self.batch_norm1_e = nn.BatchNorm1d(out_dim)

        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_h_layer2 = nn.Linear(out_dim * 2, out_dim)

        # FFN for e
        self.FFN_e_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_e_layer2 = nn.Linear(out_dim * 2, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)
            self.layer_norm2_e = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim)
            self.batch_norm2_e = nn.BatchNorm1d(out_dim)

        self.blocks1 = nn.ModuleList()
        for i in range(2):
            self.blocks1.append(
                RepConvBlock(dim=in_dim)
            )
        self.lepe = nn.Sequential(
            DilatedReparamBlock(in_dim, kernel_size=7, deploy=False, use_sync_bn=False,
                                attempt_use_lk_impl=False),
            nn.BatchNorm2d(in_dim),
        )

        self.se_layer = SEModule(in_dim)

        self.gate = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.SiLU(),
        )
        self.proj = nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.Conv2d(in_dim, out_dim, kernel_size=1),
        )

        self.dyconv_proj = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_dim),
        )

    def forward(self, h, s):
        raw_h = h
        b, c, he, wi = h.shape
        for conv in self.blocks1:
            h = conv(h)
        gate = self.gate(h)
        lepe = self.lepe(h)
        h_in1 = self.dyconv_proj(h)  # for first residual connection [2, 128, 100]
        h_in1 = rearrange(h_in1, 'b c h w -> b (h w) c', h=he, w=wi)
        # print("h_in1=", h_in1.shape)

        # multi-head attention out
        h_attn_out = self.attention(h, s)   # [2, 128, 100]

        # print("h:", h.shape)
        # print("h_attn_out:", h_attn_out.shape)
        h = h_attn_out.view(h.size(0), h.size(1), -1)
        h = rearrange(h, 'b c (h w) -> b c h w', h=he, w=wi)
        # print("h_attn_out_reshape:", h.shape)

        # h = F.dropout(h, self.dropout, training=self.training)

        h = h + lepe
        h = self.se_layer(h)

        h = gate * h
        h = self.proj(h)

        h = rearrange(h, 'b c h w -> b (h w) c', h=he, w=wi)

        # print("h=", h.shape)
        # print("h_attn_out=", h_attn_out.shape)
        # print("O_h=", self.O_h)
        # assert False
        h = self.O_h(h)
        # h = rearrange(h, 'b (h w) c-> b c h w', h=he, w=wi)
        # print("h=", h.shape)

        # print("h_in1:", h_in1.shape)
        # assert False


        if self.residual:
            h = h_in1 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm1_h(h)

        if self.batch_norm:
            h = self.batch_norm1_h(h.permute(0, 2, 1)).permute(0, 2, 1)

        h_tmp = rearrange(h, 'b (h w) c-> b c h w', h=he, w=wi)
        h_in2 = self.dyconv_proj(h_tmp)  # for second residual connection
        h_in2 = rearrange(h_in2, 'b c h w -> b (h w) c', h=he, w=wi)

        # FFN for h
        h = self.FFN_h_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        if self.residual:
            h = h_in2 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm2_h(h)

        if self.batch_norm:
            h = self.batch_norm2_h(h.permute(0, 2, 1))
        h = rearrange(h, 'b c (h w)-> b c h w', h=he, w=wi)
        for conv in self.blocks1:
            h = conv(h)
        return h


class DilatedReparamBlock(nn.Module):
    """
    Dilated Reparam Block proposed in UniRepLKNet (https://github.com/AILab-CVC/UniRepLKNet)
    We assume the inputs to this block are (N, C, H, W)
    """

    def __init__(self, channels, kernel_size, deploy, use_sync_bn=False, attempt_use_lk_impl=True):
        super().__init__()
        self.lk_origin = get_conv2d(channels, channels, kernel_size, stride=1,
                                    padding=kernel_size // 2, dilation=1, groups=channels, bias=deploy,
                                    attempt_use_lk_impl=attempt_use_lk_impl)
        self.attempt_use_lk_impl = attempt_use_lk_impl

        #   Default settings. We did not tune them carefully. Different settings may work better.
        if kernel_size == 19:
            self.kernel_sizes = [5, 7, 9, 9, 3, 3, 3]
            self.dilates = [1, 1, 1, 2, 4, 5, 7]
        elif kernel_size == 17:
            self.kernel_sizes = [5, 7, 9, 3, 3, 3]
            self.dilates = [1, 1, 2, 4, 5, 7]
        elif kernel_size == 15:
            self.kernel_sizes = [5, 7, 7, 3, 3, 3]
            self.dilates = [1, 1, 2, 3, 5, 7]
        elif kernel_size == 13:
            self.kernel_sizes = [5, 7, 7, 3, 3, 3]
            self.dilates = [1, 1, 2, 3, 4, 5]
        elif kernel_size == 11:
            self.kernel_sizes = [5, 7, 5, 3, 3, 3]
            self.dilates = [1, 1, 2, 3, 4, 5]
        elif kernel_size == 9:
            self.kernel_sizes = [5, 7, 5, 3, 3]
            self.dilates = [1, 1, 2, 3, 4]
        elif kernel_size == 7:
            self.kernel_sizes = [5, 3, 3, 3]
            self.dilates = [1, 1, 2, 3]
        elif kernel_size == 5:
            self.kernel_sizes = [3, 3]
            self.dilates = [1, 2]
        else:
            raise ValueError('Dilated Reparam Block requires kernel_size >= 5')

        if not deploy:
            self.origin_bn = get_bn(channels, use_sync_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__setattr__('dil_conv_k{}_{}'.format(k, r),
                                 nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=k, stride=1,
                                           padding=(r * (k - 1) + 1) // 2, dilation=r, groups=channels,
                                           bias=False))
                self.__setattr__('dil_bn_k{}_{}'.format(k, r), get_bn(channels, use_sync_bn=use_sync_bn))

    def forward(self, x):
        if not hasattr(self, 'origin_bn'):  # deploy mode
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
            bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
            out = out + bn(conv(x))
        return out

    def merge_dilated_branches(self):
        if hasattr(self, 'origin_bn'):
            origin_k, origin_b = fuse_bn(self.lk_origin, self.origin_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
                bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
                branch_k, branch_b = fuse_bn(conv, bn)
                origin_k = merge_dilated_into_large_kernel(origin_k, branch_k, r)
                origin_b += branch_b
            merged_conv = get_conv2d(origin_k.size(0), origin_k.size(0), origin_k.size(2), stride=1,
                                     padding=origin_k.size(2) // 2, dilation=1, groups=origin_k.size(0), bias=True,
                                     attempt_use_lk_impl=self.attempt_use_lk_impl)
            merged_conv.weight.data = origin_k
            merged_conv.bias.data = origin_b
            self.lk_origin = merged_conv
            self.__delattr__('origin_bn')
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__delattr__('dil_conv_k{}_{}'.format(k, r))
                self.__delattr__('dil_bn_k{}_{}'.format(k, r))

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    Originally proposed in ConvNeXt V2 (https://arxiv.org/abs/2301.00808)
    This implementation is more efficient than the original (https://github.com/facebookresearch/ConvNeXt-V2)
    We assume the inputs to this layer are (N, C, H, W)
    """
    def __init__(self, dim, use_bias=True):
        super().__init__()
        self.use_bias = use_bias
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        if self.use_bias:
            self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(-1, -2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        # print("Nx=",Nx.shape)
        # assert False
        if self.use_bias:
            return (self.gamma * Nx + 1) * x + self.beta
        else:
            return (self.gamma * Nx + 1) * x


class SEModule(nn.Module):
    def __init__(self, dim, red=8, inner_act=nn.GELU, out_act=nn.Sigmoid):
        super().__init__()
        inner_dim = max(16, dim // red)
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, inner_dim, kernel_size=1),
            inner_act(),
            nn.Conv2d(inner_dim, dim, kernel_size=1),
            out_act(),
        )

    def forward(self, x):
        x = x * self.proj(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, 1, 1, 1)*init_value,
                                   requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(dim), requires_grad=True)

    def forward(self, x):
        x = F.conv2d(x, weight=self.weight, bias=self.bias, groups=x.shape[1])
        return x


class ResDWConv(nn.Conv2d):
    '''
    Depthwise convolution with residual connection
    '''

    def __init__(self, dim, kernel_size=3):
        super().__init__(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)

    def forward(self, x):
        x = x + super().forward(x)
        return x

class RepConvBlock(nn.Module):

    def __init__(self,
                 dim=64,
                 kernel_size=7,
                 mlp_ratio=4,
                 ls_init_value=None,
                 res_scale=False,
                 drop_path=0,
                 norm_layer=LayerNorm2d,
                 use_gemm=False,
                 deploy=False,
                 use_checkpoint=False):
        super().__init__()

        self.res_scale = res_scale
        self.use_checkpoint = use_checkpoint

        mlp_dim = int(dim * mlp_ratio)

        self.dwconv = ResDWConv(dim, kernel_size=3)

        self.proj = nn.Sequential(
            norm_layer(dim),
            DilatedReparamBlock(dim, kernel_size=kernel_size, deploy=deploy, use_sync_bn=False,
                                attempt_use_lk_impl=use_gemm),
            nn.BatchNorm2d(dim),
            SEModule(dim),
            nn.Conv2d(dim, mlp_dim, kernel_size=1),
            nn.GELU(),
            ResDWConv(mlp_dim, kernel_size=3),
            GRN(mlp_dim),
            nn.Conv2d(mlp_dim, dim, kernel_size=1),
            DropPath(drop_path) if drop_path > 0 else nn.Identity(),
        )

        self.ls = LayerScale(dim, init_value=ls_init_value) if ls_init_value is not None else nn.Identity()

    def forward_features(self, x):

        x = self.dwconv(x)

        if self.res_scale:
            x = self.ls(x) + self.proj(x)
        else:
            drop_path = self.proj[-1]
            x = x + drop_path(self.ls(self.proj[:-1](x)))

        return x

    def forward(self, x):
        x = self.forward_features(x)
        return x


class SEModule(nn.Module):
    def __init__(self, dim, red=8, inner_act=nn.GELU, out_act=nn.Sigmoid):
        super().__init__()
        inner_dim = max(16, dim // red)
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, inner_dim, kernel_size=1),
            inner_act(),
            nn.Conv2d(inner_dim, dim, kernel_size=1),
            out_act(),
        )

    def forward(self, x):
        x = x * self.proj(x)
        return x




    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                                                                   self.in_channels,
                                                                                   self.out_channels, self.num_heads,
                                                                                   self.residual)