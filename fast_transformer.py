"""
Adapted from https://github.com/lukemelas/simple-bert
"""

import numpy as np
import torch
from torch import nn
from torch import Tensor 
from torch.nn import functional as F
import numerator_and_denominator as num_and_den
from fast_transformers.attention.linear_attention import LinearAttention
from fast_transformers.masking import LengthMask, FullMask

def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., attention_type='nystrom', seq_len=2048, num_landmarks=256, kernel_size=33):
        super().__init__()
        self.num_heads = num_heads
        self._hidden_dim = dim // num_heads
        self.head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attention_type = attention_type
        # feature_map: callable, a callable that applies the feature map
        # to the last dimension of a tensor (default: elu(x)+1) -> same as paper
        if attention_type == 'linear':
            self.attn = LinearAttention(dim, feature_map=None, eps=1e-06, event_dispatcher='')

        if attention_type == 'nystrom':
            self.seq_len = seq_len
            self.num_landmarks = num_landmarks
            self.conv = nn.Conv2d(
                in_channels = self.num_heads, out_channels = self.num_heads,
                kernel_size = (kernel_size, 1), padding = (kernel_size // 2, 0),
                bias = False,
                groups = self.num_heads)

    def forward(self, x):
        if self.attention_type == 'classical':
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x

        elif self.attention_type == 'linear': # transformer are RNNs
            B, N, C = x.shape # 2, 197, 768 batch sequence d_model
            # Changed permutation ordering to make compatible with linear attention library
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
            # how many queries each sequence in the batch consists of
            length_mask = LengthMask(x.new_full((B,), N, dtype=torch.int64))
            # where each query attend to, we set it to all/ full mask
            attn_mask = FullMask(N, device=x.device)
            # last two length_mask are for query and key length, which are equal in this case
            # for cross attention they would be different
            x = self.attn(q, k, v, attn_mask, length_mask, length_mask)
            x = x.view(B, N, -1) # torch.Size([2, 197, 768])
            x = self.proj(x)
            x = self.proj_drop(x)
            return x

        elif self.attention_type == 'nystrom':  # Nystromformer
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            Q, K, V = qkv[0]/self.scale, qkv[1]/self.scale, qkv[2]  # make torchscript happy (cannot use tensor as tuple)

            Q_landmarks = torch.cat((Q[...,0:1,:], Q[...,1:,:].reshape(-1, self.num_heads, self.num_landmarks, self.seq_len // self.num_landmarks, self.head_dim).mean(dim = -2)), 2)
            K_landmarks = torch.cat((K[...,0:1,:], K[...,1:,:].reshape(-1, self.num_heads, self.num_landmarks, self.seq_len // self.num_landmarks, self.head_dim).mean(dim = -2)), 2)

            kernel_1 = F.softmax(torch.matmul(Q, K_landmarks.transpose(-1, -2)), dim = -1)
            kernel_2 = F.softmax(torch.matmul(Q_landmarks, K_landmarks.transpose(-1, -2)), dim = -1)
            kernel_3 = F.softmax(torch.matmul(Q_landmarks, K.transpose(-1, -2)) , dim = -1)
            X = torch.matmul(torch.matmul(kernel_1, self.iterative_inv(kernel_2)), torch.matmul(kernel_3, V))

            X += self.conv(V) # TODO: this may be optional according to the Nystrom paper

            return X.permute(0,2,1,3).reshape(B,N,C)

    def iterative_inv(self, mat, n_iter = 6):
        I = torch.eye(mat.size(-1), device = mat.device)
        K = mat
        V = 1 / (torch.max(torch.sum(torch.abs(K), dim = -2)) * torch.max(torch.sum(torch.abs(K), dim = -1))) * K.transpose(-1, -2)
        for _ in range(n_iter):
            KV = torch.matmul(K, V)
            V = torch.matmul(0.25 * V, 13 * I - torch.matmul(KV, 15 * I - torch.matmul(KV, 7 * I - KV)))
        return V


# From timm
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

# From timm
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

# From timm
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# From timm
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, attention_type='nystrom', seq_len=None, num_landmarks=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, attention_type=attention_type,
                              seq_len=seq_len, num_landmarks=num_landmarks)
        # self.attn = MultiHeadedSelfAttention(dim, num_heads, drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


