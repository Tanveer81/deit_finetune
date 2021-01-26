"""
Adapted from https://github.com/lukemelas/simple-bert
"""

import numpy as np
import torch
from torch import nn
from torch import Tensor 
from torch.nn import functional as F
import numerator_and_denominator as num_and_den


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
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., feature_type='favor+', compute_type='iter'):
        super().__init__()
        self.num_heads = num_heads
        self._hidden_dim = dim // num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self._feature_type = feature_type
        self._compute_type = compute_type

    def forward(self, x):
        if self._feature_type == 'classical':
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

        if self._feature_type == 'favor+':
            queries, keys, values = self._get_queries_keys_values(x, self.sample_rfs(x.device))

            num_sums, den_sums = self.init_sums(x.device)

            if self._compute_type == 'iter':
                num, _ = num_and_den.num_iter(queries, keys, values, num_sums)
                den, _ = num_and_den.den_iter(queries, keys, den_sums)
            elif self._compute_type == 'ps':
                num, _ = num_and_den.num_ps(queries, keys, values, num_sums, False)
                den, _ = num_and_den.den_ps(queries, keys, den_sums, False)
            else:
                num, _ = num_and_den.num_ps(queries, keys, values, num_sums, True)
                den, _ = num_and_den.den_ps(queries, keys, den_sums, True)

            num = torch.transpose(num, 0, 1)
            den = torch.transpose(den, 0, 1)

            outputs = num / (den[Ellipsis, None] + 1e-16)
            outputs = outputs.reshape(x.shape)

            return outputs

    def init_sums(self, device):

        head_dim = self._hidden_dim

        if self._feature_type.startswith('favor+_'):
            splitted = self._feature_type.split('_')
            feature_dim = int(splitted[1]) * head_dim
        else:
            feature_dim = head_dim

        num_sums = torch.zeros([1, self.num_heads, feature_dim, head_dim],
                            device=device)
        den_sums = torch.zeros([1, self.num_heads, feature_dim], device=device)

        return num_sums, den_sums

    def incr_step(self, x, num_sums, den_sums, on_forward, rfs, on_start):

        queries, keys, values = self._get_queries_keys_values(x, rfs)

        if not on_forward:

            if on_start:
                num_sums = torch.zeros_like(num_sums)
                den_sums = torch.zeros_like(den_sums)
            elif self._compute_type == 'iter':
                num_sums = num_and_den.num_reverse_sums_iter(queries, keys, values,
                                                                num_sums)
                den_sums = num_and_den.den_reverse_sums_iter(queries, keys, den_sums)
            else:
                num_sums = num_and_den.num_reverse_sums_ps(queries, keys, values,
                                                            num_sums)
                den_sums = num_and_den.den_reverse_sums_ps(queries, keys, den_sums)

            num_sums = num_sums.detach().clone()
            num_sums.requires_grad = True
            den_sums = den_sums.detach().clone()
            den_sums.requires_grad = True

            init_num_sums = num_sums
            init_den_sums = den_sums

        if self._compute_type == 'iter':
            num, num_sums = num_and_den.num_iter(queries, keys, values, num_sums)
            den, den_sums = num_and_den.den_iter(queries, keys, den_sums)
        elif self._compute_type == 'ps':
            num, num_sums = num_and_den.num_ps(queries, keys, values, num_sums, False)
            den, den_sums = num_and_den.den_ps(queries, keys, den_sums, False)
        else:
            num, num_sums = num_and_den.num_ps(queries, keys, values, num_sums, True)
            den, den_sums = num_and_den.den_ps(queries, keys, den_sums, True)

        num = torch.transpose(num, 0, 1)
        den = torch.transpose(den, 0, 1)

        outputs = num / (den[Ellipsis, None] + 1e-16)
        outputs = outputs.reshape(x.shape)

        if on_forward:
            return outputs, num_sums, den_sums

        return outputs, init_num_sums, init_den_sums, num_sums, den_sums

    def _get_queries_keys_values(self, inputs, rfs):

        # queries = self.proj_q(inputs)
        # keys = self.proj_k(inputs)
        # values = self.proj_v(inputs)
        B, N, C = inputs.shape
        qkv = self.qkv(inputs)
        qkv = qkv.reshape(B, N, 3, C).permute(2, 0, 1, 3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        queries = queries.reshape([queries.shape[0], queries.shape[1], self.num_heads, -1])
        keys = keys.reshape([keys.shape[0], keys.shape[1], self.num_heads, -1])
        values = values.reshape([values.shape[0], values.shape[1], self.num_heads, -1])

        if self._feature_type == 'relu':
            queries = torch.nn.functional.relu(queries)
            keys = torch.nn.functional.relu(keys)
        elif self._feature_type == 'elu+1':
            queries = torch.nn.functional.elu(queries) + 1
            keys = torch.nn.functional.elu(keys) + 1
        elif self._feature_type == 'sqr':
            queries = queries**2
            keys = keys**2
        elif self._feature_type == 'abs':
            queries = torch.abs(queries)
            keys = torch.abs(keys)
        else:
            head_dim = self._hidden_dim

            queries = queries * np.power(head_dim, -0.25)
            queries = torch.einsum('ijkl,klm->ijkm', queries, rfs) - (queries**2).sum(3, keepdim=True) / 2
            queries = torch.exp(queries)

            keys = keys * np.power(head_dim, -0.25)
            keys = torch.einsum('ijkl,klm->ijkm', keys, rfs) - (keys**2).sum(
                3, keepdim=True) / 2
            keys = torch.exp(keys)

        queries = queries.transpose(0, 1)
        keys = keys.transpose(0, 1)
        values = values.transpose(0, 1)

        return queries, keys, values

    def sample_rfs(self, device):

        if not self._feature_type.startswith('favor+'):
            return None

        if self._feature_type == 'favor+':
            factor = 1
        else:
            splitted = self._feature_type.split('_')
            factor = int(splitted[1])

        head_dim = self._hidden_dim

        rfs = [[
            _sample_orth_matrix(head_dim, device)[None, Ellipsis] for _ in range(factor)
        ] for _ in range(self.num_heads)]
        rfs = [torch.cat(x, 2) for x in rfs]
        rfs = torch.cat(rfs, 0)
        rfs = rfs * np.sqrt(head_dim)

        return rfs

class MultiHeadedSelfAttention(nn.Module):
    """Multi-Headed Dot Product Attention"""
    def __init__(self, dim, num_heads, dropout, feature_type='favor+', compute_type='iter'):
        super().__init__()
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self._n_heads = num_heads
        self._hidden_dim = dim
        self._feature_type = feature_type
        self._compute_type = compute_type
        self.scores = None # for visualization

    def forward(self, x, mask):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(_n_heads), W(width of head)) ; D = H * W
        """
        if self._feature_type == 'classical':
            # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
            q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
            q, k, v = (split_last(x, (self._n_heads, -1)).transpose(1, 2) for x in [q, k, v])
            # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
            scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
            if mask is not None:
                mask = mask[:, None, None, :].float()
                scores -= 10000.0 * (1.0 - mask)
            scores = self.drop(F.softmax(scores, dim=-1))
            # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
            h = (scores @ v).transpose(1, 2).contiguous()
            # -merge-> (B, S, D)
            h = merge_last(h, 2)
            self.scores = scores
            return h

        # fast linear attention layer
        if self._feature_type == 'favor+':

            queries, keys, values = self._get_queries_keys_values(x, self.sample_rfs(x.device))

            num_sums, den_sums = self.init_sums(x.device)

            if self._compute_type == 'iter':
                num, _ = num_and_den.num_iter(queries, keys, values, num_sums)
                den, _ = num_and_den.den_iter(queries, keys, den_sums)
            elif self._compute_type == 'ps':
                num, _ = num_and_den.num_ps(queries, keys, values, num_sums, False)
                den, _ = num_and_den.den_ps(queries, keys, den_sums, False)
            else:
                num, _ = num_and_den.num_ps(queries, keys, values, num_sums, True)
                den, _ = num_and_den.den_ps(queries, keys, den_sums, True)

            num = torch.transpose(num, 0, 1)
            den = torch.transpose(den, 0, 1)

            outputs = num / (den[Ellipsis, None] + 1e-16)
            outputs = outputs.reshape(x.shape)

            return outputs

    def init_sums(self, device):

        head_dim = self._hidden_dim

        if self._feature_type.startswith('favor+_'):
            splitted = self._feature_type.split('_')
            feature_dim = int(splitted[1]) * head_dim
        else:
            feature_dim = head_dim

        num_sums = torch.zeros([1, self._n_heads, feature_dim, head_dim],
                            device=device)
        den_sums = torch.zeros([1, self._n_heads, feature_dim], device=device)

        return num_sums, den_sums

    def incr_step(self, x, num_sums, den_sums, on_forward, rfs, on_start):

        queries, keys, values = self._get_queries_keys_values(x, rfs)

        if not on_forward:

            if on_start:
                num_sums = torch.zeros_like(num_sums)
                den_sums = torch.zeros_like(den_sums)
            elif self._compute_type == 'iter':
                num_sums = num_and_den.num_reverse_sums_iter(queries, keys, values,
                                                                num_sums)
                den_sums = num_and_den.den_reverse_sums_iter(queries, keys, den_sums)
            else:
                num_sums = num_and_den.num_reverse_sums_ps(queries, keys, values,
                                                            num_sums)
                den_sums = num_and_den.den_reverse_sums_ps(queries, keys, den_sums)

            num_sums = num_sums.detach().clone()
            num_sums.requires_grad = True
            den_sums = den_sums.detach().clone()
            den_sums.requires_grad = True

            init_num_sums = num_sums
            init_den_sums = den_sums

        if self._compute_type == 'iter':
            num, num_sums = num_and_den.num_iter(queries, keys, values, num_sums)
            den, den_sums = num_and_den.den_iter(queries, keys, den_sums)
        elif self._compute_type == 'ps':
            num, num_sums = num_and_den.num_ps(queries, keys, values, num_sums, False)
            den, den_sums = num_and_den.den_ps(queries, keys, den_sums, False)
        else:
            num, num_sums = num_and_den.num_ps(queries, keys, values, num_sums, True)
            den, den_sums = num_and_den.den_ps(queries, keys, den_sums, True)

        num = torch.transpose(num, 0, 1)
        den = torch.transpose(den, 0, 1)

        outputs = num / (den[Ellipsis, None] + 1e-16)
        outputs = outputs.reshape(x.shape)

        if on_forward:
            return outputs, num_sums, den_sums

        return outputs, init_num_sums, init_den_sums, num_sums, den_sums

    def _get_queries_keys_values(self, inputs, rfs):

        queries = self.proj_q(inputs)
        keys = self.proj_k(inputs)
        values = self.proj_v(inputs)

        queries = queries.reshape(
            [queries.shape[0], queries.shape[1], self._n_heads, -1])
        keys = keys.reshape([keys.shape[0], keys.shape[1], self._n_heads, -1])
        values = values.reshape(
            [values.shape[0], values.shape[1], self._n_heads, -1])

        if self._feature_type == 'relu':
            queries = torch.nn.functional.relu(queries)
            keys = torch.nn.functional.relu(keys)
        elif self._feature_type == 'elu+1':
            queries = torch.nn.functional.elu(queries) + 1
            keys = torch.nn.functional.elu(keys) + 1
        elif self._feature_type == 'sqr':
            queries = queries**2
            keys = keys**2
        elif self._feature_type == 'abs':
            queries = torch.abs(queries)
            keys = torch.abs(keys)
        else:

            head_dim = self._hidden_dim

            queries = queries * np.power(head_dim, -0.25)
            queries = torch.einsum('ijkl,klm->ijkm', queries, rfs) - (queries**2).sum(
                3, keepdim=True) / 2
            queries = torch.exp(queries)

            keys = keys * np.power(head_dim, -0.25)
            keys = torch.einsum('ijkl,klm->ijkm', keys, rfs) - (keys**2).sum(
                3, keepdim=True) / 2
            keys = torch.exp(keys)

        queries = queries.transpose(0, 1)
        keys = keys.transpose(0, 1)
        values = values.transpose(0, 1)

        return queries, keys, values

    def sample_rfs(self, device):

        if not self._feature_type.startswith('favor+'):
            return None

        if self._feature_type == 'favor+':
            factor = 1
        else:
            splitted = self._feature_type.split('_')
            factor = int(splitted[1])

        head_dim = self._hidden_dim

        rfs = [[
            _sample_orth_matrix(head_dim, device)[None, Ellipsis] for _ in range(factor)
        ] for _ in range(self._n_heads)]
        rfs = [torch.cat(x, 2) for x in rfs]
        rfs = torch.cat(rfs, 0)
        rfs = rfs * np.sqrt(head_dim)

        return rfs

def _sample_orth_matrix(size, device):
    """Samples orthogonal matrix to reduce variance for random features."""
    subspace = torch.randn(size, size, device=device)
    subspace = torch.tril(subspace)
    subspace = subspace / torch.sqrt((subspace**2).sum(0, keepdim=True))

    S = torch.triu(subspace.T.mm(subspace)) - 0.5 * torch.eye(
        subspace.shape[1], device=device)

    result = torch.eye(
        subspace.shape[0], device=device) - subspace.mm(torch.inverse(S)).mm(
            subspace.T)

    return result

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
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
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


