import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange

from net.fftprompt import pre_work
# from net.fftprompt2 import pre_work2
from net.prompt import Prompt2Weight


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
        return x / torch.sqrt(sigma + 1e-5) * self.weight


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
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class FSAS(nn.Module):
    def __init__(self, dim, bias):
        super(FSAS, self).__init__()

        self.to_hidden = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(dim * 6, dim * 6, kernel_size=3, stride=1, padding=1, groups=dim * 6, bias=bias)

        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

        self.norm = LayerNorm(dim * 2, LayerNorm_type='WithBias')

        self.patch_size = 2

    def forward(self, x):
        hidden = self.to_hidden(x)

        q, k, v = self.to_hidden_dw(hidden).chunk(3, dim=1)

        q_patch = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        k_patch = rearrange(k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        q_fft = torch.fft.rfft2(q_patch.float())
        k_fft = torch.fft.rfft2(k_patch.float())

        out = q_fft * k_fft
        out = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size))
        out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                        patch2=self.patch_size)

        out = self.norm(out)

        output = v * out
        output = self.project_out(output)

        return output

class DFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):

        super(DFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.patch_size = 2

        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features , hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features , bias=bias)

        self.fft = nn.Parameter(torch.ones((hidden_features , 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.fc = nn.Linear(in_features=dim//2, out_features=int(dim * ffn_expansion_factor))
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        # self.fft2=Prompt2Weight(prompt_len=hidden_features,patch_size_h= self.patch_size,patch_size_w=self.patch_size // 2 + 1)


    def forward(self, x):
        x = self.project_in(x)

        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        x_patch_fft = torch.fft.rfft2(x_patch.float())

        # W_c=self.fft2(hprompt_expand,lprompt_flat)

        x_patch_fft = x_patch_fft * self.fft
        # x_patch_fft = x_patch_fft * W_c
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                      patch2=self.patch_size)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)

        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class FFTransformerBlock(nn.Module):
    def __init__(self, dim,decoder_flag=True, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias', att=False):#decoder_flag:false
        super(FFTransformerBlock, self).__init__()

        self.att = att
        if self.att:
            self.norm1 = LayerNorm(dim, LayerNorm_type)
            self.attn = FSAS(dim, bias)

        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = DFFN(dim, ffn_expansion_factor, bias)

        self.prompt_conv = nn.Conv2d(dim, dim, kernel_size=1)
        # 可学习的投影矩阵 Wp：shape = [c, c]
        self.Wp = nn.Parameter(torch.randn(dim, dim))
        self.prompt_block=pre_work(decoder_flag=decoder_flag,inchannels=dim)
        # self.prompt_block = pre_work2(inchannels=dim)

    def forward(self, x):
        prompt1=self.prompt_block(x)
        prompt2=self.prompt_conv(prompt1)
        prompt=torch.sigmoid(prompt2)
        x=prompt+prompt1

        # old
        # prompt=self.prompt_block(x)
        # prompt=self.prompt_conv(prompt)
        # prompt=torch.sigmoid(prompt)
        # x=prompt+x

        if self.att:
            x = x + self.attn(self.norm1(x))

        x = x + self.ffn(self.norm2(x))

        return x

# class Fuse(nn.Module):
#     def __init__(self, n_feat):
#         super(Fuse, self).__init__()
#         self.n_feat = n_feat
#         self.att_channel = TransformerBlock(dim=n_feat)
#
#
#     def forward(self, x):
#         output = self.att_channel(x)
#
#         return output
