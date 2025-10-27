import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from net.prompt import AdaptivePromptFusion


# class FrequencyGuidedAttention(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(FrequencyGuidedAttention, self).__init__()
#         # 深度可分离卷积用于计算 Q, K, V
#         self.depthwise_conv_q = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1,
#                                           groups=in_channels)
#         self.depthwise_conv_k = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1,
#                                           groups=in_channels)
#         self.depthwise_conv_v = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1,
#                                           groups=in_channels)
#
#         self.fc_out = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # 输出通道映射
#
#     def forward(self, x, freq_feat):
#         """
#         Args:
#             x (torch.Tensor): 当前层的特征图，shape [B, C, H, W]
#             freq_feat (torch.Tensor): 高频或低频特征图，shape [B, C, H, W]
#         """
#         # Query (Q), Key (K), Value (V) 通过深度可分离卷积进行计算
#         Q = self.depthwise_conv_q(freq_feat)  # Q: shape [B, C, H, W]
#         K = self.depthwise_conv_k(x)  # K: shape [B, C, H, W]
#         V = self.depthwise_conv_v(x)  # V: shape [B, C, H, W]
#
#
#         # 计算 attention
#         attention_map = torch.einsum('bchw,bchv->bchwv', Q, K)  # [B, C, H, W, H, W] 点积得到 attention map
#         attention_map = torch.softmax(attention_map, dim=-1)  # Softmax 操作
#
#         # 得到新的 V（加权和）
#         output = torch.einsum('bchwv,bchv->bchw', attention_map, V)
#
#         # 输出映射到目标通道数
#         output = self.fc_out(output)  # 最终输出 [B, out_channels, H, W]
#
#         return output

class FrequencyGuidedAttention(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False):
        super(FrequencyGuidedAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 分别计算 Q（来自 freq_feat）和 K,V（来自 x）
        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.kv_proj = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)

        # 可选深度可分离卷积增强
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1, groups=dim * 2, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, freq_feat):
        """
        x: 主干特征，作为 Key 和 Value [B, C, H, W]
        freq_feat: 频域特征，引导 Q [B, C, H, W]
        """
        B, C, H, W = x.shape

        # 得到 Q from freq_feat
        q = self.q_dwconv(self.q_proj(freq_feat))
        # 得到 K, V from x
        kv = self.kv_dwconv(self.kv_proj(x))
        k, v = kv.chunk(2, dim=1)

        # 多头 reshape
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # 归一化
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # 注意力
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature  # [B, head, HW, HW]
        attn = attn.softmax(dim=-1)

        out = torch.matmul(attn, v)  # [B, head, C, HW]
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=H, w=W)

        out = self.project_out(out)
        return out

class FrequencyGuidedAttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels,decoder_flag=False):
        super(FrequencyGuidedAttentionModule, self).__init__()
        self.decoder_flag=decoder_flag
        # 高频引导的注意力模块
        self.high_freq_attention = FrequencyGuidedAttention(in_channels, out_channels)
        # 低频引导的注意力模块
        self.low_freq_attention = FrequencyGuidedAttention(in_channels, out_channels)
        if self.decoder_flag:
            self.fusion = AdaptivePromptFusion(channels=in_channels, prompt_len=in_channels)
        else:
            self.final_proj = nn.Conv2d(in_channels*2, out_channels, kernel_size=1)

    def forward(self, x, high_freq, low_freq):
        """
        Args:
            x (torch.Tensor): 上一层输出的特征图，shape [B, C, H, W]
            high_freq (torch.Tensor): 高频特征图，shape [B, C, H, W]
            low_freq (torch.Tensor): 低频特征图，shape [B, C, H, W]
        """
        # 高频引导注意力
        high_freq_output = self.high_freq_attention(x, high_freq)
        # high_freq_output = self.high_freq_attention(high_freq, x)
        # 低频引导注意力
        low_freq_output = self.low_freq_attention(x, low_freq)
        # low_freq_output = self.low_freq_attention(low_freq, x)

        if self.decoder_flag:
        # 将两个输出进行融合
            output=self.fusion(high_freq_output, low_freq_output)#1

        else:
            output = torch.cat([high_freq_output, low_freq_output], dim=1)
            output = self.final_proj(output)
        #2
        # output=torch.cat([high_freq_output, low_freq_output], dim=1)
        #output = self.final_proj(output)

        #3
        # output = high_freq_output + low_freq_output  # 可选择其他方式结合，如加权融合

        return output



class FD(nn.Module):
    def __init__(self, inchannels, kernel_size=3, stride=1, group=8):
        super(FD, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group
        self.lamb_l = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
        self.lamb_h = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
        self.conv = nn.Conv2d(inchannels, group * kernel_size ** 2, kernel_size=1, stride=1, bias=False)#group * kernel_size ** 2
        self.bn = nn.BatchNorm2d(group * kernel_size ** 2)##group * kernel_size ** 2
        self.act = nn.Softmax(dim=-2)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.ap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        identity_input = x
        low_filter = self.ap(x)
        low_filter = self.conv(low_filter)
        low_filter = self.bn(low_filter)
        n, c, h, w = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape(n, self.group, c // self.group,self.kernel_size ** 2, h * w)
        n, c1, p, q = low_filter.shape
        low_filter = low_filter.reshape(n, c1 // self.kernel_size ** 2, self.kernel_size ** 2, p * q).unsqueeze(2)
        low_filter = self.act(low_filter)
        low = torch.sum(x * low_filter, dim=3).reshape(n, c, h, w)
        high = identity_input - low

        return low,high

class pre_att(nn.Module):
    def __init__(self, in_channels,flag_highF):
        super(pre_att, self).__init__()
        self.flag_highF = flag_highF
        k_size = 3
        dim=in_channels
        if flag_highF:
            self.body = nn.Sequential(nn.Conv2d(dim, dim, (1 ,k_size), padding=(0, k_size //2), groups=dim),
                                      nn.Conv2d(dim, dim, (k_size ,1), padding=(k_size //2, 0), groups=dim),
                                      nn.GELU())
        else:
            self.body = nn.Sequential(nn.Conv2d( 2 *dim , 2 *dim ,kernel_size=1 ,stride=1),
                                      nn.GELU(),
                                      )

    def forward(self, ffm):
        if self.flag_highF:
            y_att = self.body(ffm)*ffm
        else:
            bs,c,H,W=ffm.shape
            y = torch.fft.rfft2(ffm.to(torch.float32).cuda())
            y_imag = y.imag
            y_real = y.real
            y_f = torch.cat([y_real, y_imag], dim=1)
            y_att = self.body(y_f)
            y_f = y_f * y_att
            y_real, y_imag = torch.chunk(y_f, 2, dim=1)
            y = torch.complex(y_real, y_imag)
            y_att = torch.fft.irfft2(y, s=(H, W))

        return y_att


class pre_work(nn.Module):
    def __init__(self, decoder_flag=False,inchannels=48):
        super(pre_work, self).__init__()
        self.fd=FD(inchannels)
        self.decoder_flag=decoder_flag

        self.freguide=FrequencyGuidedAttentionModule(in_channels=inchannels, out_channels=inchannels,decoder_flag=self.decoder_flag)

        self.FSPG_high = pre_att(in_channels=inchannels, flag_highF=True)
        self.FSPG_low = pre_att( in_channels=inchannels,flag_highF=False)


    def forward(self, x):
        low_fre,high_fre = self.fd(x)
        high_fre_aft=self.FSPG_high(high_fre)
        low_fre_aft=self.FSPG_low(low_fre)

        # prompt=self.freguide(x,high_fre,low_fre)
        prompt = self.freguide(x, high_fre_aft, low_fre_aft)
        return prompt+x


