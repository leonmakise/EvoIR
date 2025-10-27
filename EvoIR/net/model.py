import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from einops import rearrange

# from net.G2P_L2P import G2P_L2P_fuse
# from net.adaffn import AdaptIR
from net.fftformer import FFTransformerBlock
# from net.fuse import Interaction
# from net.noisymoe import Fig7MoE as moe
import torch
import torch.nn.functional as F
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure


# from net.STPG_G_MESE import STPG_G_MESE


##########################################################################
## Layer Norm

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)


import torch


def _regulaztion_loss_promptlen_angle(prompt_param):
    '''
    intro:
        theta larger then the threshold. loss = max(0, cos(\theta_min) - AB / abs(a) * abs(b))
    '''
    # prompts = prompt_param[0]  # 取 batch 维度为 0 的数据, 形状为 [num_prompts, dim]
    prompt=prompt_param
    # 计算所有向量对的点积
    similarity_matrix = torch.mm(prompt.T, prompt)  # [num_prompts, num_prompts]

    # 计算 L2 范数
    norms = torch.norm(prompt, dim=1, keepdim=True)  # [num_prompts, 1]

    # 计算余弦相似度矩阵
    cosine_sim = similarity_matrix / (norms.T @ norms)  # 广播计算范数相乘, 形状 [num_prompts, num_prompts]

    # 只取上三角部分（去掉对角线）
    cos_theta_min = 0  # 对应 90°
    mask = torch.triu(torch.ones_like(cosine_sim), diagonal=1)  # 上三角掩码
    loss = torch.relu(cosine_sim - cos_theta_min) * mask  # 只保留上三角的损失

    return loss.sum()


class Taskprompt(nn.Module):
    def __init__(self, in_dim, atom_num=32, atom_dim=256):
        super(Taskprompt, self).__init__()
        hidden_dim = 64
        self.CondNet = nn.Sequential(nn.Conv2d(in_dim, hidden_dim, 3, 3), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(hidden_dim, hidden_dim, 3, 3), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(hidden_dim, hidden_dim, 1), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(hidden_dim, hidden_dim, 1), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(hidden_dim, 32, 1))
        self.lastOut = nn.Linear(32, atom_num)
        self.act = nn.GELU()
        self.dictionary = nn.Parameter(torch.randn(atom_num, atom_dim), requires_grad=True)
    def forward(self, x):
        out = self.CondNet(x)
        out = nn.AdaptiveAvgPool2d(1)(out)#[1,dim,1,1]
        out = out.view(out.size(0), -1)
        out = self.lastOut(out)
        logits = F.softmax(out, -1)
        out = logits @ self.dictionary
        out = self.act(out)

        return out

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
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class PromptGenBlock(nn.Module):
    def __init__(self, prompt_dim=128, prompt_len=5, prompt_size=96, lin_dim=192):
        super(PromptGenBlock, self).__init__()
        self.prompt_param = nn.Parameter(torch.rand(1, prompt_len, prompt_dim, prompt_size, prompt_size))
        self.linear_layer = nn.Linear(lin_dim, prompt_len)
        self.conv3x3 = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        emb = x.mean(dim=(-2, -1))
        prompt_weights = F.softmax(self.linear_layer(emb), dim=1)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param.unsqueeze(0).repeat(B, 1, 1, 1, 1, 1).squeeze(1)
        prompt = torch.sum(prompt, dim=1)
        prompt = F.interpolate(prompt, (H, W), mode="bilinear")
        prompt = self.conv3x3(prompt)

        return prompt


##########################################################################
## Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type,dropout=0.1,dim_feedforward=2048,activation="relu"):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

        # self.adaptir = AdaptIR(dim)
        # self.activation = _get_activation_fn(activation)
        # self.linear1 = nn.Linear(dim, dim_feedforward)
        # self.dropout = nn.Dropout(dropout)
        # self.dropout3 = nn.Dropout(dropout)
        # self.linear2 = nn.Linear(dim_feedforward, dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        # tgt2 = self.norm2(x)
        # adapt = self.adaptir(x)
        # # tgt2 = self.linear2(self.dropout(self.activation(self.linear1(adapt))))
        # tgt = x + self.dropout3(tgt2 + adapt)
        # return tgt

        return x

##########################################################################
## Channel-Wise Cross Attention (CA)
class Chanel_Cross_Attention(nn.Module):
    def __init__(self, dim, num_head, bias):
        super(Chanel_Cross_Attention, self).__init__()
        self.num_head = num_head
        self.temperature = nn.Parameter(torch.ones(num_head, 1, 1), requires_grad=True)

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)


        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        # x -> q, y -> kv
        assert x.shape == y.shape, 'The shape of feature maps from image and features are not equal!'

        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_head)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_head)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_head)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = q @ k.transpose(-2, -1) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_head, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x
    

##########################################################################
## H-L Unit
class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()

        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        max = torch.max(x,1,keepdim=True)[0]
        mean = torch.mean(x,1,keepdim=True)
        scale = torch.cat((max, mean), dim=1)
        scale =self.spatial(scale)
        scale = F.sigmoid(scale)
        return scale

##########################################################################
## L-H Unit
class ChannelGate(nn.Module):
    def __init__(self, dim):
        super(ChannelGate, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.max = nn.AdaptiveMaxPool2d((1,1))

        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim//16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(dim//16, dim, 1, bias=False)
        )

    def forward(self, x):
        avg = self.mlp(self.avg(x))
        max = self.mlp(self.max(x))

        scale = avg + max
        scale = F.sigmoid(scale)
        return scale

##########################################################################
## Frequency Modulation Module (FMoM)
class FreRefine(nn.Module):
    def __init__(self, dim):
        super(FreRefine, self).__init__()

        self.SpatialGate = SpatialGate()
        self.ChannelGate = ChannelGate(dim)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, low, high):
        spatial_weight = self.SpatialGate(high)
        channel_weight = self.ChannelGate(low)
        high = high * channel_weight
        low = low * spatial_weight

        out = low + high
        out = self.proj(out)
        return out
    
##########################################################################
## Adaptive Frequency Learning Block (AFLB)
class FreModule(nn.Module):
    def __init__(self, dim, num_heads, bias, in_dim=3):
        super(FreModule, self).__init__()

        self.conv = nn.Conv2d(in_dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_dim, dim, kernel_size=3, stride=1, padding=1, bias=False)

        self.score_gen = nn.Conv2d(2, 2, 7, padding=3)

        self.para1 = nn.Parameter(torch.zeros(dim, 1, 1))
        self.para2 = nn.Parameter(torch.ones(dim, 1, 1))

        self.channel_cross_l = Chanel_Cross_Attention(dim, num_head=num_heads, bias=bias)
        self.channel_cross_h = Chanel_Cross_Attention(dim, num_head=num_heads, bias=bias)
        self.channel_cross_agg = Chanel_Cross_Attention(dim, num_head=num_heads, bias=bias)

        self.frequency_refine = FreRefine(dim)

        self.rate_conv = nn.Sequential(
            nn.Conv2d(dim, dim//8, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim//8, 2, 1, bias=False),
        )

    def forward(self, x, y):
        _, _, H, W = y.size()
        x = F.interpolate(x, (H,W), mode='bilinear')

        high_feature, low_feature = self.fft(x)



        high_feature = self.channel_cross_l(high_feature, y)
        low_feature = self.channel_cross_h(low_feature, y)

        agg = self.frequency_refine(low_feature, high_feature)
        out = self.channel_cross_agg(y, agg)

        return out * self.para1 + y * self.para2

    def shift(self, x):
        '''shift FFT feature map to center'''
        b, c, h, w = x.shape
        return torch.roll(x, shifts=(int(h/2), int(w/2)), dims=(2,3))

    def unshift(self, x):
        """converse to shift operation"""
        b, c, h ,w = x.shape
        return torch.roll(x, shifts=(-int(h/2), -int(w/2)), dims=(2,3))

    def fft(self, x, n=128):
        """obtain high/low-frequency features from input"""
        x = self.conv1(x)
        mask = torch.zeros(x.shape).to(x.device)
        h, w = x.shape[-2:]
        threshold = F.adaptive_avg_pool2d(x, 1)
        threshold = self.rate_conv(threshold).sigmoid()

        for i in range(mask.shape[0]):
            h_ = (h//n * threshold[i,0,:,:]).int()
            w_ = (w//n * threshold[i,1,:,:]).int()

            mask[i, :, h//2-h_:h//2+h_, w//2-w_:w//2+w_] = 1

        fft = torch.fft.fft2(x, norm='forward', dim=(-2,-1))
        fft = self.shift(fft)

        fft_high = fft * (1 - mask)

        high = self.unshift(fft_high)
        high = torch.fft.ifft2(high, norm='forward', dim=(-2,-1))
        high = torch.abs(high)

        fft_low = fft * mask

        low = self.unshift(fft_low)
        low = torch.fft.ifft2(low, norm='forward', dim=(-2,-1))
        low = torch.abs(low)

        return high, low


# class FreModule(nn.Module):
#     def __init__(self, dim, num_heads, bias, in_dim=3, prompt_dim=256, num_experts=4, hidden_dim=64, n=128):
#         super(FreModule, self).__init__()
#
#         self.conv = nn.Conv2d(in_dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
#         self.conv1 = nn.Conv2d(in_dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
#
#         self.score_gen = nn.Conv2d(2, 2, 7, padding=3)
#
#         self.para1 = nn.Parameter(torch.zeros(dim, 1, 1))
#         self.para2 = nn.Parameter(torch.ones(dim, 1, 1))
#
#         # MoE-based FFT high-low frequency module
#         self.task_adaptive_moe_fft = TaskAdaptiveMoEFFT(in_channels=dim, prompt_dim=256,
#                                                         num_experts=num_experts, hidden_dim=hidden_dim, n=n)
#
#         # Channel cross-attention layers
#         self.channel_cross_l = Chanel_Cross_Attention(dim, num_head=num_heads, bias=bias)
#         self.channel_cross_h = Chanel_Cross_Attention(dim, num_head=num_heads, bias=bias)
#         self.channel_cross_agg = Chanel_Cross_Attention(dim, num_head=num_heads, bias=bias)
#
#         # Frequency refine module
#         self.frequency_refine = FreRefine(dim)
#
#         self.rate_conv = nn.Sequential(
#             nn.Conv2d(dim, dim//8, 1, bias=False),
#             nn.GELU(),
#             nn.Conv2d(dim//8, 2, 1, bias=False),
#         )
#
#     def forward(self, x, y, prompt):
#         """
#         :param x: 输入特征图 (B, C, H, W)
#         :param y: 目标特征图 (B, C, H, W)
#         :param prompt: 任务提示向量 (B, prompt_dim)
#         :return: 通过高低频分割后的融合结果
#         """
#         x = self.conv(x)#[1,384,128,128]
#         _, _, H, W = y.size()
#         x = F.interpolate(x, (H,W), mode='bilinear')#[1,384,16,16]
#
#         # 使用 MoE 进行高低频特征分割
#         high_feature, low_feature = self.task_adaptive_moe_fft(x, prompt)#high_feature [1,384,16,16]
#
#         # 对高频和低频特征分别进行通道交叉注意力处理
#         high_feature = self.channel_cross_l(high_feature, y)
#         low_feature = self.channel_cross_h(low_feature, y)
#
#         # 使用频率细化模块处理低频特征
#         agg = self.frequency_refine(low_feature, high_feature)
#
#         # 通过聚合交叉注意力处理后的特征
#         out = self.channel_cross_agg(y, agg)
#
#         # 最终融合输出
#         return out * self.para1 + y * self.para2
# class TaskAdaptiveMoEFFT(nn.Module):
#     def __init__(self, in_channels, prompt_dim, num_experts=4, hidden_dim=64, n=128):
#         """
#         任务感知 MoE-Prompt Interaction 机制的 FFT 高低频分割模块
#         :param in_channels: 输入特征通道数
#         :param prompt_dim: 任务提示向量维度
#         :param num_experts: MoE 专家数
#         :param hidden_dim: MLP 中间层维度
#         :param n: 计算 mask 时的尺度参数
#         """
#         super(TaskAdaptiveMoEFFT, self).__init__()
#         self.n = n
#         self.num_experts = num_experts
#         self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
#
#         # MoE 专家网络 (多个 rate_conv)
#         self.rate_convs = nn.ModuleList([
#             nn.Conv2d(in_channels, 2, kernel_size=1, bias=False) for _ in range(num_experts)
#         ])
#
#         # MoE 门控网络 (控制专家权重)
#         self.gating_network = nn.Sequential(
#             nn.Linear(prompt_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, num_experts),
#             nn.Softmax(dim=-1)  # 任务感知的专家选择
#         )
#
#     def shift(self, x):
#         """将 FFT 结果转换到中心对齐"""
#         return torch.fft.fftshift(x, dim=(-2, -1))
#
#     def unshift(self, x):
#         """将 FFT 结果转换回原始形式"""
#         return torch.fft.ifftshift(x, dim=(-2, -1))
#
#     def forward(self, x, prompt):
#         """
#         :param x: 输入特征图 (B, C, H, W)input_image
#         :prompt[1,256]
#         :param prompt: 任务提示向量 (B, prompt_dim)
#         :return: 高频特征、低频特征
#         """
#         x = self.conv1(x)
#         B, C, H, W = x.shape#[1,384,16,16]
#
#         # 任务感知 MoE 门控网络，决定专家权重
#         expert_weights = self.gating_network(prompt)  # (B, num_experts) [1,4]
#
#         # # 计算不同专家的输出并进行加权融合
#         # threshold = sum(w[:, None, None, None] * conv(F.adaptive_avg_pool2d(x, 1))
#         #                 for w, conv in zip(expert_weights.T, self.rate_convs))
#         # 初始化 threshold 为 0
#         threshold = torch.zeros((B, 2, 1, 1), device=x.device)
#
#         # 遍历每个专家
#         for expert_idx in range(self.num_experts):
#             # 获取当前专家的权重 (B, ) -> (B, 1, 1, 1) 用于广播
#             expert_weight = expert_weights[:, expert_idx].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1, 1)
#
#             # 获取当前专家的 conv 层
#             rate_conv = self.rate_convs[expert_idx]  # (Conv2d layer)
#
#             # 计算该专家的 threshold
#             expert_output = rate_conv(F.adaptive_avg_pool2d(x, 1))  # (B, 2, 1, 1)
#
#             # 按专家权重加权
#             threshold += expert_weight * expert_output  # 加权和
#
#         # 通过 sigmoid 激活，控制 mask 大小范围
#         threshold = threshold.sigmoid()  # (B, 2, 1, 1)
#
#         threshold = threshold.sigmoid()  # (B, 2, 1, 1)
#
#         mask = torch.zeros_like(x)
#         for i in range(B):
#             h_ = (H // self.n * threshold[i, 0, :, :]).int()
#             w_ = (W // self.n * threshold[i, 1, :, :]).int()
#             mask[i, :, H//2-h_:H//2+h_, W//2-w_:W//2+w_] = 1
#
#         # 进行 FFT 变换
#         fft = torch.fft.fft2(x, norm='forward', dim=(-2, -1))
#         fft = self.shift(fft)
#
#         # 高频与低频分割
#         fft_high = fft * (1 - mask)
#         fft_low = fft * mask
#
#         # 逆 FFT
#         high = torch.abs(torch.fft.ifft2(self.unshift(fft_high), norm='forward', dim=(-2, -1)))
#         low = torch.abs(torch.fft.ifft2(self.unshift(fft_low), norm='forward', dim=(-2, -1)))
#
#         return high, low


##########################################################################
##---------- AdaIR -----------------------

class AdaIR(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8],
        num_fft_blocks=[4, 2, 2, 1],
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,#2.66
        bias = False,
        LayerNorm_type = 'WithBias', 
        decoder = True,
    ):
        atom_dim = 256
        atom_num = 32#bs

        super(AdaIR, self).__init__()
        self.decoder_flag=True
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)        
        self.decoder = decoder
        
        # if self.decoder:
        #     self.fre1 = FreModule(dim*2**3, num_heads=heads[2], bias=bias)
        #     self.fre2 = FreModule(dim*2**2, num_heads=heads[2], bias=bias)
        #     self.fre3 = FreModule(dim*2**1, num_heads=heads[2], bias=bias)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.encoder_level1_fft = nn.Sequential(*[FFTransformerBlock(dim=dim) for i in range(num_fft_blocks[0])])

        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2

        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.encoder_level2_fft = nn.Sequential(*[FFTransformerBlock(dim=int(dim*2**1)) for i in range(num_fft_blocks[1])])

        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3

        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.encoder_level3_fft = nn.Sequential(*[FFTransformerBlock(dim=int(dim*2**2)) for i in range(num_fft_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4


        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        self.latent_fft = nn.Sequential(*[FFTransformerBlock(dim=int(dim*2**3)) for i in range(num_fft_blocks[1])])

        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)

        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.decoder_level3_fft = nn.Sequential(*[FFTransformerBlock(dim=int(dim * 2 ** 2),decoder_flag=self.decoder_flag) for i in range(num_fft_blocks[2])])

        self.up3_2 = Upsample(int(dim*2**2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.decoder_level2_fft = nn.Sequential(*[FFTransformerBlock(dim=int(dim * 2 ** 1),decoder_flag=self.decoder_flag) for i in range(num_fft_blocks[1])])

        self.up2_1 = Upsample(int(dim*2**1))
        self.reduce_chan_level1 = nn.Conv2d(int(dim * 2 ), int(dim), kernel_size=1, bias=bias)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.decoder_level1_fft = nn.Sequential(*[FFTransformerBlock(dim=dim,decoder_flag=self.decoder_flag) for i in range(num_fft_blocks[0])])

        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
                    
        self.output = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        self.decoder=True

        # self.task_prompt1 = Taskprompt(in_dim=3, atom_num=32, atom_dim=48)
        # self.task_prompt2 = Taskprompt(in_dim=3, atom_num=32, atom_dim=96)
        # self.task_prompt3 = Taskprompt(in_dim=3, atom_num=32, atom_dim=192)

        # self.stpg_g_mese1 = STPG_G_MESE(atom_dim=dim, dim=dim, ffn_expansion_factor=ffn_expansion_factor)
        # self.stpg_g_mese2 = STPG_G_MESE(atom_dim=dim*2, dim=int(dim * 2 ** 1), ffn_expansion_factor=ffn_expansion_factor)
        # self.stpg_g_mese3 = STPG_G_MESE(atom_dim=dim*2**2, dim=int(dim * 2 ** 2), ffn_expansion_factor=ffn_expansion_factor)

        # self.muloss=MS_SSIMLoss

        #
        # self.fuse3=Interaction(dim * 2**2)
        # self.fuse2=Interaction(dim * 2**1)
        # self.fuse1=Interaction(dim)
        #
        # self.g2p_48=G2P_L2P_fuse(dim)
        # self.g2p_96 = G2P_L2P_fuse(dim*2)
        # self.g2p_192 = G2P_L2P_fuse(dim*2**2)

    def forward(self, inp_img,noise_emb = None):

        inp_enc_level1 = self.patch_embed(inp_img)

        out_enc_level1_fft=self.encoder_level1_fft(inp_enc_level1)
        out_enc_level1 = self.encoder_level1(out_enc_level1_fft)
        out_enc_level1 = out_enc_level1+inp_enc_level1

        inp_enc_level2 = self.down1_2(out_enc_level1)

        out_enc_level2_fft=self.encoder_level2_fft(inp_enc_level2)
        out_enc_level2 = self.encoder_level2(out_enc_level2_fft)
        out_enc_level2 = out_enc_level2+inp_enc_level2

        inp_enc_level3 = self.down2_3(out_enc_level2)

        out_enc_level3_fft=self.encoder_level3_fft(inp_enc_level3)
        out_enc_level3 = self.encoder_level3(out_enc_level3_fft)
        out_enc_level3 = out_enc_level3+inp_enc_level3

        inp_enc_level4 = self.down3_4(out_enc_level3)

        out_enc_level4_fft=self.latent_fft(inp_enc_level4)
        latent = self.latent(out_enc_level4_fft)
        latent = latent+inp_enc_level4

        if self.decoder:
            inp_dec_level3 = self.up4_3(latent)
            inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
            inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
            inp_dec_level3_fft=self.decoder_level3_fft(inp_dec_level3)
            out_dec_level3 = self.decoder_level3(inp_dec_level3_fft)
            out_dec_level3 = out_dec_level3+inp_dec_level3

        if self.decoder:
            inp_dec_level2 = self.up3_2(out_dec_level3)
            inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
            inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
            inp_dec_level2_fft=self.decoder_level2_fft(inp_dec_level2)
            out_dec_level2 = self.decoder_level2(inp_dec_level2_fft)
            out_dec_level2 = out_dec_level2+inp_dec_level2

        if self.decoder:
            inp_dec_level1 = self.up2_1(out_dec_level2)
            inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
            inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
            inp_dec_level1_fft=self.decoder_level1_fft(inp_dec_level1)
            out_dec_level1 = self.decoder_level1(inp_dec_level1_fft)
            out_dec_level1 = out_dec_level1+inp_dec_level1

        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1) + inp_img

        if self.training:
            return out_dec_level1
        else:
            return out_dec_level1

class MS_SSIMLoss(torch.nn.Module):
    def __init__(self, data_range=1.0):
        super(MS_SSIMLoss, self).__init__()
        self.ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=data_range)

    def forward(self, pred, target):
        """
        计算 MS-SSIM 损失
        :param pred: 预测的特征图，形状 [B, C, H, W]
        :param target: GT 特征图，形状 [B, C, H, W]
        :return: MS-SSIM loss (1 - MS-SSIM)
        """
        ms_ssim_value = self.ms_ssim(pred, target)
        return 1 - ms_ssim_value  # 损失 = 1 - MS-SSIM，值越小表示相似度越高


# 示例代码
if __name__ == "__main__":
    B, C, H, W = 4, 3, 64, 64  # 批量大小、通道数、高度、宽度
    pred = torch.rand(B, C, H, W)  # 预测的特征图
    target = torch.rand(B, C, H, W)  # GT 特征图

    ms_ssim_loss = MS_SSIMLoss()
    loss = ms_ssim_loss(pred, target)
    print("MS-SSIM Loss:", loss.item())

    
