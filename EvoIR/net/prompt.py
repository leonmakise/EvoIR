
import torch.nn as nn
import torch
import torch.nn.functional as F

class AdaptivePromptFusion(nn.Module):
    def __init__(self, channels, prompt_len=4):
        super().__init__()
        self.channels = channels
        self.prompt_len = prompt_len

        # learnable prompt，不依赖任何先验
        # self.hprompt = nn.Parameter(torch.randn(prompt_len, channels))
        # self.lprompt = nn.Parameter(torch.randn(prompt_len, channels))

        # 融合控制器，用统计特征+prompt生成融合权重
        # self.hctrl = nn.Sequential(
        #     nn.Linear(channels + prompt_len * channels, channels),
        #     nn.ReLU(),
        #     nn.Linear(channels, channels),
        #     nn.Sigmoid()
        # )
        # self.lctrl = nn.Sequential(
        #     nn.Linear(channels + prompt_len * channels, channels),
        #     nn.ReLU(),
        #     nn.Linear(channels, channels),
        #     nn.Sigmoid()
        # )

        self.fusion_proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, freh, frel):
        """
        freh, frel: [B, C, H, W] - 高频/低频引导特征图
        """
        B, C, H, W = freh.shape

        # 图像感知：提取 freh / frel 的全局统计（平均池化）
        f_h_stat = F.adaptive_avg_pool2d(freh, 1).view(B, C)  # [B, C]
        f_l_stat = F.adaptive_avg_pool2d(frel, 1).view(B, C)  # [B, C]

        # # 广播 prompt 给每个样本
        # hprompt_expand=self.hprompt.unsqueeze(0).expand(B, -1, -1)# [B, prompt_len, C]
        # lprompt_expand = self.lprompt.unsqueeze(0).expand(B, -1, -1)  # [B, prompt_len, C]
        #
        # hprompt_flat=hprompt_expand.flatten(1) # [B, prompt_len * C]
        # lprompt_flat=lprompt_expand.flatten(1) # [B, prompt_len * C]
        #
        # # 拼接统计特征 + prompt → 融合控制器
        # hctrl_input = torch.cat([f_h_stat, hprompt_flat], dim=-1)
        # lctrl_input = torch.cat([f_l_stat, lprompt_flat], dim=-1)
        #
        # # [B, C]
        # w_h = self.hctrl(hctrl_input)
        # w_l = self.lctrl(lctrl_input)

        # 拼接成 [B, 2, C]
        w_pair = torch.stack([f_h_stat, f_l_stat], dim=1)

        # 对每个通道做 softmax，确保 w_h + w_l = 1
        w_pair = torch.softmax(w_pair, dim=1)  # [B, 2, C]

        # 拆分回去
        w_h = w_pair[:, 0].view(B, C, 1, 1)  # [B, C, 1, 1]
        w_l = w_pair[:, 1].view(B, C, 1, 1)  # [B, C, 1, 1]

        # 自适应融合高低频引导特征
        fused = freh * w_h + frel * w_l
        out = self.fusion_proj(fused)
        return out

        # return out,hprompt_expand,lprompt_flat

class Prompt2Weight(nn.Module):
    def __init__(self, prompt_len=8,patch_size_h=2,patch_size_w=2):
        super().__init__()
        self.patch_size_h=patch_size_h
        self.patch_size_w=patch_size_w
        self.mlp = nn.Sequential(
            nn.Linear(prompt_len * prompt_len, 256),
            nn.ReLU(),
            nn.Linear(256, self.patch_size_h * self.patch_size_w)
        )

    def forward(self, prompt_h, prompt_l):
        """
        prompt_h, prompt_l: [B, L, D]
        Return:
            W_c: [B, 1, 1, H, W]
        """
        B, L, D = prompt_h.shape

        # 相似度矩阵
        sim = torch.matmul(prompt_h, prompt_l.transpose(-1, -2)) / (D ** 0.5)  # [B, L, L]

        # 展开 & MLP 生成 frequency 权重
        sim_flat = sim.view(B, -1)  # [B, L*L]
        freq_weight = self.mlp(sim_flat)  # [B, H*W]
        freq_weight = freq_weight.view(B, 1, 1, self.patch_size_h, self.patch_size_w)

        return freq_weight  # 用作 W_c


if __name__ == '__main__':
    high=torch.randn(1, 48,128, 128)
    low=torch.randn(1, 48,128, 128)
    model=AdaptivePromptFusion(channels=48, prompt_len=48)
    out=model(high,low)
    print(out)
