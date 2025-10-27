import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class PerceptualLoss(nn.Module):
    def __init__(self, feature_layers=[3,5,15]):
        super(PerceptualLoss, self).__init__()
        # 使用预训练的 VGG16 网络
        vgg = models.vgg16(pretrained=True).features
        self.vgg = vgg.eval()  # 使用评估模式

        # 选择要计算损失的中间层（通常选择 VGG 网络的卷积层）
        self.feature_layers = feature_layers

        # 冻结 VGG 网络的参数
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        """
        计算感知损失
        :param pred: 预测图像，形状 [B, C, H, W]
        :param target: 真实图像，形状 [B, C, H, W]
        :return: 感知损失
        """
        # 将输入图像通过 VGG 网络计算中间特征
        pred_features = self.extract_features(pred)
        target_features = self.extract_features(target)

        # 计算不同层的感知损失
        loss = 0
        for pred_feat, target_feat in zip(pred_features, target_features):
            loss += F.mse_loss(pred_feat, target_feat)  # 使用均方误差计算特征图差异

        return loss

    def extract_features(self, x):
        """
        从 VGG 网络提取指定层的特征
        :param x: 输入图像，形状 [B, C, H, W]
        :return: 中间层的特征
        """
        features = []
        for idx, layer in enumerate(self.vgg):
            x = layer(x)
            if idx in self.feature_layers:
                features.append(x)
        return features

# 示例使用
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# pred = torch.randn(8, 3, 256, 256).to(device)  # 假设预测的图像
# target = torch.randn(8, 3, 256, 256).to(device)  # 真实的图像
#
# # 初始化感知损失
# perceptual_loss = PerceptualLoss(feature_layers=[2, 7, 12, 21, 30]).to(device)
#
# # 计算损失
# loss_d = perceptual_loss(pred, target)


# print(f"Perceptual Loss: {loss.item()}")
