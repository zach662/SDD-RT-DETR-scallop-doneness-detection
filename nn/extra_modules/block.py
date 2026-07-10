import torch
import torch.nn as nn
from ..modules.conv import Conv, DWConv, DSConv, RepConv, GhostConv, autopad, LightConv, ConvTranspose
from ..modules.block import get_activation, ConvNormLayer, BasicBlock, BottleNeck, RepC3, C3, C2f, Bottleneck
from .attention import *

__all__ = [
           'Converse2DC3', 'Converse2D', 'GCConv', 'HierarchicalRepNet', 'EfficientBalanceFusionModuleV1'
           ]
class HierarchicalRepNet(nn.Module):

    def __init__(self, c1, c2, n=1, scale=0.5, e=0.5, patch_size=4):
        super(HierarchicalRepNet, self).__init__()

        self.c = int(c2 * e)  # 隐藏层通道数
        self.mid = int(self.c * scale)  # 中间层通道数
        self.patch_size = patch_size  # 注意力模块的patch大小

        # 初始特征分割
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)

        # 特征融合（最终输出层）
        self.cv2 = Conv(self.c + self.mid * (n + 1), c2, 1)

        # 主干处理路径
        self.cv3 = GCConv(self.c, self.mid, 3)  # 替换RepConv为GCConv
        self.lg1 = LocalGlobalAttention(self.mid, patch_size=self.patch_size)  # 添加CBAM注意力

        # 中间处理层（多个卷积）
        self.m = nn.ModuleList()
        for _ in range(n - 1):
            conv = Conv(self.mid, self.mid, 3)
            lg = LocalGlobalAttention(self.mid, patch_size=self.patch_size)  # 每个卷积后添加CBAM
            self.m.append(nn.ModuleList([conv, lg]))

        # 最终处理层
        self.cv4 = Conv(self.mid, self.mid, 1)
        self.lg2 = LocalGlobalAttention(self.mid, patch_size=self.patch_size)  # 最终CBAM注意力

    def forward(self, x):
        """前向传播"""
        # 初始特征分割
        y = list(self.cv1(x).chunk(2, 1))

        # 处理第二个分支（主处理路径）
        y[-1] = self.cv3(y[-1])  # GCConv
        y[-1] = self.lg1(y[-1])  # 应用CBAM

        # 中间处理层
        for conv, lg in self.m:
            feat = conv(y[-1])
            feat = lg(feat)  # 应用CBAM
            y.append(feat)

        # 最终处理
        final_feat = self.cv4(y[-1])
        final_feat = self.lg2(final_feat)  # 应用CBAM
        y.append(final_feat)

        # 特征融合
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """使用split的前向传播（内存效率更高）"""
        y = list(self.cv1(x).split((self.c, self.c), 1))

        # 处理第二个分支
        y[-1] = self.cv3(y[-1])
        y[-1] = self.lg1(y[-1])

        # 中间处理层
        for conv, lg in self.m:
            feat = conv(y[-1])
            feat = lg(feat)
            y.append(feat)

        # 最终处理
        final_feat = self.cv4(y[-1])
        final_feat = self.lg2(final_feat)
        y.append(final_feat)

        return self.cv2(torch.cat(y, 1))


class EfficientBalanceFusionModuleV1(nn.Module):
    def __init__(self, inc) -> None:
        super().__init__()

        self.adjust_conv = nn.Identity()
        if inc[0] != inc[1]:
            self.adjust_conv = Conv(inc[0], inc[1], k=1)

        self.afgca = AFGCAttention(inc[1] * 2)

        # 可学习的融合参数
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        x0, x1 = x
        x0 = self.adjust_conv(x0)

        x_concat = torch.cat([x0, x1], dim=1)
        x_concat = self.afgca(x_concat)

        # 使用softmax确保权重和为1
        x0_weight, x1_weight = torch.split(x_concat, [x0.size()[1], x1.size()[1]], dim=1)
        x0_weight = torch.sigmoid(x0_weight)
        x1_weight = torch.sigmoid(x1_weight)

        # 双向门控融合
        alpha = torch.sigmoid(self.alpha)
        beta = torch.sigmoid(self.beta)

        # 改进的融合公式
        fused_x0 = alpha * x0 * (1 + x1_weight) + (1 - alpha) * x1 * x0_weight
        fused_x1 = beta * x1 * (1 + x0_weight) + (1 - beta) * x0 * x1_weight

        return torch.cat([fused_x0, fused_x1], dim=1)
