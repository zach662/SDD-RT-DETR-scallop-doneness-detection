import torch
import torch.nn as nn
from ..modules.conv import Conv, DWConv, DSConv, RepConv, GhostConv, autopad, LightConv, ConvTranspose
from ..modules.block import get_activation, ConvNormLayer, BasicBlock, BottleNeck, RepC3, C3, C2f, Bottleneck
from .attention import *

__all__ = [
           'Converse2DC3', 'Converse2D', 'GCConv', 'HierarchicalRepNet', 'EfficientBalanceFusionModuleV1', 'AIFI_EDFFN'
           ]
class HierarchicalRepNet(nn.Module):

    def __init__(self, c1, c2, n=1, scale=0.5, e=0.5, patch_size=4):
        super(HierarchicalRepNet, self).__init__()

        self.c = int(c2 * e)
        self.mid = int(self.c * scale)
        self.patch_size = patch_size

        self.cv1 = Conv(c1, 2 * self.c, 1, 1)


        self.cv2 = Conv(self.c + self.mid * (n + 1), c2, 1)


        self.cv3 = GCConv(self.c, self.mid, 3)
        self.lg1 = LocalGlobalAttention(self.mid, patch_size=self.patch_size)

        self.m = nn.ModuleList()
        for _ in range(n - 1):
            conv = Conv(self.mid, self.mid, 3)
            lg = LocalGlobalAttention(self.mid, patch_size=self.patch_size)
            self.m.append(nn.ModuleList([conv, lg]))

        self.cv4 = Conv(self.mid, self.mid, 1)
        self.lg2 = LocalGlobalAttention(self.mid, patch_size=self.patch_size)

    def forward(self, x):

        y = list(self.cv1(x).chunk(2, 1))

        y[-1] = self.cv3(y[-1])
        y[-1] = self.lg1(y[-1])

        for conv, lg in self.m:
            feat = conv(y[-1])
            feat = lg(feat)
            y.append(feat)

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

        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        x0, x1 = x
        x0 = self.adjust_conv(x0)

        x_concat = torch.cat([x0, x1], dim=1)
        x_concat = self.afgca(x_concat)

        x0_weight, x1_weight = torch.split(x_concat, [x0.size()[1], x1.size()[1]], dim=1)
        x0_weight = torch.sigmoid(x0_weight)
        x1_weight = torch.sigmoid(x1_weight)

        alpha = torch.sigmoid(self.alpha)
        beta = torch.sigmoid(self.beta)

        fused_x0 = alpha * x0 * (1 + x1_weight) + (1 - alpha) * x1 * x0_weight
        fused_x1 = beta * x1 * (1 + x0_weight) + (1 - beta) * x0 * x1_weight

        return torch.cat([fused_x0, fused_x1], dim=1)



class RepC3(nn.Module):

    def __init__(self, c1, c2, n=3, e=1.0):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class Converse2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale=1, padding_mode='circular', eps=1e-5, act=False):
        super(Converse2D, self).__init__()
        """
        Converse2D Operator for Image Restoration Tasks.

        Args:
            x (Tensor): Input tensor of shape (N, in_channels, H, W), where
                        N is the batch size, H and W are spatial dimensions.
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels produced by the operation.
            kernel_size (int): Size of the kernel.
            scale (int): Upsampling factor. For example, `scale=2` doubles the resolution.
            padding_mode (str, optional): Padding method. One of {'reflect', 'replicate', 'circular', 'constant'}.
                                        Default is `circular`.
            eps (float, optional): Small value added to denominators for numerical stability.
                                Default is a small value like 1e-5.

        Returns:
            Tensor: Output tensor of shape (N, out_channels, H * scale, W * scale), where spatial dimensions
                    are upsampled by the given scale factor.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.scale = scale
        self.padding = kernel_size - 1
        self.padding_mode = padding_mode
        self.eps = eps

        assert self.out_channels == self.in_channels
        self.weight = nn.Parameter(torch.randn(1, self.in_channels, self.kernel_size, self.kernel_size))
        self.bias = nn.Parameter(torch.zeros(1, self.in_channels, 1, 1))
        self.weight.data = nn.functional.softmax(self.weight.data.view(1, self.in_channels, -1), dim=-1).view(1,
                                                                                                              self.in_channels,
                                                                                                              self.kernel_size,
                                                                                                              self.kernel_size)

        self.act = nn.Identity()
        if act:
            self.act = act()

    def forward(self, x):

        if self.padding > 0:
            x = nn.functional.pad(x, pad=[self.padding, self.padding, self.padding, self.padding],
                                  mode=self.padding_mode, value=0)

        biaseps = torch.sigmoid(self.bias - 9.0) + self.eps
        _, _, h, w = x.shape
        STy = self.upsample(x, scale=self.scale)
        if self.scale != 1:
            x = nn.functional.interpolate(x, scale_factor=self.scale, mode='nearest')
            # x = nn.functional.interpolate(x, scale_factor=self.scale, mode='bilinear',align_corners=False)
        # x = torch.zeros_like(x)

        FB = self.p2o(self.weight, (h * self.scale, w * self.scale))
        FBC = torch.conj(FB)
        F2B = torch.pow(torch.abs(FB), 2)
        FBFy = FBC * torch.fft.fftn(STy, dim=(-2, -1))

        FR = FBFy + torch.fft.fftn(biaseps * x, dim=(-2, -1))
        x1 = FB.mul(FR)
        FBR = torch.mean(self.splits(x1, self.scale), dim=-1, keepdim=False)
        invW = torch.mean(self.splits(F2B, self.scale), dim=-1, keepdim=False)
        invWBR = FBR.div(invW + biaseps)
        FCBinvWBR = FBC * invWBR.repeat(1, 1, self.scale, self.scale)
        FX = (FR - FCBinvWBR) / biaseps
        out = torch.real(torch.fft.ifftn(FX, dim=(-2, -1)))

        if self.padding > 0:
            out = out[..., self.padding * self.scale:-self.padding * self.scale,
                  self.padding * self.scale:-self.padding * self.scale]

        return self.act(out)



class Converse2DC3(RepC3):
    def __init__(self, c1, c2, n=3, e=1):
        super().__init__(c1, c2, n, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*[Converse2D(c_, c_, 3) for _ in range(n)])


class EDFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2, bias=False):
        super(EDFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.patch_size = 8

        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.fft = nn.Parameter(torch.ones((dim, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x_dtype = x.dtype
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)

        b, c, h, w = x.shape
        h_n = (8 - h % 8) % 8
        w_n = (8 - w % 8) % 8

        x = torch.nn.functional.pad(x, (0, w_n, 0, h_n), mode='reflect')
        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                      patch2=self.patch_size)

        x=x[:,:,:h,:w]

        return x.to(x_dtype)

class TransformerEncoderLayer_EDFFN(nn.Module):
    def __init__(self, c1, cm=2048, num_heads=8, dropout=0.0, act=nn.GELU(), normalize_before=False):
        super().__init__()

        from ...utils.torch_utils import TORCH_1_9
        if not TORCH_1_9:
            raise ModuleNotFoundError(
                'TransformerEncoderLayer() requires torch>=1.9 to use nn.MultiheadAttention(batch_first=True).')
        self.ma = nn.MultiheadAttention(c1, num_heads, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.ffn = EDFFN(c1, 2.0, False)

        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.act = act
        self.normalize_before = normalize_before

    def with_pos_embed(tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class AIFI_EDFFN(TransformerEncoderLayer_EDFFN):

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0, act=nn.GELU(), normalize_before=False):
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)

    def forward(self, x):
        c, h, w = x.shape[1:]
        x = super().forward(x, pos=pos_embed.to(device=x.device, dtype=x.dtype))
        return x.permute(0, 2, 1).view([-1, c, h, w]).contiguous()


