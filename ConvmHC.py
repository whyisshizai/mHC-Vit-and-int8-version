import torch
import torch.nn as nn
import torch.nn.functional as F


def sinkhorn_knopp(logits, n_iters=20, eps=1e-6):
    """
    支持batch处理的Sinkhorn-Knopp
    """
    # 保持数值稳定性
    logits = logits - torch.max(logits, dim=-1, keepdim=True)[0]
    M = torch.exp(logits)
    for _ in range(n_iters):
        M = M / (M.sum(dim=-1, keepdim=True) + eps)
        M = M / (M.sum(dim=-2, keepdim=True) + eps)
    return M


class mHCGenerator(nn.Module):
    def __init__(self, n_streams, in_channels, alpha=0.01):
        super().__init__()
        self.n = n_streams
        self.C = in_channels
        # GAP 获取全局特征
        self.gap = nn.AdaptiveAvgPool2d(1)

        # 投影
        self.phi_pre = nn.Linear(in_channels, n_streams, bias=False)
        self.phi_post = nn.Linear(in_channels, n_streams, bias=False)
        self.phi_res = nn.Linear(in_channels, n_streams * n_streams, bias=False)

        # 偏置
        self.b_pre = nn.Parameter(torch.zeros(n_streams))
        self.b_post = nn.Parameter(torch.zeros(n_streams))
        self.b_res = nn.Parameter(torch.zeros(n_streams * n_streams))

        # 比例系数
        self.alpha_pre = nn.Parameter(torch.tensor(alpha))
        self.alpha_post = nn.Parameter(torch.tensor(alpha))
        self.alpha_res = nn.Parameter(torch.tensor(alpha))

        # 卷积版通常在通道维做 Norm
        self.norm = nn.RMSNorm(in_channels)

    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        B, C, H, W = x.shape
        # 全局通道
        x_gap = self.gap(x).view(B, C)
        x_norm = self.norm(x_gap)
        # 映射参数
        pre = self.alpha_pre * self.phi_pre(x_norm) + self.b_pre
        post = self.alpha_post * self.phi_post(x_norm) + self.b_post
        res = self.alpha_res * self.phi_res(x_norm) + self.b_res

        H_pre = torch.sigmoid(pre)  # [B, n]
        H_post = 2 * torch.sigmoid(post)  # [B, n]
        H_res = sinkhorn_knopp(res.view(B, self.n, self.n))  # [B, n, n]

        return H_pre, H_post, H_res


class Conv_mHCBlock(nn.Module):
    def __init__(self, n_streams, channels, conv_fn, evaluator=None):
        """
        channels: 总通道数
        conv_fn: 卷积运算函数 (例如 nn.Conv2d 或 自定义的卷积块)
        """
        super().__init__()
        assert channels % n_streams == 0, "通道数必须能被 streams 整除"
        self.n = n_streams
        self.d = channels // n_streams  # 每个 stream 的通道数
        self.channels = channels
        self.fn = conv_fn
        self.evaluator = evaluator
        self.hc = mHCGenerator(n_streams, channels)

    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        B, C, H, W = x.shape
        H_pre, H_post, H_res = self.hc(x)

        if self.evaluator is not None:
            self.evaluator.log_amax_gain(H_res)
            # def hook(grad):
            #     gain = torch.log10(grad.norm() / (x.detach().norm() + 1e-6))
            #     self.evaluator.log_grad_gain(gain)
            # x.register_hook(hook)

        # 重塑分离 Streams: [B, n, d, H, W]
        x_split = x.view(B, self.n, self.d, H, W)
        # Pre-mapping
        # H_pre: [B, n, 1, 1, 1]
        x_pre = x_split * H_pre.view(B, self.n, 1, 1, 1)
        # 跨流: [B, n, n] @ [B, n, d*H*W]
        x_res_in = x_pre.reshape(B, self.n, -1)
        x_streams = torch.matmul(H_res, x_res_in).view(B, self.n, self.d, H, W)
        # n 合并到 batch
        x_conv_in = x_streams.reshape(B * self.n, self.d, H, W)
        y_conv_out = self.fn(x_conv_in)
        y = y_conv_out.view(B, self.n, self.d, H, W)
        # Post-mapping
        y = y * H_post.view(B, self.n, 1, 1, 1)
        # Residual mixing
        res_in = x_split.reshape(B, self.n, -1)
        res_mixed = torch.matmul(H_res, res_in).view(B, self.n, self.d, H, W)
        # 原始卷积维度
        return (res_mixed + y).reshape(B, C, H, W)

#示例Block
class SimpleConvFn(nn.Module):
    def __init__(self, in_d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_d, in_d * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_d * 2, in_d, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    # 模拟卷积输入
    B, C, H, W = 2, 64, 32, 32
    n_streams = 4
    x = torch.randn(B, C, H, W)

    # 构建 Conv-mHC 块
    # 注意：fn 输入的通道数应该是 channels // n_streams
    block = Conv_mHCBlock(
        n_streams=n_streams,
        channels=C,
        conv_fn=SimpleConvFn(C // n_streams)
    )

    y = block(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")  # 保持 [2, 64, 32, 32]