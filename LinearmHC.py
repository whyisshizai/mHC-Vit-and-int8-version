import torch
import torch.nn.functional as F
import torch.nn as nn


def sinkhorn(log_alpha, n_iters=8, eps=1e-6):
    """
    log_alpha: [B, n, n]
    """
    A = torch.exp(log_alpha)

    for _ in range(n_iters):
        A = A / (A.sum(dim=-1, keepdim=True) + eps)
        A = A / (A.sum(dim=-2, keepdim=True) + eps)

    return A

class MHCMappingNet(nn.Module):
    """
    φ_l : R^{C} -> R^{n^2 + 2n}
    """
    def __init__(self, dim, n):
        super().__init__()
        self.n = n
        self.out_dim = n * n + 2 * n

        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, self.out_dim)
        )
    def forward(self, x):
        # x: [B, C]
        return self.net(x)

from mHC import mHCGenerator
class MHCLayer(nn.Module):
    def __init__(self, dim, n_streams,evaluator=None, use_sinkhorn=True):
        super().__init__()
        self.dim = dim
        self.n = n_streams
        self.use_sinkhorn = use_sinkhorn
        self.evaluator = evaluator
        self.c = dim
        self.generator = mHCGenerator(n_streams, dim)
        # 流线性层
        self.linear = nn.Linear(self.c, self.c, bias=False)

    def forward(self, x):
        """
        x: [B, n, dim]
        """
        B, n, C = x.shape
        H_pre, H_post, H_res = self.generator(x)
        if self.evaluator is not None:
            self.evaluator.log_amax_gain(H_res)
            def hook(grad):
                gain = torch.log10(grad.norm() / (x.detach().norm() + 1e-6))
                self.evaluator.log_grad_gain(gain)
            x.register_hook(hook)
        # input mapping
        x_streams = x * H_pre.unsqueeze(-1)  # broadcast to dim
        # residual mixing
        x_streams = torch.einsum("bij,bjc->bic", H_res, x_streams)
        # per-stream linear
        # split each stream
        x_split = x_streams.view(B*n, C)
        x_split = self.linear(x_split)
        x_streams = x_split.view(B, n, C)
        # output mapping
        x_streams = x_streams * H_post.unsqueeze(-1)
        return x_streams

if __name__ == '__main__':
    # Example
    B, n, C = 2, 4, 512
    x = torch.randn(B, n, C)
    block = MHCLayer(dim=C,n_streams=n)
    y = block(x)
    print(y.shape)  # [B, n, C]