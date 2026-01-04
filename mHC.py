import torch
import torch.nn as nn
import torch.nn.functional as F

def sinkhorn_knopp(logits, n_iters=20, eps=1e-6):
    """
    logits: [B, n, n] or [n, n]
    return: DSM
    """
    #非负
    M = torch.exp(logits)
    #开始归一化
    for _ in range(n_iters):
        M = M / (M.sum(dim=-1, keepdim=True) + eps)
        M = M / (M.sum(dim=-2, keepdim=True) + eps)
    return M.squeeze(0)

class mHCGenerator(nn.Module):
    def __init__(self, n_streams, dim,alpha = 0.01):
        super().__init__()
        self.n = n_streams
        self.C = dim
        #投影
        self.phi_pre = nn.Linear(n_streams * dim, n_streams, bias=False)
        self.phi_post = nn.Linear(n_streams * dim, n_streams, bias=False)
        self.phi_res = nn.Linear(n_streams * dim, n_streams * n_streams, bias=False)

        # 偏置
        self.b_pre = nn.Parameter(torch.zeros(n_streams))
        self.b_post = nn.Parameter(torch.zeros(n_streams))
        self.b_res = nn.Parameter(torch.zeros(n_streams * n_streams))

        # 系数 原论文在LLM上初始化是0.01
        self.alpha_pre = nn.Parameter(torch.tensor(alpha))
        self.alpha_post = nn.Parameter(torch.tensor(alpha))
        self.alpha_res = nn.Parameter(torch.tensor(alpha))

        # RMSNorm
        self.norm = nn.RMSNorm(n_streams * dim)

    def forward(self, x):
        """
        x: [B, n, C]
        """
        B, n, C = x.shape
        x_flat = x.reshape(B, n*C)
        x_norm = self.norm(x_flat)

        # --- x + alpha + bias ---
        pre = self.alpha_pre * self.phi_pre(x_norm) + self.b_pre  # [B, n]
        post = self.alpha_post * self.phi_post(x_norm) + self.b_post  # [B, n]
        res = self.alpha_res * self.phi_res(x_norm) + self.b_res  # [B, n*n]


        H_pre = torch.sigmoid(pre)
        H_post = 2 * torch.sigmoid(post)
        H_res = sinkhorn_knopp(res.view(B, n, n))

        return H_pre, H_post, H_res

class mHCBlock(nn.Module):
    def __init__(self, n_streams, dim, fn, evaluator=None):
        """
        fn: Attention / FFN
        """
        super().__init__()
        self.n = n_streams
        self.dim = dim
        self.fn = fn
        self.evaluator = evaluator
        self.hc = mHCGenerator(n_streams, dim)

    def forward(self, x):
        """
        x: [B, n_stream, C]
        """
        B, n, C = x.shape
        H_pre, H_post, H_res = self.hc(x)
        if self.evaluator is not None:
            self.evaluator.log_amax_gain(H_res)
            def hook(grad):
                gain = (grad / (x.detach() + 1e-12)).abs().max()
                self.evaluator.log_grad_gain(gain)
            x.register_hook(hook)


        # Pre-mapping
        x_pre = x * H_pre.unsqueeze(-1)
        x_streams = torch.einsum("bij,bjc->bic", H_res, x_pre)
        x_split = x_streams.view(B * n, C)
        # Heavy function (Attention / FFN)
        y = self.fn(x_split)
        y = y.view(B, n, C)
        # Post-mapping
        y = y * H_post.unsqueeze(-1)
        # Residual mixing
        res = torch.einsum("bij,bjc->bic", H_res, x)
        return res + y


# fn
class FFN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    # Example
    B, n, C = 2, 4, 512
    x = torch.randn(B, n, C)
    block = mHCBlock(n_streams=n, dim=C, fn=FFN(C))
    y = block(x)

    print(y.shape)  # [B, n, C]


