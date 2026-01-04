import torch
import torch.nn as nn
import torch.nn.functional as F


def sinkhorn_knopp(logits, n_iters=20, eps=1e-6):
    # 数值保护：减去最大值防止 exp 溢出
    logits = logits - torch.max(logits, dim=-1, keepdim=True)[0]
    M = torch.exp(logits)
    for _ in range(n_iters):
        M = M / (M.sum(dim=-1, keepdim=True) + eps)
        M = M / (M.sum(dim=-2, keepdim=True) + eps)
    return M

class mHCGenerator_nStream(nn.Module):
    def __init__(self, n_streams, s_len, d_model, alpha=0.01):
        super().__init__()
        self.n = n_streams
        self.total_dim = n_streams * s_len * d_model

        # 对应你之前的 phi 实现，但现在是针对全量展平的 patch 序列
        # 注意：这里 phi_pre 产生 n 个权重，phi_post 产生 n 个权重，phi_res 产生 n*n 个权重
        # 它们每一个都是基于全量信息 [B, n*s*d] 生成的，保证了“非共享”特性
        self.phi_pre = nn.Linear(self.total_dim, n_streams, bias=False)
        self.phi_post = nn.Linear(self.total_dim, n_streams, bias=False)
        self.phi_res = nn.Linear(self.total_dim, n_streams * n_streams, bias=False)

        self.b_pre = nn.Parameter(torch.zeros(n_streams))
        self.b_post = nn.Parameter(torch.zeros(n_streams))
        self.b_res = nn.Parameter(torch.zeros(n_streams * n_streams))

        self.alpha_pre = nn.Parameter(torch.tensor(alpha))
        self.alpha_post = nn.Parameter(torch.tensor(alpha))
        self.alpha_res = nn.Parameter(torch.tensor(alpha))

        self.norm = nn.RMSNorm(self.total_dim)
        self.reset_parameters()

    def reset_parameters(self):
        # 按照 2025 mHC 论文建议的初始化
        with torch.no_grad():
            nn.init.normal_(self.phi_pre.weight, std=0.02)
            nn.init.normal_(self.phi_post.weight, std=0.02)
            nn.init.normal_(self.phi_res.weight, std=0.02)
            # 初始让 H_pre 接近 1, H_post 接近 1, H_res 接近 Identity
            self.b_pre.fill_(2.0)
            self.b_post.fill_(0.0)
            self.b_res.copy_(torch.eye(self.n).flatten() * 2.0)

    def forward(self, x_flat):
        # x_flat: [B, n*s*d]
        x_norm = self.norm(x_flat)
        pre = self.alpha_pre * self.phi_pre(x_norm) + self.b_pre
        post = self.alpha_post * self.phi_post(x_norm) + self.b_post
        res = self.alpha_res * self.phi_res(x_norm) + self.b_res
        H_pre = torch.sigmoid(pre)  # [B, n] - 每个流一个独立的 Pre 权重
        H_post = 2 * torch.sigmoid(post)  # [B, n] - 每个流一个独立的 Post 权重
        H_res = sinkhorn_knopp(res.view(-1, self.n, self.n))  # [B, n, n] - 全量混合矩阵
        return H_pre, H_post, H_res


class mHCPatchBlock(nn.Module):
    def __init__(self, n_streams, s_len, d_model, fn,evaluator=None):
        super().__init__()
        self.n = n_streams
        self.s = s_len
        self.d = d_model
        self.fn = fn
        self.evaluator=evaluator
        self.hc = mHCGenerator_nStream(n_streams, s_len, d_model)

    def forward(self, x):
        """
        x: [B, n, s, d] - 输入已经被分成了 n 个流，每个流有 s 个 patch
        """
        B, n, s, d = x.shape
        x_flat = x.reshape(B, -1)  # 全量上下文 [B, n*s*d]
        H_pre, H_post, H_res = self.hc(x_flat)
        if self.evaluator is not None:
            self.evaluator.log_amax_gain(H_res)
            def hook(grad):
                gain = torch.log10(grad.norm() / (x.detach().norm() + 1e-6))
                self.evaluator.log_grad_gain(gain)
            x.register_hook(hook)

        # 这里的 H_pre.unsqueeze(-1).unsqueeze(-1) 保证了对每个 stream(n) 的 (s, d) 整体进行缩放
        x_pre = x * H_pre.view(B, n, 1, 1)

        # Heavy function (Attention/FFN)
        y = self.fn(x_pre.reshape(B * n, s, d))
        y = y.view(B, n, s, d)

        # Post-mapping
        y_gated = y * H_post.view(B, n, 1, 1)

        # [B, n, n] @ [B, n, s*d] -> [B, n, s*d]
        x_mix_in = x.reshape(B, n, -1)
        res_mixed = torch.matmul(H_res, x_mix_in).view(B, n, s, d)
        #H_res * x + H_post * F(...)
        return res_mixed + y_gated


class PatchStreamer(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_c=3, embed_dim=128, n_streams=4):
        super().__init__()
        self.patch_size = patch_size
        self.n = n_streams
        self.d = embed_dim
        # Patchify
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        # 计算每个流的序列长度
        L = (img_size // patch_size) ** 2
        assert L % n_streams == 0
        self.s = L // n_streams
    def forward(self, x):
        # [B, C, H, W] -> [B, embed_dim, h, w] -> [B, L, embed_dim]
        tokens = self.proj(x).flatten(2).transpose(1, 2)
        # [B, L, D] -> [B, n, s, d]
        return tokens.view(-1, self.n, self.s, self.d)


# --- 实验演示 ---
if __name__ == "__main__":
    # 配置
    B, C, H, W = 16, 3, 224,224
    n_streams = 4
    embed_dim = 128
    # 准备数据
    streamer = PatchStreamer(img_size=H, patch_size=4, in_c=C, embed_dim=embed_dim, n_streams=n_streams)
    x_in = torch.randn(B, C, H, W)
    x_streams = streamer(x_in)  # [16, 4, 16, 128]


    # 定义函数 F (例如简单的 FFN)
    class SimpleFFN(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(d, d * 2), nn.GELU(), nn.Linear(d * 2, d))
        def forward(self, x): return self.net(x)
    # 初始化 mHC 块
    block = mHCPatchBlock(
        n_streams=n_streams,
        s_len=streamer.s,
        d_model=embed_dim,
        fn=SimpleFFN(embed_dim)
    )
    output = block(x_streams)
    print(f"输入流形状: {x_streams.shape}")
    print(f"输出流形状: {output.shape}")