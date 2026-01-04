import torch
import torch.nn as nn
import torch.nn.functional as F
from evaluator import StabilityEvaluator
from ConvmHC import Conv_mHCBlock


# --- 基础卷积单元 ---
class SimpleConvFn(nn.Module):
    def __init__(self, in_d):
        super().__init__()
        # 为了对比公平，Baseline 使用 in_d 通道，mHC 的每个流也使用对应的通道
        self.net = nn.Sequential(
            nn.Conv2d(in_d, in_d * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_d * 2, in_d, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


# --- Baseline 卷积网络 ---
class BaselineConvNet(nn.Module):
    def __init__(self, in_d, depth=3):
        super().__init__()
        self.layers = nn.ModuleList([SimpleConvFn(in_d) for _ in range(depth)])
    def forward(self, x):
        for layer in self.layers:
            # 标准残差：x = x + f(x)
            x = x + layer(x)
        return x


#from ConvmHC2 import Conv_mHCBlock_v2 as Conv_mHCBlock
# --- mHC 卷积网络 ---
class mHCConvNet(nn.Module):
    def __init__(self, in_d, n_streams, depth=3, evaluator=None, fn_class=None):
        super().__init__()
        # 每个流分配到的通道数
        stream_channels = in_d // n_streams
        # 构建层。注意：Conv_mHCBlock 内部已经封装了 residual 逻辑
        self.layers = nn.ModuleList([
            Conv_mHCBlock(
                n_streams=n_streams,
                channels=in_d,
                conv_fn=fn_class(stream_channels),
                evaluator=evaluator
            ) for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

from tqdm import tqdm
# ---------------------------
# Training demo
# ---------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    # --- 超参数 ---
    B = 16
    dim = 16  # 总通道数
    n_streams = 4  # 流的数量
    H, W = 64, 64  # 图片尺寸
    depth = 16  # 网络深度
    steps = 200
    lr = 1e-4
    # --- 数据生成 (针对卷积域修改) ---
    # 我们模拟一个复杂的图片到图片的转换任务
    with torch.no_grad():
        # 定义一个随机但固定的目标算子，用于产生 target y
        target_op = nn.Conv2d(dim, dim, 3, padding=1)
        torch.nn.init.orthogonal_(target_op.weight)
    def get_batch():
        # 生成 4D 图像数据: [B, C, H, W]
        x = torch.randn(B, dim, H, W)
        # target y 也是 4D 图像
        y = target_op(x).detach()
        return x, y
    # 评估器
    evaluator = StabilityEvaluator(n_streams)
    baseline = BaselineConvNet(dim, depth)
    mhc = mHCConvNet(dim, n_streams, depth, evaluator=evaluator, fn_class=SimpleConvFn)
    opt_base = torch.optim.AdamW(baseline.parameters(), lr=lr)
    opt_mhc = torch.optim.AdamW(mhc.parameters(), lr=lr)

    for step in tqdm(range(steps)) :
        x_img, y_img = get_batch()
        # -------- Baseline 训练 --------
        out_base = baseline(x_img)
        loss_base = F.mse_loss(out_base, y_img)
        opt_base.zero_grad()
        loss_base.backward()
        opt_base.step()
        # -------- mHC 训练 --------
        x_img_mhc = x_img.detach().clone()
        x_img_mhc.requires_grad_(True)
        out_mhc = mhc(x_img_mhc)
        loss_mhc = F.mse_loss(out_mhc, y_img)

        opt_mhc.zero_grad()
        loss_mhc.backward()
        opt_mhc.step()

        # 记录指标
        evaluator.log_grad_norm(mhc)
        evaluator.log_loss_gap(loss_mhc, loss_base)

        if step % 50 == 0:
            print(f"Step {step:03d} | Loss(base)={loss_base.item():.4f} Loss(mHC)={loss_mhc.item():.4f}")

    print("\n" + "=" * 30)
    print(f"Max Amax Gain : {max(evaluator.amax_gain) if evaluator.amax_gain else 'N/A'}")
    print(f"Mean Grad Gain: {sum(evaluator.grad_gain) / len(evaluator.grad_gain) if evaluator.grad_gain else 'N/A'}")

    # 打印每层的梯度范数，观察梯度消失/爆炸情况
    for k, v in evaluator.grad_norms.items():
        print(k, "mean grad norm:", sum(v)/len(v) if v else 0)

    for k, v in evaluator.grad_norms.items():
        if 'mhc' in k.lower():  # 只看 mHC 层的梯度
            print(f"{k} Avg Grad Norm: {sum(v) / len(v):.6f}")
    print(f"Loss gap: {evaluator.loss_gap[-1]:.4f}")