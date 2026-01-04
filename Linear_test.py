import torch.nn as nn
from mHC import mHCBlock
import torch
import torch.nn.functional as F
from evaluator import StabilityEvaluator
from LinearmHC import MHCLayer


# Hyperparams
class BaselineLinearNet(nn.Module):
    def __init__(self, dim, depth=3):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(depth)])

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x

class mHCLinearNet(nn.Module):
    def __init__(self, dim, n_streams, depth=3,evaluator = None):
        super().__init__()
        self.layers = nn.ModuleList([MHCLayer(dim, n_streams,evaluator = evaluator) for _ in range(depth)])
    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x

# ---------------------------
# Training demo
# ---------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    # Hyperparams
    B = 32
    dim = 64
    n_streams = 4
    depth = 16
    steps = 200
    lr = 1e-3
    # Target matrix
    target = torch.randn(dim, dim)
    def get_batch():
        x = torch.randn(B, n_streams, dim)
        y = torch.einsum("bij,jk->bik", x, target)
        return x, y


    evaluator = StabilityEvaluator(n_streams)

    # Models
    baseline = BaselineLinearNet(dim, depth)
    mhc = mHCLinearNet(dim, n_streams, depth, evaluator = evaluator)

    opt_base = torch.optim.Adam(baseline.parameters(), lr=lr)
    opt_mhc = torch.optim.Adam(mhc.parameters(), lr=lr)

    for step in range(steps):
        x, y = get_batch()
        # -------- Baseline --------
        xb = x.mean(dim=1)  # single stream
        yb = y.mean(dim=1)
        loss_base = F.mse_loss(baseline(xb), yb)
        opt_base.zero_grad()
        loss_base.backward()
        opt_base.step()
        # -------- mHC --------
        # multi-stream, 对齐 baseline
        x.requires_grad_(True)
        out_mhc = mhc(x)
        loss_mhc = F.mse_loss(out_mhc.mean(dim=1), yb)
        opt_mhc.zero_grad()
        loss_mhc.backward()
        opt_mhc.step()
        evaluator.log_grad_norm(mhc)
        evaluator.log_loss_gap(loss_mhc, loss_base)
        if step % 50 == 0:
            print(f"Step {step:03d} | Loss(base)={loss_base.item():.4f} Loss(mHC)={loss_mhc.item():.4f}")

    print("Max Amax Gain:", max(evaluator.amax_gain) if evaluator.amax_gain else "N/A")
    print("Mean Amax Gain:", sum(evaluator.amax_gain)/len(evaluator.amax_gain) if evaluator.amax_gain else "N/A")
    print("Max Grad Gain:", max(evaluator.grad_gain) if evaluator.grad_gain else "N/A")
    print("Mean Grad Gain:", sum(evaluator.grad_gain) / len(evaluator.grad_gain) if evaluator.grad_gain else "N/A")
    for k, v in evaluator.grad_norms.items():
        print(k, "mean grad norm:", sum(v)/len(v) if v else 0)
    print("Final Loss Gap:", evaluator.loss_gap[-1])


