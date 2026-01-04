import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from evaluator import StabilityEvaluator
from ConvmHC2 import mHCPatchBlock,mHCGenerator_nStream
import numpy as np


class StandardTransformerBlock(nn.Module):
    """ 标准的 Baseline Block: Pre-Norm 结构 """

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B, L, D]
        # Attn
        res = x
        x = self.ln1(x)
        x, _ = self.attn(x, x, x)
        x = res + x
        # FFN
        res = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = res + x
        return x

class mHC_StandardTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_streams, s_len, evaluator=None, block_idx=0):
        super().__init__()
        self.n, self.s, self.d = n_streams, s_len, d_model
        self.evaluator = evaluator
        self.block_idx = block_idx
        # Attention 
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.hc1 = mHCGenerator_nStream(n_streams, s_len, d_model)
        # FFN 
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.hc2 = mHCGenerator_nStream(n_streams, s_len, d_model)

    def apply_mhc(self, x, fn, hc_gen, sub_layer_name):
        """
        核心 mHC 应用函数
        sub_layer_name: 用于在日志中区分是 'attn' 还是 'ffn'
        """
        B, n, s, d = x.shape
        x_flat = x.reshape(B, -1)
        H_pre, H_post, H_res = hc_gen(x_flat)

        # ---  Amax Gain ---
        if self.evaluator is not None:
            with torch.no_grad():
                # Forward Amax
                amax_fwd = H_res.abs().sum(dim=-1).max().item()
                # 记录格式: "BlockIdx_LayerName_fwd"
                # 注意：这里调用的是 log_custom_amax
                if hasattr(self.evaluator, 'log_custom_amax'):
                    self.evaluator.log_custom_amax(f"{self.block_idx}_{sub_layer_name}_fwd", amax_fwd)
                else:
                    self.evaluator.log_amax_gain(amax_fwd)
            # Backward Amax:
            # def bwd_hook(grad):
            #     with torch.no_grad():
            #         amax_bwd = H_res.abs().sum(dim=-2).max().item()
            #         if hasattr(self.evaluator, 'log_custom_amax'):
            #             self.evaluator.log_custom_amax(f"{self.block_idx}_{sub_layer_name}_bwd", amax_bwd)
            # x.register_hook(bwd_hook)
        # mHC 混合路径
        x_pre = x * H_pre.view(B, n, 1, 1)
        x_mix = torch.matmul(H_res, x_pre.reshape(B, n, -1)).view(B, n, s, d)

        y = fn(x_mix.reshape(B * n, s, d)).view(B, n, s, d)
        y = y * H_post.view(B, n, 1, 1)

        # DSM 矩阵 H_res 作用
        res_mix = torch.matmul(H_res, x.reshape(B, n, -1)).view(B, n, s, d)

        return res_mix + y

    def forward(self, x):
        def attn_fn(feat):
            feat = self.ln1(feat)
            out, _ = self.attn(feat, feat, feat)
            return out
        x = self.apply_mhc(x, attn_fn, self.hc1, 'attn')

        def ffn_fn(feat):
            feat = self.ln2(feat)
            return self.ffn(feat)
        x = self.apply_mhc(x, ffn_fn, self.hc2,"ffn")

        return x


class BaselinePatchNet(nn.Module):
    def __init__(self, d_model, n_streams, s_len, depth=3, n_heads=4):
        super().__init__()
        self.n, self.s, self.d = n_streams, s_len, d_model
        self.layers = nn.ModuleList([
            StandardTransformerBlock(d_model, n_heads=n_heads)
            for _ in range(depth)
        ])

    def forward(self, x):
        B, n, s, d = x.shape
        x = x.reshape(B * n, s, d)
        for layer in self.layers:
            x = layer(x)
        x = x.reshape(B, n, s, d)
        x = x.reshape(B, n * s, d)
        return x


class mHCPatchNet(nn.Module):
    def __init__(self, d_model, n_streams, s_len, depth=3, n_heads=4, evaluator=None):
        super().__init__()
        self.n, self.s, self.d = n_streams, s_len, d_model
        self.layers = nn.ModuleList([
            mHC_StandardTransformerBlock(d_model, n_heads, n_streams, s_len, evaluator=evaluator, block_idx=i)
            for i in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        B, n, s, d = x.shape
        x = x.reshape(B, n * s, d)
        return x


class TinyVLM(nn.Module):
    def __init__(self, d_model, n_streams, s_len, depth,n_head = 4,evaluator=None,mHC = True):
        super().__init__()
        if mHC :self.backbone = mHCPatchNet(d_model, n_streams, s_len, depth,n_head ,evaluator)
        else:self.backbone = BaselinePatchNet(d_model, n_streams, s_len, depth,n_head)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 384)# LLM emb_dim
        )
    def forward(self, x):
        features = self.backbone(x)
        # 取全局平均池化作为整张图的语义表示
        global_feature = features.mean(dim=1)
        out = self.head(global_feature)  # [B, 768]
        return out


def plot_mhc_analysis(evaluator, base_loss_history, mhc_loss_history, base_grad_norms, time_base, time_mhc,
                          steps_count, depth):
    plt.style.use('seaborn-v0_8-paper')
    fig, axes = plt.subplots(2, 2, figsize=(18, 11), dpi=120)
    steps = np.arange(steps_count)

    # --- (Grad Norm Stability) ---
    ax1 = axes[0, 0]
    ax1.plot(steps, base_grad_norms, label="Baseline (Standard)", color='gray', linestyle='--', alpha=0.5)

    # 获取 mHC 的梯度键并折叠数据
    mhc_keys = list(evaluator.grad_norms.keys())
    if mhc_keys:
        indices = [0, len(mhc_keys) // 2, -1]
        for idx in indices:
            key = mhc_keys[idx]
            y = np.array(evaluator.grad_norms[key])
            if len(y) > steps_count:
                y = y[:steps_count * (len(y) // steps_count)].reshape(steps_count, -1).mean(axis=1)
            ax1.plot(steps, y, label=f"mHC {key}", linewidth=1.2)

    ax1.set_title("A. Gradient Norm Stability", fontsize=13, fontweight='bold')
    ax1.set_yscale('log')
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Norm (Log Scale)")
    ax1.legend(fontsize=8, loc='lower right')

    # ---  Figure a: Amax Gain vs Layer ---
    # 这个图展示的是：在整个训练过程中，每一层的平均增益是否在 1.0 附近
    ax2 = axes[0, 1]
    if hasattr(evaluator, 'amax_history') and len(evaluator.amax_history) > 0:
        layer_indices = []
        fwd_amax_means = []
        bwd_amax_means = []
        labels = []

        # 遍历所有 Block，每个 Block 展开为 Attn 和 FFN
        for i in range(depth):
            for sub in ["attn", "ffn"]:
                f_key = f"{i}_{sub}_fwd"
                b_key = f"{i}_{sub}_bwd"
                if f_key in evaluator.amax_history:
                    fwd_amax_means.append(np.mean(evaluator.amax_history[f_key]))
                    # 如果记录了反向梯度增益
                    if b_key in evaluator.amax_history:
                        bwd_amax_means.append(np.mean(evaluator.amax_history[b_key]))
                    else:
                        bwd_amax_means.append(1.0)  # 占位
                    labels.append(f"B{i}_{sub}")

        x_axis = np.arange(len(fwd_amax_means))
        ax2.plot(x_axis, fwd_amax_means, marker='o', color='#d62728', label='Forward (Signal)')
        ax2.plot(x_axis, bwd_amax_means, marker='x', color='#1f77b4', label='Backward (Grad)')
        ax2.axhline(y=1.0, color='black', linestyle=':', alpha=0.8)

        ax2.set_title("B. Signal Propagation Dynamics (Amax Gain)", fontsize=13, fontweight='bold')
        ax2.set_xlabel("Unrolled Layer Index")
        ax2.set_ylabel("Amax Magnitude")
        ax2.set_xticks(x_axis[::2])  # 每隔一层显示标签防止拥挤
        ax2.set_xticklabels(labels[::2], rotation=45, fontsize=8)
        ax2.set_ylim(0.8, 1.2)
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, "No Amax History Found\nCheck sub_layer_name keys", ha='center')

    # --- Loss Curve ---
    ax3 = axes[1, 0]

    # 对 Loss 做一点平滑处理
    def smooth(vals):
        return np.convolve(vals, np.ones(5) / 5, mode='valid')

    ax3.plot(smooth(base_loss_history), label="Baseline Loss", color='gray', alpha=0.4)
    ax3.plot(smooth(mhc_loss_history), label="mHC Loss", color='#2ca02c', linewidth=2)

    ax3.set_title("C. VLM Alignment Convergence", fontsize=13, fontweight='bold')
    ax3.set_xlabel("Steps")
    ax3.set_ylabel("Loss (MSE / Cosine)")
    ax3.legend()

    # ---(Speed) ---
    ax4 = axes[1, 1]
    times = [time_base, time_mhc]
    bars = ax4.bar(['Baseline', 'mHC'], times, color=['#95a5a6', '#3498db'], width=0.5)

    overhead = (time_mhc / time_base - 1) * 100
    ax4.text(1, time_mhc + (time_mhc * 0.05), f"+{overhead:.1f}% Time",
             ha='center', color='red', fontweight='bold', fontsize=11)

    for bar in bars:
        h = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2., h + 0.1, f'{h:.1f}s', ha='center', va='bottom')

    ax4.set_title("D. Computational Overhead", fontsize=13, fontweight='bold')
    ax4.set_ylabel("Time (seconds)")

    plt.suptitle(f"mHC (Manifold-Constrained Hyper-Connections) vs Baseline Analysis\nDepth={depth}, n_streams={n_streams}",
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def evaluate_semantic_similarity(model, loader, projector, B, n_streams, s_len, D_MODEL):
    model.eval()
    total_cosine_sim = 0
    count = 0
    with torch.no_grad():
        for batch in tqdm(loader):
            img, target = batch['image'], batch['prompt']
            tokens = projector(img).flatten(2).transpose(1, 2)
            x_stream = tokens.view(B, n_streams, s_len, D_MODEL)
            out = model(x_stream.cuda())  # [B, 384]
            sim = F.cosine_similarity(out, target, dim=1).mean()
            total_cosine_sim += sim.item()
            count += 1
    model.train()
    return total_cosine_sim / count


from torch.utils.data import DataLoader,random_split
from data_loader import COCO_caption_Dataset
import torch
# --- 主训练循环 ---
if __name__ == "__main__":
    torch.manual_seed(0)
    IMAGE_SIZE = 224
    IN_CHANNELS = 3
    D_MODEL = 128 #emb_dim
    epoch = 2

    B = 64
    n_streams = 4
    num_head = 16
    patch_size = 8
    depth = 16
    lr = 1e-4
    num_patches = (IMAGE_SIZE // patch_size) ** 2
    s_len = num_patches // n_streams
    patch_projector = nn.Conv2d(IN_CHANNELS, D_MODEL, kernel_size=patch_size, stride=patch_size)

    # COCO_2014
    IMG_DIR = r"D:\VSCode\osteoporosis\CADDM-master\data\COCO\train2014"
    ANN_FILE = r"D:\VSCode\osteoporosis\RAG\annotations\captions_train2014.json"
    instance_file = r"D:\VSCode\osteoporosis\RAG\annotations\instances_train2014.json"
    ca_dataset = COCO_caption_Dataset(img_dir=IMG_DIR,
                                   ann_file=ANN_FILE,
                                   image_size = IMAGE_SIZE,
                                   )

    train_size = int(0.8 * len(ca_dataset))
    val_size = len(ca_dataset) - train_size
    train_ds, val_ds = random_split(ca_dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=B, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=B, shuffle=False, drop_last=True)

    # with torch.no_grad():
    #     target_op = nn.Linear(dim, dim)
    #
    # def get_batch():
    #     x_img = torch.randn(B, dim, H, W)
    #     with torch.no_grad():
    #         tokens = patchify(x_img).flatten(2).transpose(1, 2)
    #         x_stream = tokens.view(B, n_streams, s_len, dim)
    #         y_stream = target_op(x_stream).detach()
    #     return x_stream, y_stream
    print("build_model")
    evaluator = StabilityEvaluator(n_streams)
    baseline = TinyVLM(D_MODEL, n_streams, s_len, depth,num_head,mHC = False).cuda()
    mhc = TinyVLM(D_MODEL, n_streams, s_len, depth, num_head,evaluator=evaluator).cuda()
    print("perpare_work")
    opt_base = torch.optim.AdamW(baseline.parameters(), lr=lr)
    opt_mhc = torch.optim.AdamW(mhc.parameters(), lr=lr)
    base_avg_grad_history = []
    base_loss_list = []
    mhc_loss_list = []
    base_val_sims = []
    mhc_val_sims = []
    print("start_training")
    # --- me show Baseline speed---
    for epoch in range(epoch):
        start_time = time.perf_counter()
        pbar = tqdm(train_loader)
        for step in pbar:
            img, caption = step['image'], step['prompt']
            tokens = patch_projector(img).flatten(2).transpose(1, 2)
            x_stream = tokens.view(B, n_streams, s_len, D_MODEL)
            out_base = baseline(x_stream.cuda())

            loss_base = F.mse_loss(out_base, caption)
            base_loss_list.append(loss_base.item())
            opt_base.zero_grad()
            loss_base.backward()
            grads = [p.grad.norm().item() for p in baseline.parameters() if p.grad is not None]
            base_avg_grad_history.append(sum(grads) / len(grads))
            opt_base.step()
        time_base = time.perf_counter() - start_time

        base_sim = evaluate_semantic_similarity(baseline, val_loader, patch_projector, B, n_streams, s_len, D_MODEL)
        base_val_sims.append(base_sim)

        # --- you show speed ---
        start_time = time.perf_counter()
        pbar = tqdm(train_loader)
        for step in pbar:
            img, caption = step['image'], step['prompt']
            tokens = patch_projector(img).flatten(2).transpose(1, 2)
            x_stream = tokens.view(B, n_streams, s_len, D_MODEL)
            out_mhc = mhc(x_stream.cuda())
            loss_mhc = F.mse_loss(out_mhc, caption)
            mhc_loss_list.append(loss_mhc.item())
            opt_mhc.zero_grad()
            loss_mhc.backward()
            opt_mhc.step()
            evaluator.log_grad_norm(mhc)
            evaluator.log_loss_gap(loss_mhc, loss_base)
        time_mhc = time.perf_counter() - start_time
        mhc_sim = evaluate_semantic_similarity(mhc, val_loader, patch_projector, B, n_streams, s_len, D_MODEL)
        mhc_val_sims.append(mhc_sim)
        print(f"Baseline Cost: {time_base:.2f}s (每步 {time_base / len(train_loader):.4f}s)")
        print(f"mHC Cost: {time_mhc:.2f}s (每步 {time_mhc / len(train_loader):.4f}s)")
        print(f"mHC Cost: {((time_mhc / time_base) - 1) * 100:.2f}%")
        print("\n" + "=" * 30)
        if evaluator.amax_gain:
            print(f"Max Amax Gain : {max(evaluator.amax_gain):.6f}")
        if evaluator.grad_gain:
            print(f"Mean Grad Gain: {sum(evaluator.grad_gain) / len(evaluator.grad_gain):.6f}")
        print("\n[Recorded Gradient Norms]")
        for k, v in evaluator.grad_norms.items():
            if len(v) > 0:
                print(f"{k} Avg Norm: {sum(v) / len(v):.6f}")
        print(f"\nFinal Loss Gap: {evaluator.loss_gap[-1]:.6f}")
        print(f"\nEpoch {epoch+1} 总结:")
        print(f"Baseline Val Semantic Similarity: {base_sim:.4f}")
        print(f"mHC Val Semantic Similarity: {mhc_sim:.4f}")
        print(f"Epoch {epoch+1} Sim gap:  {((mhc_sim / base_sim) - 1) * 100:.2f}%")
        plot_mhc_analysis(evaluator, base_loss_list, mhc_loss_list, base_avg_grad_history, time_base, time_mhc, len(train_loader), depth)
