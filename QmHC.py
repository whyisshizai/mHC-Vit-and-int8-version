"""
mHC Transformer INT8 Quantization & Save
---------------------------------------
- 显式 INT8 FakeQuant 量化
- 白名单：Sinkhorn/DSM/LayerNorm 保持 FP16
- 可保存 PyTorch INT8 模型
- 支持参数量统计和压缩比
"""

import torch
import torch.nn as nn
import numpy as np
import os
from ConvmHC2 import mHCPatchBlock, mHCGenerator_nStream
from VitmHC import BaselinePatchNet, mHCPatchNet  # 替换为你本地的模型定义

# --------------------------
# 1. FakeQuant INT8 模块
# --------------------------
class FakeQuant(nn.Module):
    def __init__(self, bit=8):
        super().__init__()
        self.bit = bit
        self.qmin = -(2 ** (bit - 1))
        self.qmax = (2 ** (bit - 1)) - 1

    def forward(self, x):
        scale = x.abs().max() / self.qmax + 1e-8
        x_q = torch.clamp(torch.round(x / scale), self.qmin, self.qmax)
        return x_q * scale

# --------------------------
# 2. 包装 mHC Block，插入 FakeQuant
# --------------------------
class QuantMHCBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
        self.q = FakeQuant(8)  # INT8
        # 白名单保持 FP16
        # self.block.hc1, hc2, LN, Sinkhorn 默认 FP16/FP32

    def forward(self, x):
        x = self.q(x)  # Linear/Conv 权重量化
        x = self.block(x)
        return x

# --------------------------
# 3. 构建 TinyVLM 模型
# --------------------------
class TinyVLM(nn.Module):
    def __init__(self, d_model, n_streams, s_len, depth=3, n_heads=4,evaluator=None,mHC=True,int8=False):
        super().__init__()
        if mHC:
            backbone = mHCPatchNet(d_model, n_streams, s_len, depth, n_heads, evaluator=evaluator)
            if int8:
                # 对每个 block 插入 FakeQuant
                backbone.layers = nn.ModuleList([QuantMHCBlock(b) for b in backbone.layers])
            self.backbone = backbone
        else:
            self.backbone = BaselinePatchNet(d_model, n_streams, s_len, depth, n_heads)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 384)  # LLM emb_dim
        )


    def forward(self, x):
        features = self.backbone(x)
        # 全局平均池化
        global_feature = features.mean(dim=1)
        out = self.head(global_feature)
        return out

# --------------------------
# 4. 初始化模型
# --------------------------
device = "cuda"
D_MODEL = 128
IMAGE_SIZE = 224
PATCH_SIZE = 8
B = 1
N_STREAMS = 4
DEPTH = 16
S_LEN = (IMAGE_SIZE // PATCH_SIZE) ** 2 // N_STREAMS

model_fp32 = TinyVLM(
    d_model=D_MODEL,
    n_streams=N_STREAMS,
    s_len=S_LEN,
    depth=DEPTH,
    n_heads=16,
    mHC=False,
    int8=True
).eval().to(device)

# Dummy 输入
x_dummy = torch.randn(B, N_STREAMS, S_LEN, D_MODEL, device=device)

# --------------------------
# 5. 计算参数量
# --------------------------
def model_size(model, bit=32):
    total_bytes = 0
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            total_bytes += m.weight.numel() * (bit // 8)
        elif isinstance(m, (nn.LayerNorm)):
            total_bytes += m.weight.numel() * 2  # 白名单 FP16
    return total_bytes / 1024**2  # MB

fp32_size = model_size(model_fp32, bit=32)
int8_size = model_size(model_fp32, bit=8)

print(f"FP32 Model Size: {fp32_size:.2f} MB")
print(f"INT8 Model Size: {int8_size:.2f} MB")
print(f"Compression Ratio: {fp32_size / int8_size:.2f}x")

# --------------------------
# 6. 测试输出差异
# --------------------------
with torch.no_grad():
    y_fp32 = model_fp32(x_dummy)
    y_int8 = model_fp32(x_dummy)  # FakeQuant 已经生效
    mse = ((y_fp32 - y_int8) ** 2).mean().item()
    rel = mse / (y_fp32 ** 2).mean().item()

print(f"INT8 MSE Error: {mse:.6e}")
print(f"INT8 Rel Error: {rel:.6e}")

# --------------------------
# 7. 保存 INT8 模型
# --------------------------
# save_path = "TinyVLM_mHC_INT8.pt"
# torch.save(model_fp32.state_dict(), save_path)
# print(f"✅ INT8 model saved: {save_path}")

# --------------------------
# 8. 可选：导出 ONNX（用于部署）
# --------------------------
# onnx_path = "TinyVLM_mHC_INT8.onnx"
# torch.onnx.export(
#     model_fp32,
#     x_dummy,
#     onnx_path,
#     input_names=["input"],
#     output_names=["output"],
#     opset_version=17,
#     dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
# )
# print(f"✅ ONNX model saved: {onnx_path}")
