# mHC: mHC and mHC-Vit (Unofficial Paper Reproduction For Personal Reference)
![alt text](https://img.shields.io/badge/arXiv-2512.24880-b31b1b.svg)
![alt text](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)


<img width="1811" height="841" alt="Image_1767514880010_0" src="https://github.com/user-attachments/assets/54db1c41-c62d-4d48-8710-85513cc026a4" />


This repository contains a faithful reproduction and extension of the mHC (Multi-Head Contraction / Manifold-constrained Heavy Ball) architecture, inspired by the optimization breakthroughs in recent large-scale model training (e.g., DeepSeek-V3 context).
The project focuses on ensuring numerical stability in extremely deep Vision Transformers (ViT) and providing a robust path for INT8 deployment.
# 1. Project Structure
code
Bash
mHC

├── ConvmHC.py          # Early mHC convolution experiments

├── ConvmHC2.py         # Final mHCGenerator / mHCPatchBlock (Paper-aligned)

├── LinearmHC.py        # Linear mHC baseline components

├── LinearRes.py        # Standard residual linear baseline

├── mHC.py              # Core Manifold-constrained Heavy Ball logic

├── evaluator.py        # StabilityEvaluator (Amax / GradNorm / Loss metrics)

├── data_loader.py      # COCO Caption / VLM Dataset utilities

├── Conv_test.py        # Convolutional mHC experiments

├── Linear_test.py      # Linear stability benchmarks

├── Vit_test.py         # ViT + mHC multi-modal alignment (VLM)

├── mHC_Full_Analysis.png # Stability analysis visualization

└── README.md


# 2. Core Design Principles
- Paper-strict non-shared gating (`H_pre`, `H_post`, `H_res`)
- Sinkhorn-Knopp normalization for residual mixing
- Identity-biased initialization
- RMSNorm-based global conditioning
- No heuristic simplifications
RMSNorm Conditioning: Global conditioning based on reordered RMSNorm to minimize computational latency while maintaining mathematical equivalence to the paper.
# 3. Stability Metrics & Analysis
The implementation reproduces the core metrics reported in the original research:
Forward Amax Gain: Ensure it stays ≈1.0 .
Backward Gradient Gain: Using hooks to track the column sum of H_res during backpropagation, verifying isometry in gradient flow.
Layer-wise Gradient Norm: Visualizing the "flat" gradient distribution across 16+ unrolled layers (Attn & FFN).
All metrics are automatically logged and visualized into a 4-panel analysis report.
# 4. Vision Transformer Integration
Spatial Patch Streaming: Images are "patchified" and mapped into n independent streams.
Unrolled Block Design: Each standard Transformer block is unrolled into two independent mHC-constrained sub-layers (Attention and FFN).
VLM Alignment: Tested on COCO-style image-to-text semantic alignment, forcing the model to aggregate spatial streams into a single semantic vector (compatible with Sentence-BERT embeddings).
# 5. INT8 Quantization & Deployment
To avoid the well-known incompatibilities of RMSNorm and Sinkhorn iterations with TensorRT/ONNX (especially regarding dynamic shapes and custom ops), this repository advocates for PyTorch-native INT8 dynamic quantization.
##5.1 Quantize Model

```Python 
from torch.ao.quantization import quantize_dynamic
import torch.nn as nn

## Load your trained mHC model
model_fp32.eval()

## Quantize linear layers to INT8
model_int8 = quantize_dynamic(
    model_fp32, 
    {nn.Linear}, 
    dtype=torch.qint8
)
```

## 5.2 Performance Comparison

Precision	Model Size	Notes
FP32	    ~12 MB	    Reference precision

INT8	    ~3 MB	    4x compression with negligible accuracy drop
# 6. Environment & Requirements
Python ≥ 3.9
PyTorch ≥ 2.1
Matplotlib & tqdm
CUDA recommended for training; INT8 quantization works efficiently on CPU for inference.
# 7. Citation
If you use this reproduction in your research, please cite the original mHC paper:
code url={https://arxiv.org/pdf/2512.24880}
Note: This codebase prioritizes numerical correctness over heuristic speedups. All architectural deviations from the paper were explicitly rejected to ensure high-fidelity research artifacts.
