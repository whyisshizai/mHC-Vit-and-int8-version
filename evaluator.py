import torch
import torch.nn.functional as F
from collections import defaultdict


class StabilityEvaluator:
    def __init__(self, n_streams):
        self.amax_gain = []
        self.grad_gain = []
        self.amax_history = {}
        self.grad_norms = {f'layer_{i}': [] for i in range(n_streams)}
        self.loss_gap = []

    def log_loss_gap(self, loss_mhc, loss_base):
        self.loss_gap.append((loss_mhc - loss_base).item())

    def log_custom_amax(self, key, val):
        if key not in self.amax_history:
            self.amax_history[key] = []
        self.amax_history[key].append(val)

    def log_amax_gain(self, val):
        if 'default' not in self.amax_history: self.amax_history['default'] = []
        self.amax_history['default'].append(val)
    def log_amax_gain(self, H_res):
        # Amax Gain = spectral norm approximation
        with torch.no_grad():
            # 简单用 max singular value 近似
            sv = torch.linalg.svdvals(H_res)
            self.amax_gain.append(sv.max().item())

    def log_grad_gain(self, gain):
        self.grad_gain.append(gain.item())
    def log_grad_norm(self, model):
        for i, p in enumerate(model.parameters()):
            if p.grad is not None:
                self.grad_norms[f'layer_{i%len(self.grad_norms)}'].append(p.grad.norm().item())

