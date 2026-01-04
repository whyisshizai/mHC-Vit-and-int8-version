import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim, dim) * 0.02)
        self.identity = nn.Identity()

    def forward(self, x):
        return x + F.linear(x, self.weight)

class SimpleResNet(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.res1 = LinearResBlock(dim)
        self.res2 = LinearResBlock(dim)

    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        return x


if __name__ == '__main__':
    x = torch.randn(4, 3, 32, 32)
    net = SimpleResNet(dim=32)
    y = net(x)
    print(y.shape)