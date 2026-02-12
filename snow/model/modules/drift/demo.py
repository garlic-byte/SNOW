import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Device selection
# - In Colab: prefer CUDA (GPU) and warn if GPU isn't enabled
# - Locally: prefer CUDA, then Apple Silicon MPS, then CPU
IN_COLAB = (
    "google.colab" in sys.modules
    or os.environ.get("COLAB_GPU") is not None
    or os.environ.get("COLAB_TPU_ADDR") is not None
)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif (not IN_COLAB) and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE} (IN_COLAB={IN_COLAB})")
if IN_COLAB and DEVICE.type != "cuda":
    print(
        "[Warning] Colab is not running with GPU enabled. Please select 'Runtime -> Change runtime type -> GPU' in the menu and re-run this cell."
    )


def drifting_loss(gen: torch.Tensor, pos: torch.Tensor, compute_drift):
    """Drifting loss: MSE(gen, stopgrad(gen + V))."""
    with torch.no_grad():
        V = compute_drift(gen, pos)
        target = (gen + V).detach()
    return F.mse_loss(gen, target)


# ============================================================
# Core: Compute Drift V and Loss (from toy_mean_drift.py)
# ============================================================

def compute_drift(gen: torch.Tensor, pos: torch.Tensor, temp: float = 0.05):
    """
    Compute drift field V with attention-based kernel.

    Args:
        gen: Generated samples [G, D]
        pos: Data samples [P, D]
        temp: Temperature for softmax kernel

    Returns:
        V: Drift vectors [G, D]
    """
    targets = torch.cat([gen, pos], dim=0)
    G = gen.shape[0]

    dist = torch.cdist(gen, targets)
    dist[:, :G].fill_diagonal_(1e6)  # mask self
    kernel = (-dist / temp).exp()  # unnormalized kernel

    normalizer = kernel.sum(dim=-1, keepdim=True) * kernel.sum(dim=-2,
                                                               keepdim=True)  # normalize along both dimensions, which we found to slightly improve performance
    normalizer = normalizer.clamp_min(1e-12).sqrt()
    normalized_kernel = kernel / normalizer

    pos_coeff = normalized_kernel[:, G:] * normalized_kernel[:, :G].sum(dim=-1, keepdim=True)
    pos_V = pos_coeff @ targets[G:]
    neg_coeff = normalized_kernel[:, :G] * normalized_kernel[:, G:].sum(dim=-1, keepdim=True)
    neg_V = neg_coeff @ targets[:G]

    return pos_V - neg_V


# Quick visualization of drift vectors
torch.manual_seed(42)

def sample_checkerboard(n: int, noise: float = 0.05, seed: int | None = None) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    b = torch.randint(0, 2, (n,), generator=g)
    i = torch.randint(0, 2, (n,), generator=g) * 2 + b
    j = torch.randint(0, 2, (n,), generator=g) * 2 + b
    u = torch.rand(n, generator=g)
    v = torch.rand(n, generator=g)
    pts = torch.stack([i + u, j + v], dim=1) - 2.0
    pts = pts / 2.0
    if noise > 0:
        pts = pts + noise * torch.randn(pts.shape, generator=g)
    return pts


def sample_swiss_roll(n: int, noise: float = 0.03, seed: int | None = None) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    u = torch.rand(n, generator=g)
    t = 0.5 * math.pi + 4.0 * math.pi * u
    pts = torch.stack([t * torch.cos(t), t * torch.sin(t)], dim=1)
    pts = pts / (pts.abs().max() + 1e-8)
    if noise > 0:
        pts = pts + noise * torch.randn(pts.shape, generator=g)
    return pts


def sample_triangle(n: int, noise: float = 0.03, seed: int | None = None) -> torch.Tensor:
    """
    生成均匀分布在单位三角形内的样本点（添加可选高斯噪声），风格对齐sample_swiss_roll。

    Args:
        n: 生成的样本数量
        noise: 高斯噪声的标准差，0表示无噪声
        seed: 随机数种子，None表示不固定种子

    Returns:
        torch.Tensor: 形状为 [n, 2] 的二维样本点矩阵，坐标归一化到[-1, 1]范围
    """
    # 初始化随机数生成器（和原函数保持一致）
    g = torch.Generator().manual_seed(seed) if seed is not None else None

    # 生成三角形内的均匀随机点（核心逻辑：二维均匀采样到三角形的转换）
    # 方法：先采样两个[0,1)的随机数u1,u2，再通过变换得到三角形内的点
    u1 = torch.rand(n, generator=g)
    u2 = torch.rand(n, generator=g)

    # 转换公式：将单位正方形的均匀采样转换为等腰直角三角形（顶点(0,0),(1,0),(0,1)）
    # 公式推导：https://math.stackexchange.com/questions/18686/uniform-random-point-in-triangle
    x = torch.sqrt(u1) * u2
    y = torch.sqrt(u1) * (1 - u2)

    # 拼接为二维坐标，并调整为对称的三角形（顶点(-1,-1),(1,-1),(0,1)），更美观
    pts = torch.stack([x * 2 - 1, y * 2 - 1], dim=1)

    # 归一化（和原函数保持一致，确保坐标范围在[-1,1]）
    pts = pts / (pts.abs().max() + 1e-8)

    # 添加高斯噪声（和原函数逻辑一致）
    if noise > 0:
        pts = pts + noise * torch.randn(pts.shape, generator=g)

    return pts


def sample_curve(n: int, noise: float = 0.03, seed: int | None = None,
                 freq: float = 2.0, amp: float = 1.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    x = torch.linspace(0, 2 * math.pi, n)
    x = x + 0.01 * torch.randn(n, generator=g)
    y = amp * torch.sin(freq * x)
    pts = torch.stack([x, y], dim=1)
    pts[:, 0] = (pts[:, 0] - pts[:, 0].min()) / (pts[:, 0].max() - pts[:, 0].min() + 1e-8) * 2 - 1
    pts[:, 1] = (pts[:, 1] - pts[:, 1].min()) / (pts[:, 1].max() - pts[:, 1].min() + 1e-8) * 2 - 1
    if noise > 0:
        pts = pts + noise * torch.randn(pts.shape, generator=g)
    return pts


# ============================================================
# Training Loop for Toy 2D
# ============================================================
from functools import partial


class MLP(nn.Module):
    """MLP: noise -> output. 3 hidden layers with SiLU."""

    def __init__(self, in_dim=32, hidden=256, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z):
        return self.net(z)


def train_toy(sampler, steps=2000, data_batch_size=2048, gen_batch_size=2048, lr=1e-3, temp=0.05,
              in_dim=32, hidden=256, plot_every=500, seed=42):
    """Train drifting model. Returns model and loss history."""
    torch.manual_seed(seed)
    model = MLP(in_dim=in_dim, hidden=hidden, out_dim=2).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    ema = None
    pbar = tqdm(range(1, steps + 1))
    for step in pbar:
        pos = sampler(data_batch_size).to(DEVICE)
        gen = model(torch.randn(gen_batch_size, in_dim, device=DEVICE))
        loss = drifting_loss(gen, pos, compute_drift=partial(compute_drift, temp=temp))

        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_history.append(loss.item())
        ema = loss.item() if ema is None else 0.96 * ema + 0.04 * loss.item()
        pbar.set_postfix(loss=f"{ema:.2e}")

        if step % plot_every == 0 or step == 1:
            with torch.no_grad():
                vis = model(torch.randn(5000, in_dim, device=DEVICE)).cpu().numpy()
                gt = sampler(5000).numpy()
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
            ax1.scatter(gt[:, 0], gt[:, 1], s=2, alpha=0.3, c='black')
            ax1.set_title('Target');
            ax1.set_aspect('equal');
            ax1.axis('off')
            ax2.scatter(vis[:, 0], vis[:, 1], s=2, alpha=0.3, c='tab:orange')
            ax2.set_title(f'Generated (step {step})');
            ax2.set_aspect('equal');
            ax2.axis('off')
            plt.tight_layout();
            plt.show()

    return model, loss_history

# Train on Swiss Roll
print("Training on Swiss Roll...")
model_swiss, loss_swiss = train_toy(
    sample_swiss_roll,
    steps=20000,
    data_batch_size=20,
    gen_batch_size=20,
    lr=1e-3,
    temp=0.05)

plt.figure(figsize=(6, 3))
plt.plot(loss_swiss, alpha=0.7)
plt.xlabel('Step'); plt.ylabel('Loss'); plt.title('Swiss Roll Loss Curve')
plt.yscale('log'); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()