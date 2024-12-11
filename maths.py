import math

import torch
from torch import nn, Tensor
from einops import rearrange


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor):
	"""
		attention of q, k, v
	"""
	q, k = apply_rope(q, k, pe)
	x = nn.functional.scaled_dot_product_attention(q, k, v)
	x = rearrange(x, "B H L D -> B L (H D)")
	return x


def rope2D(grid: Tensor, dim: int, base: int) -> Tensor:
	"""
		rotary position embedding for 2-dim
		grid must has size (2, 1, H, W)
	"""
	assert dim % 4 == 0, "dim must dividable by 4 for RoPE 2D embedding"

	emb_height = rope(grid[0], dim // 2, base)      # (1, H*W, D // 2, 2, 2)
	emb_width = rope(grid[1], dim // 2, base)       # (1, H*W, D // 2, 2, 2)
	emb_pos = torch.cat([emb_height, emb_width], dim=-3)   # (1, H*W, D, 2, 2)
	return emb_pos


def rope(pos: Tensor, dim: int, base: int) -> Tensor:
	"""
		rotary position embedding for 1-dim
	"""
	assert dim % 2 == 0, "dim must dividable by 2 for RoPE embedding"

	half_dim = dim // 2
	i = torch.arange(0, half_dim, dtype=torch.float64, device=pos.device)        # (D // 2,)
	theta = 1.0 / (base ** (i / half_dim))                      # (D // 2,)
	pos = pos.reshape(-1)                                           # (N,)
	out = torch.einsum("n,d->nd", pos, theta)             # (N, D // 2)
	out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)  # (N, D // 2, 4)
	out = rearrange(out, "n d (i j)-> n d i j", i=2, j=2)   # (N, D // 2, 2, 2)
	return out.unsqueeze(0)                                     # (1, N, D // 2, 2, 2)


def apply_rope(q: Tensor, k: Tensor, pe: Tensor) -> tuple[Tensor, Tensor]:
	"""
		apply RoPE on query and key
	"""
	q_ = q.reshape(*q.shape[:-1], -1, 1, 2)
	k_ = k.reshape(*k.shape[:-1], -1, 1, 2)

	q_out = pe[..., 0] * q_[..., 0] + pe[..., 1] * q_[..., 1]
	k_out = pe[..., 0] * k_[..., 0] + pe[..., 1] * k_[..., 1]
	return q_out.reshape(*q.shape).type_as(q), k_out.reshape(*k.shape).type_as(k)


def sinusoidalPE(t: Tensor, dim: int, max_period: int = 10_000, time_factor: float = 1000.0):
	"""
		sinusoidal position embedding
	"""
	t = time_factor * t
	half_dim = dim // 2
	freqs = torch.exp(
		-math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32) / half_dim
	).to(device=t.device)

	args = t[:, None] * freqs[None]
	embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
	if dim % 2:
		embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
	if torch.is_floating_point(t):
		embedding = embedding.to(t)
	return embedding
