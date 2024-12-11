import torch
from torch import Tensor, nn

from dataclasses import dataclass
from einops import rearrange

from maths import attention


class MLPEmbedder(nn.Module):
	"""
		MLP embedder composed of 2 layer linear and one silu activation function
	"""
	def __init__(self, in_dim: int, out_dim: int):
		super(MLPEmbedder, self).__init__()
		self.in_layer = nn.Linear(in_dim, out_dim, bias=True)
		self.silu = nn.SiLU()
		self.out_layer = nn.Linear(out_dim, out_dim, bias=True)

	def forward(self, x):
		return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(nn.Module):
	"""
		root-mean-square normalization
		scale is a learnable scale parameter
	"""
	def __init__(self, dim: int):
		super(RMSNorm, self).__init__()
		self.scale = nn.Parameter(torch.ones(dim))

	def forward(self, x: Tensor) -> Tensor:
		rrms = torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-8)
		return rrms * x * self.scale


class QKNorm(nn.Module):
	"""
		RMS normalization on query and key
	"""
	def __init__(self, dim: int):
		super(QKNorm, self).__init__()
		self.query_norm = RMSNorm(dim)
		self.key_norm = RMSNorm(dim)

	def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
		q = self.query_norm(q)
		k = self.key_norm(k)
		return q.to(v), k.to(v)


class SelfAttention(nn.Module):
	"""
		Self-attention combine with q, k normalization(RMS normalization) and linear
	"""
	def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
		super(SelfAttention, self).__init__()
		self.num_heads = num_heads
		head_dim = dim // num_heads

		self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
		self.norm = QKNorm(head_dim)
		self.proj = nn.Linear(dim, dim)

	def forward(self, x: Tensor, pe: Tensor) -> Tensor:
		qkv = self.qkv(x)
		q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
		q, k = self.norm(q, k, v)
		x = attention(q, k, v, pe)
		x = self.proj(x)
		return x


@dataclass
class ModulationOut:
	scale: Tensor
	shift: Tensor
	gate: Tensor


class Modulation(nn.Module):
	"""
		Modulation for stream block and final layer, use as scale, shift and gate
		compose of silu activation function and linear
	"""
	def __init__(self, dim):
		super(Modulation, self).__init__()
		self.linear = nn.Linear(dim, 3 * dim, bias=True)

	def forward(self, vec: Tensor) -> ModulationOut:
		out = self.linear(nn.functional.silu(vec)).chunk(3, dim=-1)
		return ModulationOut(*out)


class StreamBlock(nn.Module):
	"""
		stream block as DiT block
		use timestep as modulator, divide into 3 shift, scale, gate for self-attn, cross-attn and mlp
		pe use in self-attn and cross-attn(RoPE)
		x -> layer norm -> modulation -> self-attn(residual connect with x)
		-> x -> layer norm -> modulation -> cross-attn with condition(residual connect with x)
		-> x -> layer norm -> modulation -> mlp(residual connect with x) -> return x
	"""
	def __init__(
		self,
		hidden_size,
		num_heads=8,
	):
		super(StreamBlock, self).__init__()
		self.hidden_size = hidden_size
		self.num_heads = num_heads
		head_dim = hidden_size // num_heads

		self.mod = self.adaLN_modulation = nn.Sequential(
			nn.SiLU(),
			nn.Linear(hidden_size, 9 * hidden_size, bias=True)
		)
		self.pre_norm = nn.LayerNorm(hidden_size)
		self.middle_norm = nn.LayerNorm(hidden_size)
		self.final_norm = nn.LayerNorm(hidden_size)
		self.qkv_linear = nn.Linear(hidden_size, 3 * hidden_size)
		self.qk_norm = QKNorm(head_dim)
		self.cro_attn = CrossAttention(hidden_size, num_heads)
		self.mlp = MLPEmbedder(hidden_size, hidden_size)

	def forward(self, x: Tensor, t_emb: Tensor, cond: Tensor, pe: Tensor):
		shift_msa, scale_msa, gate_msa, shift_cro, scale_cro, gate_cro, shift_mlp, scale_mlp, gate_mlp \
			= self.adaLN_modulation(t_emb).chunk(9, dim=-1)

		x_norm = self.pre_norm(x) * (1 + scale_msa) + shift_msa
		qkv = self.qkv_linear(x_norm)
		q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
		q, k = self.qk_norm(q, k, v)  # q.shape: (B, H, L, D), k.shape: (B, H, L, D)
		attn = attention(q, k, v, pe)
		x = x + gate_msa * attn
		x_norm = self.middle_norm(x) * (1 + scale_cro) + shift_cro
		cro_attn = self.cro_attn(x_norm, cond, pe)
		x = x + gate_cro * cro_attn
		x_norm = self.final_norm(x) * (1 + scale_mlp) + shift_mlp
		x = x + gate_mlp * self.mlp(x_norm)
		return x


class CrossAttention(nn.Module):
	"""
		cross attention on input and condition
		contain q, k normalization(RMS normalization)
	"""
	def __init__(self, hidden_size, num_heads):
		super(CrossAttention, self).__init__()
		self.num_heads = num_heads
		self.query = nn.Linear(hidden_size, hidden_size)
		self.kv = nn.Linear(hidden_size, 2 * hidden_size)
		head_dim = hidden_size // num_heads
		self.qk_norm = QKNorm(head_dim)

	def forward(self, x, cond, pe):
		q = self.query(x)
		q = rearrange(q, "B L (H D) -> B H L D", H=self.num_heads)
		kv = self.kv(cond)
		k, v = rearrange(kv, "B L (K H D) -> K B H L D", K=2, H=self.num_heads)
		q, k = self.qk_norm(q, k, v)
		attn = attention(q, k, v, pe)

		return attn


class FinalLayer(nn.Module):
	"""
		final layer of model, contain layer normalization, modulation
		and linear that change dimension to suitable size for unpatchify
	"""
	def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
		super(FinalLayer, self).__init__()
		self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-8)
		self.linear_final = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
		self.mod = Modulation(hidden_size)

	def forward(self, x, t):
		mod = self.mod(t)
		x = self.norm_final(x) * (1 + mod.scale) + mod.shift
		x = x + mod.gate * x
		x = self.linear_final(x)
		return x
