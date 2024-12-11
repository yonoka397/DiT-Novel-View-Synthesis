import torch
from torch import nn, Tensor

from dataclasses import dataclass
from timm.models.vision_transformer import PatchEmbed

from layers import MLPEmbedder, StreamBlock, FinalLayer
from maths import rope2D, sinusoidalPE


@dataclass
class DiTParameter:
	input_size: int
	patch_size: int
	hidden_size: int
	in_channel: int
	out_channel: int
	num_heads: int
	base: int
	depth: int
	img_ids: Tensor


# define DiT model
class DiT_model(nn.Module):
	"""
		x -> patch emb -> x_emb
		condition -> patch emb -> cond_emb
		timesteps -> sinusoidalPE -> mlp -> t
		pe = RoPE

		x_emb, t, cond_emb -> stream block x depth -> final layer -> unpatchify
	"""
	def __init__(self, params: DiTParameter, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.out_channel = params.out_channel
		self.img_ids = params.img_ids
		self.pe = rope2D(self.img_ids, params.hidden_size // params.num_heads, params.base)

		self.patch_embedder = PatchEmbed(
			params.input_size, params.patch_size, params.in_channel, params.hidden_size, bias=True
		)  # B T D    D is hidden size
		self.condition_embedder = PatchEmbed(
			img_size=params.input_size, patch_size=params.patch_size, in_chans=params.in_channel, embed_dim=params.hidden_size, bias=True
		)
		self.time_in = MLPEmbedder(in_dim=256, out_dim=params.hidden_size)
		self.stream_blocks = nn.ModuleList(
			[
				StreamBlock(params.hidden_size, num_heads=params.num_heads)
				for _ in range(params.depth)
			]
		)
		self.final_layer = FinalLayer(params.hidden_size, params.patch_size, params.out_channel)

	def forward(self, x, timesteps, condition):
		timesteps = timesteps.to(dtype=torch.float32)  # (B, )
		t = self.time_in(sinusoidalPE(t=timesteps, dim=256)).unsqueeze(1)  # (B, 1, D)

		img = self.patch_embedder(x)  # (B, L, D)
		cond = self.condition_embedder(condition)  # (B, L, D)

		# img.shape: (B, L, D)
		# t.shape: (B, 1, D)
		# pe.shape: (1, L, D, 2, 2)
		for block in self.stream_blocks:
			img = block(img, t, cond, self.pe)  # (B, L, D)

		# img.shape: (B, L, D)
		# t.shape: (B, 1, D)
		img = self.final_layer(img, t)  # (B, L, C * (patch size) ** 2)
		img = self.unpatchify(img)  # (B, C, H, W)
		return img

	def unpatchify(self, x):
		c = self.out_channel
		p = self.patch_embedder.patch_size[0]
		h = w = int(x.shape[1] ** 0.5)
		assert h * w == x.shape[1]

		x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
		x = torch.einsum('nhwpqc->nchpwq', x)
		imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
		return imgs
