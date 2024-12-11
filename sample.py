import torch
from torchvision import transforms
from torchvision.utils import save_image

import argparse
import os
from PIL import Image

from diffusion.diffusion_init import create_diffusion
from model import DiT_model, DiTParameter


def main(args):
	# disable gradient
	torch.set_grad_enabled(False)

	device = "cuda" if torch.cuda.is_available() else "cpu"

	# initial model parameters
	assert args.input_size % args.patch_size == 0, "input size must dividable by patch size"
	# ------------PE of input image-------------
	grid_size = args.input_size // args.patch_size
	height_pos = torch.arange(grid_size, dtype=torch.float32, device=device)
	width_pos = torch.arange(grid_size, dtype=torch.float32, device=device)
	grid = torch.meshgrid(width_pos, height_pos)
	grid = torch.stack(grid, dim=0)
	grid = grid.reshape([2, 1, grid_size, grid_size])
	# ----------------------------------------------

	params = DiTParameter(
		input_size=args.input_size,
		patch_size=args.patch_size,
		hidden_size=args.hidden_size,
		in_channel=args.in_channel,
		out_channel=(2 * args.in_channel),
		num_heads=args.num_heads,
		base=args.base,
		depth=args.depth,
		img_ids=grid,
	)

	model = DiT_model(params).to(device)

	# download model weights
	model_weights_path = "./model_weights_Objaverse.pth"\
		if args.dataset_train == "Objaverse" else "./model_weights_MVC.pth"

	# load model weights
	state_dict = torch.load(
		model_weights_path,
		weights_only=True,
		map_location=device,
	)
	model.load_state_dict(state_dict)
	model.eval()

	# set diffusion
	diffusion = create_diffusion(diffusion_steps=1000)

	# load input image
	transform = transforms.Compose([
		transforms.Lambda(lambda img: img.convert('RGB').resize((64, 64))),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
	])
	img_path = args.image_load_path
	img = Image.open(img_path)
	img_tensor = transform(img)
	input_img = img_tensor.unsqueeze(0).to(device)

	# random normal distribution noise
	z = torch.randn_like(input_img)

	model_kwargs = dict(condition=input_img)

	# sample image
	
	final = None
	for sample in diffusion.p_sample_loop_progressive(
		model=model,
		shape=z.shape,
		noise=z,
		clip_denoised=False,
		denoised_fn=None,
		model_kwargs=model_kwargs,
		device=device,
		progress=True,
		progress_callback=args.progress_callback,
	):
		final = sample
	
	sample = final["sample"].squeeze(0)

	# save output image
	img_dir, img_name = os.path.split(args.image_load_path)
	output_path = os.path.join(img_dir, 'output.png')
	save_image(sample, output_path, normalize=True, value_range=(-1, 1))
	return output_path


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("--dataset-train", type=str, default="Objaverse")

	# sample path
	parser.add_argument("--image-load-path", type=str, default="/tmp/input.png")
	
	parser.add_argument("--progress-callback", type=str, default=None)

	# model parameter
	parser.add_argument("--input-size", type=int, default=64)
	parser.add_argument("--patch-size", type=int, default=4)
	parser.add_argument("--hidden-size", type=int, default=512)
	parser.add_argument("--in-channel", type=int, default=3)
	parser.add_argument("--num-heads", type=int, default=8)
	parser.add_argument("--base", type=float, default=10_000)
	parser.add_argument("--depth", type=int, default=8)

	args = parser.parse_args()

	main(args)
