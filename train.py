import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

import argparse
import logging
import os
from PIL import Image
from time import time

from diffusion.diffusion_init import create_diffusion
from model import DiTParameter, DiT_model


def main(args):
	assert torch.cuda.is_available(), "must have GPU for training"
	device = torch.device("cuda")

	# initial logger
	logging.basicConfig(
		level=logging.INFO,
		format='[\033[34m%(asctime)s\033[0m] %(message)s',
		datefmt='%Y-%m-%d %H:%M:%S',
		handlers=[logging.StreamHandler(), logging.FileHandler(f"./log.txt")]
	)
	logger = logging.getLogger(__name__)

	# initial model parameters
	assert args.input_size % args.patch_size == 0, "input size must dividable by patch size"
	# ------------PE of input image-------------
	"""
		first dim is [[0, 0, ..., 0], [1, 1, ..., 1], ..., [n-1, n-1, ..., n-1]]
		second dim is [[0, 1, ..., n-1], [0, 1, ..., n-1], ..., [0, 1, ..., n-1]]
		first dim is PE for height
		second dim is PE for width
	"""
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

	start_epoch = args.start_epoch

	model = DiT_model(params).to(device)

	diffusion = create_diffusion()

	if start_epoch != 0:
		state_dict = torch.load(f"./model_weights_epoch_{start_epoch}.pth", weights_only=True)
		model.load_state_dict(state_dict)

	opt = torch.optim.Adam(model.parameters(), lr=(2.5 * 1e-4), weight_decay=0)
	if start_epoch != 0:
		opt_state = torch.load(f"./optimizer_state_epoch_{start_epoch}.pth", weights_only=True)
		opt.load_state_dict(opt_state)

	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=2)

	datapath = args.datapath

	transform = transforms.Compose([
		# if all images are preprocess to suitable size, there is no need to resize images
		# transforms.Lambda(lambda img: img.resize((args.input_size, args.input_size))),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
	])

	# default train on Objaverse dataset, official website: https://objaverse.allenai.org/objaverse-1.0
	# change to CustomImageFolderMVC for training on MVC dataset,
	# official website: https://github.com/MVC-Datasets/MVC
	dataset = CustomImageFolderObjaverse(datapath, transform=transform)

	loader = DataLoader(
		dataset,
		batch_size=args.batch_size,
		shuffle=True,
	)
	logger.info(f"loader length: {len(loader)}")

	model.train()

	train_steps = 0
	log_steps = 0
	running_loss = 0
	start_time = time()

	losses = 0.0

	# start training model
	for epoch in range(start_epoch, args.epochs):

		for x, y in loader:
			x = x.to(device)
			y = y.to(device)

			t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)

			# let x be condition(original image), y be target image
			model_kwargs = dict(condition=x)
			loss_dict = diffusion.training_losses(model, y, t, model_kwargs=model_kwargs)
			loss = loss_dict["loss"].mean()
			opt.zero_grad()
			loss.backward()
			opt.step()

			running_loss += loss.item()
			log_steps += 1
			train_steps += 1

			losses += loss.item()

			# log training loss every n steps
			if train_steps % args.log_every == 0:
				end_time = time()
				avg_loss = torch.tensor(running_loss / log_steps, device=device)
				logger.info(
					f"epoch={epoch + 1}, step={train_steps}/{len(loader)} Train Loss: {avg_loss:.10f} Time cost: {end_time - start_time}")
				running_loss = 0
				log_steps = 0
				start_time = time()

		# log training loss of epoch
		scheduler.step(losses / train_steps)
		logger.info(f"epoch={epoch + 1} Train Loss: {losses / train_steps:.10f}")
		losses = 0
		train_steps = 0

		# save model and optimizer state every n epochs
		if (epoch + 1) % args.state_save_every == 0:
			torch.save(model.state_dict(), f'model_weights_epoch_{epoch + 1}.pth')
			torch.save(opt.state_dict(), f'optimizer_state_epoch_{epoch + 1}.pth')

	# end of training and save final state
	model.eval()
	torch.save(model.state_dict(), 'model_weights.pth')
	torch.save(opt.state_dict(), 'optimizer_state.pth')


class CustomImageFolderObjaverse(Dataset):
	def __init__(self, root, transform=None):
		self.dataset = datasets.ImageFolder(root=root, transform=transform)
		self.data = self._organize_data()
		self.transform = transform

	def _organize_data(self):
		# every 3D model have 6 views, 000.png -> horizontal rotate 60 degree -> 001.png
		#  -> horizontal rotate 60 degree -> 002.png -> ... -> 005.png -> horizontal rotate 60 degree -> 000.png
		# one model can have 6 pairs data that each pair is horizontal rotate 60 between pair
		data = []
		for class_idx in range(len(self.dataset.classes)):
			class_folder = self.dataset.classes[class_idx]
			class_images = [img[0] for img in self.dataset.samples if class_folder in img[0]]
			class_images.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
			if len(class_images) >= 6:
				for i in range(5):
					x_image = class_images[i]
					y_image = class_images[i + 1]
					data.append((x_image, y_image))
				x_image = class_images[5]
				y_image = class_images[0]
				data.append((x_image, y_image))
			else:
				print(f"Skipping class '{class_folder}' due to insufficient images.")
		print(f"Total groups: {len(data)}")
		return data

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		pair = self.data[index]
		try:
			x_image = Image.open(pair[0]).convert("RGB")
			y_image = Image.open(pair[1]).convert("RGB")
		except Exception as e:
			print(f"Error loading image pair {pair}: {e}")
			raise e
		if self.transform:
			x_image = self.transform(x_image)
			y_image = self.transform(y_image)
		return x_image, y_image


class CustomImageFolderMVC(Dataset):
	def __init__(self, root, transform=None):
		self.dataset = datasets.ImageFolder(root=root, transform=transform)
		self.data = self._organize_data()
		self.transform = transform

	def _organize_data(self):
		# use 1.jpg and 3.jpg as a pair which is the most relative to the horizontal rotate 60 degree
		data = []
		for class_idx in range(len(self.dataset.classes)):
			class_folder = self.dataset.classes[class_idx]
			class_images = [img[0] for img in self.dataset.samples if class_folder in img[0]]

			# check that there are 1.jpg and 3.jpg
			has_1jpg = any(os.path.basename(image) == '1.jpg' for image in class_images)
			has_3jpg = any(os.path.basename(image) == '3.jpg' for image in class_images)

			if has_1jpg and has_3jpg:
				x_image = next(image for image in class_images if os.path.basename(image) == '1.jpg')
				y_image = next(image for image in class_images if os.path.basename(image) == '3.jpg')
				data.append((x_image, y_image))
			else:
				print(f"Skipping class '{class_folder}' due to missing 1.jpg or 3.jpg.")

		print(f"Total pairs: {len(data)}")
		return data

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		pair = self.data[index]
		try:
			x_image = Image.open(pair[0]).convert("RGB")
			y_image = Image.open(pair[1]).convert("RGB")
		except Exception as e:
			print(f"Error loading image pair {pair}: {e}")
			raise e
		if self.transform:
			x_image = self.transform(x_image)
			y_image = self.transform(y_image)
		return x_image, y_image


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("--start-epoch", type=int, default=0)
	parser.add_argument("--epochs", type=int, default=500)
	parser.add_argument("--batch-size", type=int, default=64)
	parser.add_argument("--input-size", type=int, default=64)
	parser.add_argument("--patch-size", type=int, default=4)
	parser.add_argument("--hidden-size", type=int, default=512)
	parser.add_argument("--in-channel", type=int, default=3)
	parser.add_argument("--num-heads", type=int, default=8)
	parser.add_argument("--base", type=float, default=10_000)
	parser.add_argument("--depth", type=int, default=8)
	parser.add_argument("--datapath", type=str, default="/home/learningcorner1/py_project/views_64/")
	parser.add_argument("--log-every", type=int, default=100)  # default log every 100 batch
	parser.add_argument("--state-save-every", type=int, default=1)
	args = parser.parse_args()

	main(args)
